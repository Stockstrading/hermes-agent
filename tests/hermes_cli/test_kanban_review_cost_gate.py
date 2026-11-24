"""Focused tests for the review cost gate, round-cap override, respawn-guard PR
dedup, Rule 3 findings blocking, and review-head-unchanged dedupe ported from
PR #3 (old oversized PR) into this scoped recut.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _insert_verdict_run(conn, task_id, profile, *, started_at, ended_at, outcome="completed", metadata=None):
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at, metadata) "
            "VALUES (?, ?, 'done', ?, ?, ?, ?)",
            (task_id, profile, outcome, started_at, ended_at, kb._json_or_null(metadata) if metadata else None),
        )


# ---------------------------------------------------------------------------
# (a) _review_cost_gate: round cap + per-PR override
# ---------------------------------------------------------------------------


def test_review_cost_gate_blocks_after_default_round_cap(kanban_home: Path) -> None:
    """A PR with no per-PR override hits the global default cap after
    ``_REVIEW_ROUND_CAP`` verdict runs."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Review strategy-api#99999 for release", assignee="reviewer")
        now = int(time.time())
        for i in range(kbd._REVIEW_ROUND_CAP):
            other = kb.create_task(conn, title="Review strategy-api#99999 for release", assignee="reviewer")
            _insert_verdict_run(
                conn, other, "reviewer",
                started_at=now - 10000 + i * 100, ended_at=now - 10000 + i * 100 + 200,
            )
        reason = kbd._review_cost_gate(conn, tid, "reviewer", dry_run=True)
        assert reason == "review_round_cap"


def test_review_cost_gate_honours_higher_per_pr_override(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: a ``_REVIEW_ROUND_CAP_BY_PR`` entry HIGHER than the
    default cap must actually raise the effective cap for that PR, not get
    silently reset back to the default ('the overwrite bug must not return')."""
    pr_key = "strategy-api#88888"
    higher_cap = kbd._REVIEW_ROUND_CAP + 3
    monkeypatch.setitem(kbd._REVIEW_ROUND_CAP_BY_PR, pr_key, higher_cap)

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=f"Review {pr_key} for release", assignee="reviewer")
        now = int(time.time())
        # One more round than the DEFAULT cap, but still under the override.
        rounds = kbd._REVIEW_ROUND_CAP + 1
        assert rounds < higher_cap
        for i in range(rounds):
            other = kb.create_task(conn, title=f"Review {pr_key} for release", assignee="reviewer")
            _insert_verdict_run(
                conn, other, "reviewer",
                started_at=now - 10000 + i * 100, ended_at=now - 10000 + i * 100 + 200,
            )
        # Must NOT be capped yet: the override (higher_cap) must be honoured,
        # not silently reset to the default cap that these `rounds` already exceed.
        reason = kbd._review_cost_gate(conn, tid, "reviewer", dry_run=True)
        assert reason is None


# ---------------------------------------------------------------------------
# (b) check_respawn_guard PR-dedup
# ---------------------------------------------------------------------------


def test_respawn_guard_not_rearmed_by_comment_repeating_known_pr(kanban_home: Path) -> None:
    """A card whose title already names ``#50`` is not re-guarded merely
    because a comment repeats the same PR's URL."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Review PR #50", assignee="worker")
        kb.add_comment(conn, tid, author="worker", body="See https://github.com/example/repo/pull/50 for context.")
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


def test_respawn_guard_rearmed_by_comment_naming_different_pr(kanban_home: Path) -> None:
    """A comment naming a genuinely different PR (not already named by the
    card) arms the active-PR guard."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Review PR #50", assignee="worker")
        kb.add_comment(conn, tid, author="worker", body="Opened https://github.com/example/repo/pull/107 instead.")
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


# ---------------------------------------------------------------------------
# (c) should_block_on_findings
# ---------------------------------------------------------------------------


def test_should_block_on_findings_only_blocks_major_or_higher() -> None:
    assert kbd.should_block_on_findings([{"severity": "MAJOR"}]) is True
    assert kbd.should_block_on_findings([{"severity": "CRITICAL"}]) is True
    assert kbd.should_block_on_findings([{"severity": "BLOCKER"}]) is True
    assert kbd.should_block_on_findings([{"severity": "MINOR"}]) is False
    assert kbd.should_block_on_findings([{"severity": "NIT"}]) is False
    assert kbd.should_block_on_findings([]) is False


def test_should_block_on_findings_plain_string_form() -> None:
    assert kbd.should_block_on_findings(["MAJOR: file.py:10 missing check"]) is True
    assert kbd.should_block_on_findings(["MINOR: file.py:10 style nit"]) is False
    assert kbd.should_block_on_findings(["[NIT] trivial"]) is False


# ---------------------------------------------------------------------------
# (d) _review_head_unchanged
# ---------------------------------------------------------------------------


def test_review_head_unchanged_false_with_no_prior_verdict(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        unchanged, prior = kbd._review_head_unchanged(conn, tid, "reviewer")
        assert (unchanged, prior) == (False, None)


def test_review_head_unchanged_false_when_metadata_lacks_head_sha(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        now = int(time.time())
        _insert_verdict_run(conn, tid, "reviewer", started_at=now - 500, ended_at=now - 400, metadata={})
        unchanged, prior = kbd._review_head_unchanged(conn, tid, "reviewer")
        assert (unchanged, prior) == (False, None)


def test_review_head_unchanged_false_when_new_comment_after_verdict(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_cli.kanban_pr_acceptance.fetch_pr_head_sha",
        lambda contract, assignee=None: "shaABC",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        kb.add_comment(conn, tid, author="worker", body="Opened https://github.com/example/repo/pull/5")
        now = int(time.time())
        _insert_verdict_run(
            conn, tid, "reviewer", started_at=now - 500, ended_at=now - 400,
            metadata={"head_sha": "shaABC"},
        )
        # A comment strictly AFTER the verdict closed.
        kb.add_comment(conn, tid, author="human", body="one more thing to check")
        unchanged, prior = kbd._review_head_unchanged(conn, tid, "reviewer")
        assert (unchanged, prior) == (False, None)


def test_review_head_unchanged_true_when_sha_matches_and_no_later_comment(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hermes_cli.kanban_pr_acceptance.fetch_pr_head_sha",
        lambda contract, assignee=None: "shaABC",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        # Comment (PR link) precedes the verdict; the verdict's ended_at is
        # stamped AFTER it (future-dated here, same as a real wall-clock run
        # that lands after the comment) so there is no comment strictly after
        # the verdict closed.
        kb.add_comment(conn, tid, author="worker", body="Opened https://github.com/example/repo/pull/5")
        now = int(time.time())
        _insert_verdict_run(
            conn, tid, "reviewer", started_at=now, ended_at=now + 500,
            metadata={"head_sha": "shaABC"},
        )
        unchanged, prior = kbd._review_head_unchanged(conn, tid, "reviewer")
        assert unchanged is True
        assert prior is not None
        assert prior["metadata"].get("head_sha") == "shaABC"
