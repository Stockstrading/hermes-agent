"""Round-cap counting and Rule 3 prior-round selection (card t_13787972).

PR#1 release-gate MAJOR #4 said a handoff (``review_requested``) or a block could
consume a round-cap slot and be fed to Rule 3 as "the prior round". The cap keeps
counting every paid review run (live data: most verdicts come from standalone
review cards claimed from ``ready``), and Rule 3 now skips a newer non-verdict run
that stored no findings.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _run(conn, task_id, *, outcome, start, duration=300, metadata=None, profile="reviewer"):
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at, metadata) "
            "VALUES (?, ?, 'done', ?, ?, ?, ?)",
            (task_id, profile, outcome, start, start + duration,
             kb._json_or_null(metadata) if metadata else None),
        )


def _review_card(conn, pr):
    return kb.create_task(conn, title=f"Review {pr}: change", assignee="reviewer")


def test_standalone_review_cards_count_every_paid_outcome_toward_cap(kanban_home: Path) -> None:
    """Runs on standalone review cards (claimed from ready, no ``source_status:
    review``) that ended completed / changes_requested / blocked / review_requested
    all count: the cap is reached after ``_REVIEW_ROUND_CAP`` of them."""
    pr = "candle-api#77701"
    outcomes = ["completed", "blocked", "review_requested", "changes_requested"]
    assert len(outcomes) >= kbd._REVIEW_ROUND_CAP
    with kbc.connect() as conn:
        tid = _review_card(conn, pr)
        now = int(time.time())
        for i, outcome in enumerate(outcomes[: kbd._REVIEW_ROUND_CAP]):
            _run(conn, _review_card(conn, pr), outcome=outcome, start=now - 10000 + i * 1000)
        assert len(kbd._review_verdict_runs(conn, "reviewer", {pr})) == kbd._REVIEW_ROUND_CAP
        assert kbd._review_cost_gate(conn, tid, "reviewer", dry_run=True) == "review_round_cap"


def test_short_runs_and_non_review_outcomes_do_not_count(kanban_home: Path) -> None:
    """Crashes, reclaims, rate limits and sub-``_REVIEW_MIN_RUN_SECONDS`` runs are not
    paid rounds, so they never reach the cap."""
    pr = "candle-api#77702"
    with kbc.connect() as conn:
        tid = _review_card(conn, pr)
        now = int(time.time())
        for i, outcome in enumerate(["crashed", "reclaimed", "rate_limited", "scheduled"]):
            _run(conn, _review_card(conn, pr), outcome=outcome, start=now - 10000 + i * 1000)
        for i in range(kbd._REVIEW_ROUND_CAP):
            _run(conn, _review_card(conn, pr), outcome="completed", start=now - 5000 + i * 100,
                 duration=kbd._REVIEW_MIN_RUN_SECONDS - 1)
        assert kbd._review_verdict_runs(conn, "reviewer", {pr}) == []
        assert kbd._review_cost_gate(conn, tid, "reviewer", dry_run=True) is None


def test_delta_context_skips_newer_block_without_findings(kanban_home: Path) -> None:
    """MAJOR #4: a newer blocked run with no metadata must not replace the prior
    verdict -- the delta prompt keeps that verdict's reviewed SHA and finding."""
    pr = "candle-api#77703"
    with kbc.connect() as conn:
        tid = _review_card(conn, pr)
        now = int(time.time())
        _run(conn, _review_card(conn, pr), outcome="changes_requested", start=now - 9000, metadata={
            "reviewed_sha": SHA_A,
            "findings": [{"id": "F1", "severity": "MAJOR", "file": "x.py", "line": 3, "status": "open"}],
        })
        _run(conn, _review_card(conn, pr), outcome="blocked", start=now - 5000)
        _run(conn, _review_card(conn, pr), outcome="review_requested", start=now - 3000)
        ctx = kbd.build_round_delta_context(conn, tid, "reviewer")
        assert ctx is not None
        assert f"Diff since reviewed SHA: {SHA_A}" in ctx
        assert "[MAJOR] x.py:3 F1" in ctx
        assert "No prior findings were recorded." not in ctx


def test_delta_context_uses_newer_block_that_stored_findings(kanban_home: Path) -> None:
    """A blocked run that DID store Rule 3 data ("REQUEST CHANGES posted, waiting")
    is a verdict and wins over the older one."""
    pr = "candle-api#77704"
    with kbc.connect() as conn:
        tid = _review_card(conn, pr)
        now = int(time.time())
        _run(conn, _review_card(conn, pr), outcome="completed", start=now - 9000,
             metadata={"reviewed_sha": SHA_A})
        _run(conn, _review_card(conn, pr), outcome="blocked", start=now - 5000,
             metadata={"reviewed_sha": SHA_B, "findings": ["MAJOR: y.py:9 missing guard"]})
        ctx = kbd.build_round_delta_context(conn, tid, "reviewer")
        assert ctx is not None
        assert f"Diff since reviewed SHA: {SHA_B}" in ctx
        assert "- MAJOR: y.py:9 missing guard" in ctx


def test_delta_context_is_none_when_no_prior_run_was_a_verdict(kanban_home: Path) -> None:
    """Only handoffs/blocks without findings so far: do a normal full review
    instead of a delta against nothing."""
    pr = "candle-api#77705"
    with kbc.connect() as conn:
        tid = _review_card(conn, pr)
        now = int(time.time())
        _run(conn, _review_card(conn, pr), outcome="blocked", start=now - 9000)
        _run(conn, _review_card(conn, pr), outcome="review_requested", start=now - 5000)
        assert len(kbd._review_verdict_runs(conn, "reviewer", {pr})) == 2
        assert kbd.build_round_delta_context(conn, tid, "reviewer") is None
