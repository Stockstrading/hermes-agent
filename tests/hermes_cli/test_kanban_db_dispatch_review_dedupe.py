"""Tests for ``hermes_cli.kanban_db_dispatch``'s review-lane dedupe guard.

``check_respawn_guard`` never guards the review lane by design (it is the
handoff's INPUT, not a duplicate-work signal). ``_review_head_unchanged`` /
``_replay_review_dedupe_verdict`` are the dedicated check that sits beside
it: when a PR's head SHA and findings are unchanged since the reviewer's
last verdict, replay that verdict instead of spawning a fresh reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_pr_acceptance as pr_acceptance

SHA_A = "a" * 40
SHA_B = "b" * 40
PR_COMMENT = "Opened https://github.com/example/repo/pull/77 for review."


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _review_dispatch_always_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here dispatches the review lane against real profiles."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True, "review_dedupe": True}},
    )


def _events(conn, task_id, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    out = [(r["kind"], json.loads(r["payload"]) if r["payload"] else None) for r in rows]
    if kind is not None:
        out = [e for e in out if e[0] == kind]
    return out


def _latest_run(conn, task_id):
    return conn.execute(
        "SELECT profile, outcome, summary, metadata FROM task_runs "
        "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()


def _build_second_round_review_card(conn, *, changes_requested_head_sha: str) -> str:
    """A task that already went through one review round (``changes_requested``
    at ``changes_requested_head_sha``) and is now back in the review column
    for a second look -- the exact situation ``_review_head_unchanged`` must
    judge.
    """
    task_id = kb.create_task(conn, title="feature work", assignee="worker")
    kb.add_comment(conn, task_id, author="worker", body=PR_COMMENT)
    impl_1 = kb.claim_task(conn, task_id)
    assert kb.request_review(
        conn, task_id, summary="Implementation v1, see PR.",
        reviewer="reviewer", expected_run_id=impl_1.current_run_id,
    )
    review_1 = kb.claim_review_task(conn, task_id)
    assert review_1 is not None
    ok, implementer = kb.request_changes(
        conn, task_id, reason="Please add a regression test.",
        expected_run_id=review_1.current_run_id,
        metadata={"head_sha": changes_requested_head_sha, "review_outcome": "changes_requested"},
    )
    assert ok is True and implementer == "worker"

    impl_2 = kb.claim_task(conn, task_id)
    assert kb.request_review(
        conn, task_id, summary="Added the regression test.",
        expected_run_id=impl_2.current_run_id,
    )
    task = kb.get_task(conn, task_id)
    assert task.status == "review"
    assert task.assignee == "reviewer"
    return task_id


def test_same_head_no_new_findings_replays_prior_verdict_without_spawning(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatching a review-column card whose PR head SHA is unchanged since
    the reviewer's last verdict must NOT spawn a fresh reviewer: it replays
    the prior verdict as a synthesized run and moves the card on."""
    monkeypatch.setattr(pr_acceptance, "fetch_pr_head_sha", lambda *a, **k: SHA_A)
    spawned: list[str] = []

    def spawn(task, workspace):
        spawned.append(task.id)

    with kbc.connect() as conn:
        task_id = _build_second_round_review_card(conn, changes_requested_head_sha=SHA_A)

        result = kbd.dispatch_once(conn, spawn_fn=spawn)

        # No new reviewer worker was spawned for this card.
        assert task_id not in spawned
        assert task_id not in [s[0] for s in result.spawned]

        # The replayed verdict mirrors the prior changes_requested outcome:
        # the card is handed back to the implementer, not left stuck in review.
        after = kb.get_task(conn, task_id)
        assert after.status == "ready"
        assert after.assignee == "worker"

        replayed = _events(conn, task_id, kind="review_dedupe_replayed")
        assert len(replayed) == 1
        assert replayed[0][1]["head_sha"] == SHA_A
        assert replayed[0][1]["prior_run_outcome"] == "changes_requested"

        run = _latest_run(conn, task_id)
        assert run["outcome"] == "changes_requested"
        assert run["profile"] == "reviewer"
        assert run["summary"].startswith("[replayed, head unchanged]")


def test_head_moved_since_last_verdict_spawns_for_real(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different live head SHA than the one on the prior verdict must fall
    through to a real reviewer spawn -- never silently replay a stale verdict."""
    monkeypatch.setattr(pr_acceptance, "fetch_pr_head_sha", lambda *a, **k: SHA_B)
    spawned: list[str] = []

    def spawn(task, workspace):
        spawned.append(task.id)

    with kbc.connect() as conn:
        task_id = _build_second_round_review_card(conn, changes_requested_head_sha=SHA_A)

        result = kbd.dispatch_once(conn, spawn_fn=spawn)

        assert task_id in spawned
        assert task_id in [s[0] for s in result.spawned]
        assert _events(conn, task_id, kind="review_dedupe_replayed") == []
        # A real claim happened -- the card is now running under the reviewer.
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.assignee == "reviewer"


def test_no_prior_verdict_spawns_for_real(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first-time review handoff has no prior ``completed``/``changes_requested``
    run for the reviewer profile -- it must always spawn, with no live GraphQL
    lookup at all (asserted via a stub that raises if called)."""
    def _unexpected(*a, **k):
        raise AssertionError("fetch_pr_head_sha must not be called with no prior verdict")

    monkeypatch.setattr(pr_acceptance, "fetch_pr_head_sha", _unexpected)
    spawned: list[str] = []

    def spawn(task, workspace):
        spawned.append(task.id)

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="feature work", assignee="worker")
        kb.add_comment(conn, task_id, author="worker", body=PR_COMMENT)
        impl_1 = kb.claim_task(conn, task_id)
        assert kb.request_review(
            conn, task_id, summary="Implementation v1, see PR.",
            reviewer="reviewer", expected_run_id=impl_1.current_run_id,
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "review"

        result = kbd.dispatch_once(conn, spawn_fn=spawn)

        assert task_id in spawned
        assert task_id in [s[0] for s in result.spawned]
        assert _events(conn, task_id, kind="review_dedupe_replayed") == []


def test_review_dedupe_disabled_always_spawns(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``kanban.review_dedupe=false`` must bypass the guard entirely, even
    when the head SHA plainly hasn't moved."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True, "review_dedupe": False}},
    )

    def _unexpected(*a, **k):
        raise AssertionError("fetch_pr_head_sha must not be called when review_dedupe is disabled")

    monkeypatch.setattr(pr_acceptance, "fetch_pr_head_sha", _unexpected)
    spawned: list[str] = []

    def spawn(task, workspace):
        spawned.append(task.id)

    with kbc.connect() as conn:
        task_id = _build_second_round_review_card(conn, changes_requested_head_sha=SHA_A)

        result = kbd.dispatch_once(conn, spawn_fn=spawn)

        assert task_id in spawned
        assert task_id in [s[0] for s in result.spawned]


def test_new_comment_after_verdict_spawns_for_real(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A comment added after the prior verdict closed (a possible new finding)
    must block the dedupe replay even though the head SHA is unchanged."""
    monkeypatch.setattr(pr_acceptance, "fetch_pr_head_sha", lambda *a, **k: SHA_A)
    spawned: list[str] = []

    def spawn(task, workspace):
        spawned.append(task.id)

    with kbc.connect() as conn:
        task_id = _build_second_round_review_card(conn, changes_requested_head_sha=SHA_A)
        comment_id = kb.add_comment(
            conn, task_id, author="worker", body="Actually, one more thing changed.",
        )
        # Force the comment's timestamp strictly after the prior verdict's
        # ended_at: both land in the same wall-clock second in a fast test run,
        # and the dedupe guard's "newer activity" check is a strict ``>``.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_comments SET created_at = created_at + 5 WHERE id = ?",
                (comment_id,),
            )

        result = kbd.dispatch_once(conn, spawn_fn=spawn)

        assert task_id in spawned
        assert task_id in [s[0] for s in result.spawned]
        assert _events(conn, task_id, kind="review_dedupe_replayed") == []
