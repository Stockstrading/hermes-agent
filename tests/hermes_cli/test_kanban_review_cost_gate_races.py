"""Race contracts for the review cost gate (PR #3 round-3 findings F5/F6).

F5: parking a review-lane card for the owner must go review -> blocked in one
transaction; no other connection may ever see it ``ready`` (claimable).
F6: ``duplicate_review_running`` is checked before the claim, so two
dispatchers on one board can both pass the check for two cards on the same
PR; the claim itself must re-check it so exactly one reviewer runs.
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


def _set_status(conn, task_id: str, status: str) -> None:
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def _insert_verdict_run(conn, task_id: str, profile: str, *, started_at: int, ended_at: int) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, ?, 'done', 'completed', ?, ?)",
            (task_id, profile, started_at, ended_at),
        )


def test_round_cap_block_of_review_card_is_never_claimable(kanban_home: Path, monkeypatch) -> None:
    """F5: a second dispatcher probing at every step of the round-cap block sees
    ``review`` then ``blocked``, never ``ready``, and its claim never succeeds."""
    pr = "strategy-api#77777"
    with kbc.connect() as conn, kbc.connect() as other:
        tid = kb.create_task(conn, title=f"Review {pr} round 5", assignee="reviewer")
        _set_status(conn, tid, "review")
        now = int(time.time())
        for i in range(kbd._REVIEW_ROUND_CAP):
            prior = kb.create_task(conn, title=f"Review {pr}", assignee="reviewer")
            _insert_verdict_run(conn, prior, "reviewer", started_at=now - 10000 + i * 300,
                                ended_at=now - 10000 + i * 300 + 200)

        seen: list = []
        claims: list = []
        real_block, real_end = kb.block_task, kb._end_or_synthesize_run

        def probe_then_block(c, task_id, **kw):
            # Between the gate's decision and the block's own transaction.
            seen.append(kb._task_status(other, task_id))
            claims.append(kb.claim_task(other, task_id))
            return real_block(c, task_id, **kw)

        def probe_inside_txn(c, task_id, **kw):
            seen.append(kb._task_status(other, task_id))  # committed state only (WAL reader)
            return real_end(c, task_id, **kw)

        monkeypatch.setattr(kb, "block_task", probe_then_block)
        monkeypatch.setattr(kb, "_end_or_synthesize_run", probe_inside_txn)

        assert kbd._review_cost_gate(conn, tid, "reviewer", dry_run=False) == "review_round_cap"

        assert "ready" not in seen, seen
        assert seen and set(seen) == {"review"}
        assert claims == [None]
        assert kb._task_status(other, tid) == "blocked"
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]
        assert "claimed" not in kinds
        assert "blocked" in kinds
        # Unblock keeps today's landing lane: the card resumes in 'ready'.
        assert kb.unblock_task(conn, tid)
        assert kb._task_status(conn, tid) == "ready"


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_two_dispatchers_same_pr_exactly_one_reviewer_claims(kanban_home: Path, monkeypatch, lane: str) -> None:
    """F6: both dispatchers pass the gate's pre-claim duplicate check (the race
    window); the claim-time veto lets exactly one card for the PR run."""
    pr = "strategy-api#66666"
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    with kbc.connect() as a, kbc.connect() as b:
        x = kb.create_task(a, title=f"Review {pr}", assignee="reviewer")
        y = kb.create_task(a, title=f"Re-review {pr} after fix", assignee="reviewer")
        for tid in (x, y):
            _set_status(a, tid, lane)

        # The check half of the race: neither card is running yet, both pass.
        assert kbd._review_cost_gate(a, x, "reviewer", dry_run=True) is None
        assert kbd._review_cost_gate(b, y, "reviewer", dry_run=True) is None
        # Replay those (now stale) verdicts in the claim half.
        monkeypatch.setattr(kbd, "_review_cost_gate", lambda *_a, **_k: None)
        monkeypatch.setattr(kbd, "_review_head_unchanged", lambda *_a, **_k: (False, None))

        spawned: list = []

        def dispatch(conn, tid):
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
            return kbd._dispatch_lane_task(
                conn, row, "reviewer", kbd.DispatchResult(), lane=lane, dry_run=False,
                ttl_seconds=None, board=None, failure_limit=2,
                spawn_fn=lambda task, workspace: spawned.append(task.id),
                per_profile_cap=None, per_profile_running={},
            )

        assert dispatch(a, x) is True
        assert dispatch(b, y) is False

        assert spawned == [x]
        assert kb._task_status(b, x) == "running"
        assert kb._task_status(b, y) == lane  # untouched, re-checked next tick
        rejected = b.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'claim_rejected'", (y,),
        ).fetchall()
        assert [kb._json_dict(r["payload"]).get("reason") for r in rejected] == ["duplicate_review_running"]


def test_claim_veto_ignored_for_non_review_profiles(kanban_home: Path, monkeypatch) -> None:
    """The veto is wired only for review-gate profiles: two implementer cards
    naming the same PR still both claim."""
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    with kbc.connect() as conn:
        x = kb.create_task(conn, title="Fix strategy-api#55555 lint", assignee="backend-dev")
        y = kb.create_task(conn, title="Fix strategy-api#55555 tests", assignee="backend-dev")
        spawned: list = []
        for tid in (x, y):
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
            kbd._dispatch_lane_task(
                conn, row, "backend-dev", kbd.DispatchResult(), lane="ready", dry_run=False,
                ttl_seconds=None, board=None, failure_limit=2,
                spawn_fn=lambda task, workspace: spawned.append(task.id),
                per_profile_cap=None, per_profile_running={},
            )
        assert spawned == [x, y]
