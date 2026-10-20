"""Generalized review round cap (card t_480b842d, ADR Rule 4):
3 real review rounds OR 3 quota stalls (429s/scheduled-but-no-verdict runs)
on the same PR escalates to the owner, same as the existing real-round cap.

A "quota stall" is a spawn that never produced a verdict: either it ran the
resume_at scheduling path (outcome='scheduled', card t_36404fba) or it ended
rate_limited/blocked while the card is currently block_kind='needs_input' --
and either way the run was SHORT (< _REVIEW_MIN_RUN_SECONDS), so a real
reviewed-and-blocked verdict is never miscounted as a stall.
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
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _insert_run(conn, task_id, profile, *, outcome, started_at=None, ended_at=None, short=False):
    now = int(time.time())
    if short:
        started_at = now - 600 if started_at is None else started_at
        ended_at = started_at + 30 if ended_at is None else ended_at  # < _REVIEW_MIN_RUN_SECONDS
    else:
        started_at = now - 600 if started_at is None else started_at
        ended_at = now - 400 if ended_at is None else ended_at  # >= _REVIEW_MIN_RUN_SECONDS
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome) "
        "VALUES (?, ?, 'done', ?, ?, ?)",
        (task_id, profile, started_at, ended_at, outcome),
    )
    conn.commit()


def _insert_quota_stall(conn, task_id, profile, *, via="scheduled"):
    """A spawn that never produced a verdict -- short runtime by construction."""
    if via == "scheduled":
        _insert_run(conn, task_id, profile, outcome="scheduled", short=True)
    else:
        _insert_run(conn, task_id, profile, outcome=via, short=True)
        conn.execute("UPDATE tasks SET block_kind = 'needs_input' WHERE id = ?", (task_id,))
        conn.commit()


def _insert_real_round(conn, task_id, profile):
    _insert_run(conn, task_id, profile, outcome="completed", short=False)


def test_third_quota_stall_escalates_with_required_summary_fields(conn, monkeypatch):
    """3rd quota stall on a PR blocks for the owner, needs_input, with findings,
    head SHA, CI status and the three named options in the summary."""
    task_id = kb.create_task(conn, title="review strategy-lab#99", assignee="reviewer")
    for _ in range(3):
        _insert_quota_stall(conn, task_id, "reviewer", via="scheduled")

    monkeypatch.setattr(
        kbd, "_pr_url_for_key", lambda c, pr: "https://github.com/Stockstrading/strategy-lab/pull/99"
    )

    from hermes_cli import kanban_pr_acceptance as kpa
    monkeypatch.setattr(
        kpa, "collect_acceptance",
        lambda contract, published_pr, assignee=None: {"head_sha": "abc123deadbeef", "classification": "passing"},
    )

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)

    assert reason == "review_round_cap"
    task = kb.get_task(conn, task_id)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"
    # Pull the actual block reason off the event log (block_task stores it there).
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    payload = row["payload"]
    assert "quota stall cap" in payload.lower()
    assert "head sha: abc123deadbeef" in payload.lower()
    assert "ci: passing" in payload.lower()
    assert "merge" in payload.lower() and "waive" in payload.lower() and "close" in payload.lower()


def test_mixed_two_real_two_quota_does_not_prematurely_escalate(conn):
    """2 real rounds + 2 quota stalls hits neither individual cap (3 each) --
    the card must stay spawnable, not escalate."""
    task_id = kb.create_task(conn, title="review strategy-lab#100", assignee="reviewer")
    for _ in range(2):
        _insert_real_round(conn, task_id, "reviewer")
    for _ in range(2):
        _insert_quota_stall(conn, task_id, "reviewer", via="scheduled")

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)

    assert reason is None
    task = kb.get_task(conn, task_id)
    assert task.status != "blocked"


def test_escalated_card_excluded_from_next_auto_decompose_tick(conn):
    """A card escalated via the quota-stall cap is block_kind='needs_input',
    so list_triage_ids (the auto-decompose sweep's source) must never see it."""
    from hermes_cli import kanban_specify

    task_id = kb.create_task(conn, title="review strategy-lab#101", assignee="reviewer")
    for _ in range(3):
        _insert_quota_stall(conn, task_id, "reviewer", via="scheduled")

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)
    assert reason == "review_round_cap"
    assert kb.get_task(conn, task_id).block_kind == "needs_input"

    assert task_id not in kanban_specify.list_triage_ids()


def test_real_round_cap_still_honours_owner_per_pr_override(conn):
    """Regression guard for the fixed bug: a per-PR owner override in
    _REVIEW_ROUND_CAP_BY_PR must actually raise the cap, not be silently
    discarded by a stray re-assignment to the default."""
    pr_key = "strategy-lab#27"  # has an owner override of 8 in _REVIEW_ROUND_CAP_BY_PR
    assert kbd._REVIEW_ROUND_CAP_BY_PR.get(pr_key) == 8
    task_id = kb.create_task(conn, title=f"review {pr_key}", assignee="reviewer")
    for _ in range(5):  # below the override cap of 8, above the default cap of 4
        _insert_real_round(conn, task_id, "reviewer")

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)

    assert reason is None
    assert kb.get_task(conn, task_id).status != "blocked"
