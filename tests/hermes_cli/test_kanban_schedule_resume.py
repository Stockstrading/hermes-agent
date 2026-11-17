"""Tests for the ``resume_at`` wake timer and ``exclude_block_loop`` filter

in ``hermes_cli.kanban_db`` (card t_8686b192; split into its own file so
``test_kanban_db.py`` does not grow past its FILE_LINES cap)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_schedule_task_sets_resume_at(kanban_home):
    """``schedule_task``'s ``resume_at`` kwarg (the quota/429 backoff path)
    is persisted on the row; omitting it keeps the legacy NULL/park-forever
    behaviour."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota wait", assignee="a")
        wake = int(time.time()) + 900
        assert kb.schedule_task(conn, tid, reason="SCHEDULED_UNTIL=later", resume_at=wake)
        task = kb.get_task(conn, tid)
        assert task.status == "scheduled"
        assert conn.execute(
            "SELECT resume_at FROM tasks WHERE id=?", (tid,),
        ).fetchone()[0] == wake

        tid2 = kb.create_task(conn, title="human park", assignee="a")
        assert kb.schedule_task(conn, tid2, reason="waiting on a human")
        assert conn.execute(
            "SELECT resume_at FROM tasks WHERE id=?", (tid2,),
        ).fetchone()[0] is None


def test_recompute_scheduled_promotes_due_card_and_leaves_future_one(kanban_home):
    """A ``resume_at`` in the past is promoted to ``ready`` (and cleared);
    one still in the future is left ``scheduled``, untouched."""
    now = int(time.time())
    with kbc.connect() as conn:
        due = kb.create_task(conn, title="due", assignee="a")
        assert kb.schedule_task(conn, due, resume_at=now - 10)
        not_due = kb.create_task(conn, title="not due", assignee="a")
        assert kb.schedule_task(conn, not_due, resume_at=now + 3600)
        parked = kb.create_task(conn, title="parked forever", assignee="a")
        assert kb.schedule_task(conn, parked, reason="human will unblock")

        promoted = kb.recompute_scheduled(conn)
        assert promoted == 1

        due_task = kb.get_task(conn, due)
        assert due_task.status == "ready"
        assert conn.execute(
            "SELECT resume_at FROM tasks WHERE id=?", (due,),
        ).fetchone()[0] is None

        assert kb.get_task(conn, not_due).status == "scheduled"
        assert kb.get_task(conn, parked).status == "scheduled"

        # Idempotent: nothing left to promote on a second sweep.
        assert kb.recompute_scheduled(conn) == 0


def test_list_tasks_exclude_block_loop_drops_most_recent_escalation(kanban_home):
    """``exclude_block_loop=True`` drops a card with an unresolved
    ``block_loop_detected`` escalation; a plain ``commented`` event must NOT
    un-hide it (that was the bug -- any later event, including a bare comment,
    defeated the filter), but an explicit resolving event (``specified``, via
    the real ``specify_triage_task`` the triage sweep/human actually calls on
    a triage-escalated card) does."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="escalated", assignee="a", triage=True)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "block_loop_detected", {"kind": "capability"})

        all_tasks = kb.list_tasks(conn, status="triage")
        assert any(t.id == tid for t in all_tasks)

        filtered = kb.list_tasks(conn, status="triage", exclude_block_loop=True)
        assert not any(t.id == tid for t in filtered)

        # A bare comment is not a resolution -- the escalation must stay hidden.
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "commented", {})
        filtered_after_comment = kb.list_tasks(conn, status="triage", exclude_block_loop=True)
        assert not any(t.id == tid for t in filtered_after_comment)

        # An explicit resolving event clears the escalation. The card is in
        # 'triage' (where block_loop escalation lands it), so the production
        # resolution path is specify_triage_task (triage -> todo -> auto-promoted
        # to 'ready', since this task has no parents -- 'specified' event);
        # unblock_task only applies to 'blocked'/'scheduled' and would no-op here.
        assert kb.specify_triage_task(conn, tid, title="escalated (specified)")
        filtered_after_unblock = kb.list_tasks(conn, status="ready", exclude_block_loop=True)
        assert any(t.id == tid for t in filtered_after_unblock)

        # Re-escalated for an unrelated reason after being cleared: a fresh
        # ``block_loop_detected`` with nothing resolving it yet must hide again.
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "block_loop_detected", {"kind": "needs_input"})
        filtered_after_reescalation = kb.list_tasks(conn, status="ready", exclude_block_loop=True)
        assert not any(t.id == tid for t in filtered_after_reescalation)
