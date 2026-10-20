"""dispatch_once / recompute_scheduled wiring for the quota/429 wait-until-
reset scheduling path (tasks.resume_at).

A worker/guard that hits a provider quota wall parks the card with
``kanban_schedule(reason="SCHEDULED_UNTIL=<reset time>")`` instead of
``kanban_block(kind="needs_input")``. The dispatcher must auto-wake it once
``resume_at`` is due, in the SAME tick that would otherwise spawn ready work
— not one tick later — and such a card must never show up in the triage
auto-decompose sweep (it is in 'scheduled', not 'triage').
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


def test_dispatch_once_calls_recompute_scheduled_before_ready_rows(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    """A card parked via the quota path with a due ``resume_at`` is promoted
    to ``ready`` and spawned within the SAME ``dispatch_once`` tick."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota wait", assignee="a")
        assert kb.schedule_task(conn, tid, resume_at=int(time.time()) - 5)

        calls: list[str] = []
        real_recompute_scheduled = kb.recompute_scheduled
        real_recompute_ready = kb.recompute_ready

        def _tracked_scheduled(conn_):
            calls.append("scheduled")
            return real_recompute_scheduled(conn_)

        def _tracked_ready(conn_, **kw):
            calls.append("ready")
            return real_recompute_ready(conn_, **kw)

        monkeypatch.setattr(kb, "recompute_scheduled", _tracked_scheduled)
        monkeypatch.setattr(kb, "recompute_ready", _tracked_ready)

        res = kbd.dispatch_once(conn, dry_run=True)

        assert calls == ["scheduled", "ready"]
        assert res.scheduled_promoted == 1
        assert [t for t, _a, _w in res.spawned] == [tid]
        assert kb.get_task(conn, tid).status == "ready"


def test_dispatch_once_leaves_not_yet_due_scheduled_card_alone(
    kanban_home, all_assignees_spawnable,
):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota wait later", assignee="a")
        assert kb.schedule_task(conn, tid, resume_at=int(time.time()) + 3600)

        res = kbd.dispatch_once(conn, dry_run=True)

        assert res.scheduled_promoted == 0
        assert res.spawned == []
        assert kb.get_task(conn, tid).status == "scheduled"


def test_quota_scheduled_card_never_enters_triage_auto_decompose_sweep(kanban_home):
    """A card scheduled via the quota path is in 'scheduled', never
    'triage' — ``list_triage_ids`` (the auto-decompose sweep's source) must
    not see it, due or not."""
    from hermes_cli import kanban_specify

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota wait", assignee="a")
        assert kb.schedule_task(conn, tid, resume_at=int(time.time()) - 5)
        assert kb.get_task(conn, tid).status == "scheduled"

    assert tid not in kanban_specify.list_triage_ids()

    with kbc.connect() as conn:
        kb.recompute_scheduled(conn)
        assert kb.get_task(conn, tid).status == "ready"

    assert tid not in kanban_specify.list_triage_ids()
