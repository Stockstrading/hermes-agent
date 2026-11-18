"""Tests for kb.specify_triage_task — the DB-layer atomic promotion
from the triage column to todo. LLM-free by design."""

from __future__ import annotations

import threading
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


def _create_triage(conn, title="rough idea", body=None, assignee=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        triage=True,
    )


def test_specify_promotes_triage_to_todo(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rough idea")
        assert kb.get_task(conn, tid).status == "triage"
    with kbc.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            tid,
            title="Refined: rough idea",
            body="**Goal**\nDo the thing.",
            author="specifier-bot",
        )
    assert ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # No parents → recompute_ready should have flipped it past todo to ready.
    assert task.status == "ready"
    assert task.title == "Refined: rough idea"
    assert "**Goal**" in (task.body or "")


def test_specify_rejects_blank_title(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="rough")
    with kbc.connect() as conn, pytest.raises(ValueError):
        kb.specify_triage_task(conn, tid, title="   ", body="ok")


def test_specify_records_audit_comment_only_when_author_given(kanban_home):
    # With author → comment added.
    with kbc.connect() as conn:
        tid1 = _create_triage(conn, title="a")
        kb.specify_triage_task(
            conn, tid1, title="A-spec", body="b", author="ace"
        )
        comments1 = kb.list_comments(conn, tid1)
    assert len(comments1) == 1
    assert "Specified" in comments1[0].body
    assert comments1[0].author == "ace"

    # Without author → no comment (silent).
    with kbc.connect() as conn:
        tid2 = _create_triage(conn, title="b")
        kb.specify_triage_task(conn, tid2, title="B-spec", body="b")
        comments2 = kb.list_comments(conn, tid2)
    assert comments2 == []


# ---------------------------------------------------------------------------
# PR12 release-gate MAJOR 2 — the CAS must refuse promotion atomically, not
# rely on a stale enumeration made minutes earlier.
# ---------------------------------------------------------------------------


def _running_task(conn, title="t"):
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _triage_card_with_unresolved_block_loop(conn, title="t"):
    """Reach the real production path to a triage card carrying an unresolved
    ``block_loop_detected`` event (mirrors the existing list_triage_ids tests)."""
    tid = _running_task(conn, title)
    kb.block_task(conn, tid, reason="x", kind="capability")
    kb.unblock_task(conn, tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    kb.block_task(conn, tid, reason="x", kind="capability")
    assert kb.get_task(conn, tid).status == "triage"
    return tid


def test_specify_triage_task_refuses_when_block_loop_unresolved(kanban_home):
    """Even though the row's ``status`` column is still ``triage`` (what a
    caller's earlier, now-stale enumeration saw), the CAS inside
    specify_triage_task must independently see the unresolved escalation and
    refuse to promote -- this is the fix for the TOCTOU the release gate
    found: enumeration and mutation no longer trust different reads."""
    with kbc.connect() as conn:
        tid = _triage_card_with_unresolved_block_loop(conn)

    with kbc.connect() as conn:
        ok = kb.specify_triage_task(conn, tid, title="Sneaked past the gate", author="sweep")
    assert ok is False

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = [ev.kind for ev in kb.list_events(conn, tid)]
    # Refused: still in triage, title untouched, and the sweep's own write
    # never landed a 'specified' event that would have resolved the escalation.
    assert task.status == "triage"
    assert task.title == "t"
    assert "specified" not in events


def test_specify_triage_task_cas_is_atomic_not_a_stale_pre_check(kanban_home):
    """PR12 release-gate round-5 MINOR 4: the test above only proves the CAS
    refuses an escalation already fully committed before the call starts --
    a regression that replaced the CAS with a non-transactional pre-check
    (read escalation, THEN an unguarded UPDATE in its own write_txn) would
    pass that test too, since the escalation is old news by the time either
    version even starts. This test uses two REAL connections and relies on
    SQLite's own BEGIN IMMEDIATE lock (not a sleep) as the barrier: connB
    opens a write transaction and holds it -- lock acquired, escalation
    written but NOT YET COMMITTED -- while connA's specify_triage_task call
    is literally blocked trying to acquire that same write lock. Only once
    connB commits does connA's call proceed; its own read of the escalation
    can therefore never be older than connB's write. A separate, earlier,
    non-transactional pre-check inside connA would have run (and found
    nothing) before connB ever touched the row, and would then promote once
    the lock frees up -- this test fails on that mutation."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="race", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))

    lock_held = threading.Event()
    connB_thread_started = threading.Event()
    release_lock = threading.Event()

    def _hold_lock_with_uncommitted_escalation():
        connB_thread_started.set()
        with kbc.connect() as connB:
            with kb.write_txn(connB):
                kb._append_event(connB, tid, "block_loop_detected", None)
                lock_held.set()
                release_lock.wait(timeout=5)
            # __exit__ commits here, after connA is already blocked on the lock.

    tB = threading.Thread(target=_hold_lock_with_uncommitted_escalation)
    tB.start()
    assert connB_thread_started.wait(timeout=5)
    assert lock_held.wait(timeout=5)

    result: dict = {}
    connA_started = threading.Event()

    def _call_specify():
        connA_started.set()
        with kbc.connect() as connA:
            result["ok"] = kb.specify_triage_task(
                connA, tid, title="Sneaked past the gate", author="sweep",
            )

    tA = threading.Thread(target=_call_specify)
    tA.start()
    assert connA_started.wait(timeout=5)
    release_lock.set()
    tA.join(timeout=5)
    tB.join(timeout=5)
    assert not tA.is_alive() and not tB.is_alive()

    assert result.get("ok") is False
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        events = [ev.kind for ev in kb.list_events(conn, tid)]
    assert task.status == "triage"
    assert task.title == "race"
    assert "specified" not in events


