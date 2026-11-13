"""Review-loop convergence ADR (card t_f526b652), Rules 3 and 5.

Rule 3 — delta-only review after round 1: round >= 2 gets the diff since the
prior verdict's reviewed SHA plus the prior findings' open/addressed status,
never a full re-review; only a MAJOR-or-higher finding blocks approval.

Rule 5 — fallback model for low-risk/text-only review PRs: only a non-capital,
docs/tests/CI-only, small PR may run on ``auxiliary.review_fallback`` when the
required gate model is confirmed unavailable; capital-sensitive PRs never
fall back, regardless of size.
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


def _insert_verdict_run(conn, task_id, profile, *, outcome="completed",
                         metadata=None, started_at=None, ended_at=None):
    now = int(time.time())
    started_at = now - 600 if started_at is None else started_at
    ended_at = now - 400 if ended_at is None else ended_at
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, metadata) "
        "VALUES (?, ?, 'done', ?, ?, ?, ?)",
        (task_id, profile, started_at, ended_at, outcome, kb._json_or_null(metadata) if metadata else None),
    )
    conn.commit()


# --- Rule 3 -------------------------------------------------------------------

def test_round1_has_no_delta_context(conn):
    """Round 1 (no prior verdict yet) must return None: a normal full review."""
    task_id = kb.create_task(conn, title="review strategy-lab#27", assignee="reviewer")
    assert kbd.build_round_delta_context(conn, task_id, "reviewer") is None


def test_round2_prompt_only_receives_delta_diff_and_prior_findings(conn):
    """Round >= 2 gets ONLY the diff since the prior reviewed SHA plus the prior
    findings' status, never an instruction to re-review the whole PR."""
    task_id = kb.create_task(conn, title="review strategy-lab#27", assignee="reviewer")
    findings = [
        {"id": "F1", "severity": "MAJOR", "file": "app/core/x.py", "line": 10, "status": "open"},
        {"id": "F2", "severity": "MINOR", "file": "app/core/y.py", "line": 20, "status": "open"},
    ]
    _insert_verdict_run(
        conn, task_id, "reviewer",
        metadata={"findings": findings, "reviewed_sha": "abc123"},
    )

    context = kbd.build_round_delta_context(conn, task_id, "reviewer")

    assert context is not None
    assert "abc123" in context
    assert "F1" in context and "F2" in context
    assert "MAJOR" in context and "MINOR" in context
    assert "re-review the whole PR from scratch" not in context.lower().replace("not ", "")
    assert "do not re-review the whole pr from scratch" in context.lower()


def test_round2_no_prior_findings_still_returns_delta_context(conn):
    task_id = kb.create_task(conn, title="review strategy-lab#27", assignee="reviewer")
    _insert_verdict_run(conn, task_id, "reviewer", metadata={"reviewed_sha": "deadbee"})
    context = kbd.build_round_delta_context(conn, task_id, "reviewer")
    assert context is not None
    assert "deadbee" in context
    assert "No prior findings were recorded." in context


@pytest.mark.parametrize(
    "findings, expected",
    [
        ([{"severity": "MINOR"}], False),
        ([{"severity": "NIT"}], False),
        ([{"severity": "MINOR"}, {"severity": "NIT"}], False),
        ([{"severity": "MAJOR"}], True),
        ([{"severity": "CRITICAL"}], True),
        ([{"severity": "minor"}, {"severity": "major"}], True),
        ([], False),
    ],
)
def test_should_block_on_findings_severity_gate(findings, expected):
    """After round 2, a MINOR finding does not block approval; a MAJOR finding does."""
    assert kbd.should_block_on_findings(findings) is expected


# --- Rule 5 -------------------------------------------------------------------

def _docs_task(title="review bff-api#141"):
    class Task:
        def __init__(self):
            self.id = "t_fake"
            self.title = title
            self.body = ""

    return Task()


def test_docs_only_small_pr_is_low_risk():
    task = _docs_task()
    pr_meta = {"files": ["docs/foo.md", "tests/test_bar.py"], "additions": 8, "deletions": 2}
    assert kbd._is_low_risk_review(task, pr_meta) is True


def test_40_line_docs_pr_is_not_low_risk_over_size_threshold():
    task = _docs_task()
    pr_meta = {"files": ["docs/foo.md"], "additions": 30, "deletions": 10}
    assert kbd._is_low_risk_review(task, pr_meta) is False


def test_capital_repo_pr_is_never_low_risk_regardless_of_size():
    task = _docs_task(title="[CAPITAL] review strategy-api#276")
    pr_meta = {"files": ["docs/readme.md"], "additions": 1, "deletions": 1}
    assert kbd._is_low_risk_review(task, pr_meta) is False


def test_non_docs_file_is_not_low_risk():
    task = _docs_task()
    pr_meta = {"files": ["app/core/handler.py"], "additions": 5, "deletions": 1}
    assert kbd._is_low_risk_review(task, pr_meta) is False


def test_capital_deny_substring_in_path_is_never_low_risk():
    task = _docs_task()
    pr_meta = {"files": ["docs/trade-notes.md"], "additions": 1, "deletions": 1}
    assert kbd._is_low_risk_review(task, pr_meta) is False


def test_docs_only_pr_with_gate_down_uses_fallback_model_and_tags_verdict(conn, monkeypatch):
    """A docs-only 10-line PR with the required gate down must fall back to the
    configured model, and the dispatcher must disclose the substitution on the card."""
    task_id = kb.create_task(conn, title="review bff-api#141", assignee="reviewer")
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, error) "
        "VALUES (?, 'reviewer', 'done', ?, ?, 'rate_limited', '429 quota exceeded')",
        (task_id, int(time.time()) - 60, int(time.time()) - 30),
    )
    conn.commit()
    monkeypatch.setattr(
        kbd, "_review_fallback_model", lambda: ("claude-haiku-5", "anthropic")
    )

    class Task:
        id = task_id
        title = "review bff-api#141"
        body = ""

    fallback = kbd._maybe_review_fallback_model(
        conn, Task(), "reviewer",
        pr_meta={"files": ["docs/readme.md"], "additions": 4, "deletions": 6},
    )

    assert fallback == ("claude-haiku-5", "anthropic")
    comments = conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,)
    ).fetchone()
    assert "claude-haiku-5" in comments["body"]
    assert "429 quota exceeded" in comments["body"]


def test_capital_repo_pr_with_gate_down_never_falls_back(conn, monkeypatch):
    task_id = kb.create_task(conn, title="[CAPITAL] review strategy-api#276", assignee="reviewer")
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, error) "
        "VALUES (?, 'reviewer', 'done', ?, ?, 'rate_limited', '429 quota exceeded')",
        (task_id, int(time.time()) - 60, int(time.time()) - 30),
    )
    conn.commit()
    monkeypatch.setattr(
        kbd, "_review_fallback_model", lambda: ("claude-haiku-5", "anthropic")
    )

    class Task:
        id = task_id
        title = "[CAPITAL] review strategy-api#276"
        body = ""

    fallback = kbd._maybe_review_fallback_model(
        conn, Task(), "reviewer",
        pr_meta={"files": ["docs/readme.md"], "additions": 1, "deletions": 1},
    )
    assert fallback is None


def test_gate_only_slow_not_confirmed_unavailable_never_falls_back(conn, monkeypatch):
    """A latest run that simply completed (gate merely slow, not down) must never
    trigger a fallback substitution (OWNER-RULES #3: confirmed unavailable only)."""
    task_id = kb.create_task(conn, title="review bff-api#150", assignee="reviewer")
    _insert_verdict_run(conn, task_id, "reviewer", outcome="completed")
    monkeypatch.setattr(
        kbd, "_review_fallback_model", lambda: ("claude-haiku-5", "anthropic")
    )

    class Task:
        id = task_id
        title = "review bff-api#150"
        body = ""

    fallback = kbd._maybe_review_fallback_model(
        conn, Task(), "reviewer",
        pr_meta={"files": ["docs/readme.md"], "additions": 1, "deletions": 1},
    )
    assert fallback is None


def test_no_fallback_configured_returns_none(conn, monkeypatch):
    task_id = kb.create_task(conn, title="review bff-api#150", assignee="reviewer")
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, error) "
        "VALUES (?, 'reviewer', 'done', ?, ?, 'rate_limited', '429 quota exceeded')",
        (task_id, int(time.time()) - 60, int(time.time()) - 30),
    )
    conn.commit()
    monkeypatch.setattr(kbd, "_review_fallback_model", lambda: None)

    class Task:
        id = task_id
        title = "review bff-api#150"
        body = ""

    fallback = kbd._maybe_review_fallback_model(
        conn, Task(), "reviewer",
        pr_meta={"files": ["docs/readme.md"], "additions": 1, "deletions": 1},
    )
    assert fallback is None
