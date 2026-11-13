"""Review contract v2 (ADR-review-contract-v2.md, card t_cca086d6 / t_3183cd6d):
delta reviews, frozen findings, ONE synced round cap.

Rule 1 -- evidence-gated hand-off: a ``kanban_request_review`` hand-off missing
a scope line / file list / commands-run block / ``base: origin/main@<sha>``
freshness statement bounces the card instead of spawning a reviewer, and the
bounce is never counted as a round.

Rule 2/3 extension -- an "addressed" finding needs a ``test_ref`` to count as
resolved; missing it keeps the finding open.

Rule 4 -- frozen findings: after round 2, a brand-new (non frozen-list)
finding only stays eligible to block the PR if its category is
capital-safety, data-loss, or security; everything else becomes a follow-up.

Rule 7 -- single source of truth: three named round caps (default/capital/
docs-only) replace the single flat ``_REVIEW_ROUND_CAP``, with the owner's
per-PR override (``_REVIEW_ROUND_CAP_BY_PR``) still taking precedence.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_dispatch_review_contract as kbd_review_contract
from hermes_cli import kanban_pr_acceptance as kb_pr_acceptance


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


def _insert_real_round(conn, task_id, profile, *, metadata=None, started_at=None, ended_at=None):
    now = int(time.time())
    started_at = now - 600 if started_at is None else started_at
    ended_at = now - 400 if ended_at is None else ended_at
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, metadata) "
        "VALUES (?, ?, 'done', ?, ?, 'completed', ?)",
        (task_id, profile, started_at, ended_at, kb._json_or_null(metadata) if metadata else None),
    )
    conn.commit()


_EVIDENCE_OK = (
    "Scope: fix the review round cap.\n"
    "Files: hermes_cli/kanban_db_dispatch.py, tests/hermes_cli/test_x.py\n"
    "Commands run: scripts/run_tests.sh tests/hermes_cli/test_x.py -> 4 passed\n"
    "base: origin/main@abcdef1234567890\n"
)


def _insert_review_requested(conn, task_id, *, summary, metadata=None, started_at=None):
    started_at = int(time.time()) - 60 if started_at is None else started_at
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, summary, metadata) "
        "VALUES (?, 'backend-dev', 'done', ?, ?, 'review_requested', ?, ?)",
        (task_id, started_at, started_at + 30, summary,
         kb._json_or_null(metadata) if metadata else None),
    )
    conn.commit()


# --- Rule 1: evidence-gated hand-off, bounce doesn't count --------------------
#
# ADR-review-contract-v2.md §3: "Do not make the evidence gate retroactive --
# only applies to request_review calls made after this ships." The rollout
# cutover is ``kbd_review_contract.EVIDENCE_GATE_ROLLOUT_AT`` (module
# constant, ``None`` until the owner flips it once the SOUL.md diffs that
# teach backend-dev/frontend-dev the new hand-off format are live). Tests
# that exercise enforcement set it explicitly; tests that exercise the
# default/pre-cutover state leave it alone.

# Standard gate1->gate2 hand-off shape mandated by the reviewer/release-reviewer
# system prompt TODAY (before the SOUL.md diffs in PR #5 land) -- no Scope/
# Files/Commands-run/base lines at all. This must never be bounced while the
# gate is disabled or while the hand-off predates the rollout cutover.
_GATE1_TO_GATE2_SUMMARY = (
    "gate 1 (Sonnet) clean at head abc1234: ruff/pytest pass, no capital-scope "
    "findings; PR https://github.com/Stockstrading/example-repo/pull/9"
)
_GATE1_TO_GATE2_METADATA = {
    "published_pr": "https://github.com/Stockstrading/example-repo/pull/9",
    "gate1": "clean",
    "head": "abc1234",
}


def test_gate_disabled_by_default_never_bounces_standard_handoff(conn):
    """Default state (no owner cutover yet): the gate must be fully inert, even
    against the EXACT hand-off format reviewer/release-reviewer produce today."""
    assert kbd_review_contract.EVIDENCE_GATE_ROLLOUT_AT is None
    task_id = kb.create_task(conn, title="review strategy-lab#199", assignee="reviewer")
    _insert_review_requested(
        conn, task_id, summary=_GATE1_TO_GATE2_SUMMARY, metadata=_GATE1_TO_GATE2_METADATA,
    )

    ok, missing = kbd._handoff_has_evidence(conn, task_id)
    assert ok is True
    assert missing is None
    assert kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False) is None


def test_handoff_predating_cutover_is_grandfathered_in(conn, monkeypatch):
    """Once the owner flips the cutover, a hand-off written BEFORE it (already
    sitting in the queue) must still pass -- not retroactive."""
    cutover = int(time.time())
    monkeypatch.setattr(kbd_review_contract, "EVIDENCE_GATE_ROLLOUT_AT", cutover)
    task_id = kb.create_task(conn, title="review strategy-lab#204", assignee="reviewer")
    _insert_review_requested(
        conn, task_id, summary=_GATE1_TO_GATE2_SUMMARY, metadata=_GATE1_TO_GATE2_METADATA,
        started_at=cutover - 3600,
    )

    ok, missing = kbd._handoff_has_evidence(conn, task_id)
    assert ok is True
    assert missing is None


def test_handoff_after_cutover_is_enforced(conn, monkeypatch):
    """A hand-off written AFTER the owner's cutover is held to the new bar."""
    cutover = int(time.time()) - 3600
    monkeypatch.setattr(kbd_review_contract, "EVIDENCE_GATE_ROLLOUT_AT", cutover)
    task_id = kb.create_task(conn, title="review strategy-lab#205", assignee="reviewer")
    _insert_review_requested(
        conn, task_id, summary="fixed the thing, trust me", started_at=cutover + 60,
    )

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)

    assert reason == "handoff_missing_evidence"
    task = kb.get_task(conn, task_id)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"
    # No verdict run was created -- the bounce must never be counted as a round.
    assert kbd._review_verdict_runs(conn, "reviewer", {"strategy-lab#205"}) == []


def test_handoff_missing_base_sha_names_that_item(conn, monkeypatch):
    cutover = int(time.time()) - 3600
    monkeypatch.setattr(kbd_review_contract, "EVIDENCE_GATE_ROLLOUT_AT", cutover)
    task_id = kb.create_task(conn, title="review strategy-lab#201", assignee="reviewer")
    missing_base = _EVIDENCE_OK.replace("base: origin/main@abcdef1234567890\n", "")
    _insert_review_requested(conn, task_id, summary=missing_base, started_at=cutover + 60)

    ok, missing = kbd._handoff_has_evidence(conn, task_id)

    assert ok is False
    assert "base: origin/main@" in missing


def test_handoff_with_full_evidence_is_not_bounced(conn, monkeypatch):
    cutover = int(time.time()) - 3600
    monkeypatch.setattr(kbd_review_contract, "EVIDENCE_GATE_ROLLOUT_AT", cutover)
    task_id = kb.create_task(conn, title="review strategy-lab#202", assignee="reviewer")
    _insert_review_requested(conn, task_id, summary=_EVIDENCE_OK, started_at=cutover + 60)

    ok, missing = kbd._handoff_has_evidence(conn, task_id)
    assert ok is True
    assert missing is None

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)
    assert reason is None


def test_no_recorded_handoff_never_bounces(conn, monkeypatch):
    """A card with no review_requested run on record (e.g. owner/test-created)
    must not be bounced for evidence this gate cannot see, even post-cutover."""
    monkeypatch.setattr(kbd_review_contract, "EVIDENCE_GATE_ROLLOUT_AT", int(time.time()) - 3600)
    task_id = kb.create_task(conn, title="review strategy-lab#203", assignee="reviewer")
    ok, missing = kbd._handoff_has_evidence(conn, task_id)
    assert ok is True
    assert missing is None


# --- Rule 2/3 extension: test_ref required for "addressed" --------------------

def test_addressed_finding_without_test_ref_counts_as_open():
    finding = {"id": "F1", "severity": "MAJOR", "status": "addressed"}
    assert kbd._finding_is_addressed(finding) is False


def test_addressed_finding_with_test_ref_counts_as_resolved():
    finding = {"id": "F1", "severity": "MAJOR", "status": "addressed", "test_ref": "tests/test_x.py::test_f1"}
    assert kbd._finding_is_addressed(finding) is True


def test_open_findings_from_runs_excludes_addressed_with_test_ref(conn):
    task_id = kb.create_task(conn, title="review strategy-lab#210", assignee="reviewer")
    findings = [
        {"id": "F1", "severity": "MAJOR", "status": "addressed", "test_ref": "tests/test_x.py::test_f1"},
        {"id": "F2", "severity": "MAJOR", "status": "addressed"},  # no test_ref -> still open
    ]
    _insert_real_round(conn, task_id, "reviewer", metadata={"findings": findings})
    runs = kbd._review_verdict_runs(conn, "reviewer", {"strategy-lab#210"})

    open_findings = kbd._open_findings_from_runs(runs)

    assert [f["id"] for f in open_findings] == ["F2"]


# --- Rule 4: frozen findings after round 2 -------------------------------------

def test_round1_and_round2_are_never_frozen():
    findings = [{"id": "F1", "severity": "MAJOR", "category": "style"}]
    assert kbd._finding_allowed_after_freeze(findings[0], round_no=1, frozen_ids=set()) is True
    assert kbd._finding_allowed_after_freeze(findings[0], round_no=2, frozen_ids=set()) is True


def test_round3_new_non_safety_finding_is_excluded_after_freeze():
    finding = {"id": "F3", "severity": "MAJOR", "category": "style"}
    assert kbd._finding_allowed_after_freeze(finding, round_no=3, frozen_ids=set()) is False


def test_round3_new_capital_safety_finding_stays_eligible():
    finding = {"id": "F9", "severity": "CRITICAL", "category": "capital-safety"}
    assert kbd._finding_allowed_after_freeze(finding, round_no=3, frozen_ids=set()) is True


def test_round3_frozen_list_finding_stays_eligible_regardless_of_category():
    finding = {"id": "F1", "severity": "MAJOR", "category": "style"}
    assert kbd._finding_allowed_after_freeze(finding, round_no=3, frozen_ids={"F1"}) is True


def test_round3_missing_category_on_new_finding_fails_closed():
    """F4 (gate-2 round): missing category must fail CLOSED (stay blocking),
    not be silently treated as a safe/non-blocking category."""
    finding = {"id": "F5", "severity": "MAJOR"}
    assert kbd._finding_allowed_after_freeze(finding, round_no=3, frozen_ids=set()) is True


def test_filter_blocking_findings_splits_eligible_and_followups():
    findings = [
        {"id": "F1", "severity": "MAJOR", "category": "style"},       # frozen -> eligible
        {"id": "F3", "severity": "MAJOR", "category": "style"},       # new, non-safety -> followup
        {"id": "F9", "severity": "CRITICAL", "category": "security"},  # new, safety -> eligible
    ]
    eligible, followups = kbd.filter_blocking_findings(findings, round_no=3, frozen_ids={"F1"})
    assert [f["id"] for f in eligible] == ["F1", "F9"]
    assert [f["id"] for f in followups] == ["F3"]


def test_frozen_finding_ids_reads_only_round_1_and_2():
    runs = [
        {"metadata": kb._json_or_null({"findings": [{"id": "F1"}]})},
        {"metadata": kb._json_or_null({"findings": [{"id": "F2"}]})},
        {"metadata": kb._json_or_null({"findings": [{"id": "F3"}]})},  # round 3: NOT frozen
    ]
    frozen = kbd._frozen_finding_ids(runs)
    assert frozen == {"F1", "F2"}


def test_build_round_delta_context_round4_moves_new_non_safety_finding_to_followups(conn):
    task_id = kb.create_task(conn, title="review strategy-lab#220", assignee="reviewer")
    _insert_real_round(conn, task_id, "reviewer", metadata={"findings": [{"id": "F1", "severity": "MAJOR", "category": "style"}]})
    _insert_real_round(conn, task_id, "reviewer", metadata={"findings": [{"id": "F1", "severity": "MAJOR", "category": "style"}]})
    _insert_real_round(conn, task_id, "reviewer", metadata={
        "findings": [{"id": "F3", "severity": "MAJOR", "category": "style"}],
        "reviewed_sha": "cafef00d",
    })

    context = kbd.build_round_delta_context(conn, task_id, "reviewer")

    assert context is not None
    assert "Follow-ups" in context
    # F3 is new in round 3 and non-safety -> it must appear in the follow-ups
    # section, never in the "Prior findings" (blocking) section.
    followups_idx = context.index("Follow-ups")
    f3_idx = context.index("F3")
    assert f3_idx > followups_idx
    prior_section = context[:followups_idx]
    assert "F3" not in prior_section


# --- Rule 7: single source of truth for the base round cap --------------------

def test_non_capital_pr_base_cap_is_default():
    task = kbd._TaskLike(title="review strategy-lab#230", body="")
    assert kbd._review_round_cap_for(task) == kbd.REVIEW_ROUND_CAP_DEFAULT
    assert kbd.REVIEW_ROUND_CAP_DEFAULT == 3


def test_capital_repo_pr_base_cap_is_capital():
    task = kbd._TaskLike(title="review strategy-api#276", body="")
    assert kbd._review_round_cap_for(task) == kbd.REVIEW_ROUND_CAP_CAPITAL
    assert kbd.REVIEW_ROUND_CAP_CAPITAL == 2


def test_capital_marker_pr_base_cap_is_capital():
    task = kbd._TaskLike(title="[CAPITAL] review bff-api#999", body="")
    assert kbd._review_round_cap_for(task) == kbd.REVIEW_ROUND_CAP_CAPITAL


def test_docs_only_pr_base_cap_is_one():
    task = kbd._TaskLike(title="review bff-api#141", body="")
    pr_meta = {"files": ["docs/readme.md"], "additions": 50, "deletions": 50}
    assert kbd._review_round_cap_for(task, pr_meta) == kbd.REVIEW_ROUND_CAP_DOCS_ONLY
    assert kbd.REVIEW_ROUND_CAP_DOCS_ONLY == 1


def test_is_docs_only_review_has_no_size_cap_unlike_is_low_risk_review():
    task = kbd._TaskLike(title="review bff-api#141", body="")
    pr_meta = {"files": ["docs/readme.md"], "additions": 500, "deletions": 500}
    assert kbd._is_docs_only_review(task, pr_meta) is True
    assert kbd._is_low_risk_review(task, pr_meta) is False  # over _SMALL_PR_LINES


def test_is_docs_only_review_rejects_test_file():
    task = kbd._TaskLike(title="review bff-api#141", body="")
    pr_meta = {"files": ["docs/readme.md", "tests/test_x.py"]}
    assert kbd._is_docs_only_review(task, pr_meta) is False


def test_capital_pr_never_docs_only_regardless_of_files():
    task = kbd._TaskLike(title="[CAPITAL] review strategy-api#276", body="")
    pr_meta = {"files": ["docs/readme.md"]}
    assert kbd._is_docs_only_review(task, pr_meta) is False


def test_capital_round_2_escalates_non_capital_round_3_does_not(conn):
    """Regression guard for Rule 7: a capital PR escalates at round 2; the
    SAME round count on a non-capital PR must not escalate yet (cap 3)."""
    capital_task = kb.create_task(conn, title="review strategy-api#276", assignee="reviewer")
    for _ in range(2):
        _insert_real_round(conn, capital_task, "reviewer")
    # strategy-api#276 carries an owner override of 7 in _REVIEW_ROUND_CAP_BY_PR --
    # use a capital PR with no override to exercise the base cap of 2.
    assert kbd._REVIEW_ROUND_CAP_BY_PR.get("strategy-api#276") == 7
    capital_task2 = kb.create_task(conn, title="review risk-api#999", assignee="reviewer")
    for _ in range(2):
        _insert_real_round(conn, capital_task2, "reviewer")

    non_capital_task = kb.create_task(conn, title="review strategy-lab#240", assignee="reviewer")
    for _ in range(2):
        _insert_real_round(conn, non_capital_task, "reviewer")

    capital_reason = kbd._review_cost_gate(conn, capital_task2, "reviewer", dry_run=False)
    non_capital_reason = kbd._review_cost_gate(conn, non_capital_task, "reviewer", dry_run=False)

    assert capital_reason == "review_round_cap"
    assert non_capital_reason is None


def test_by_pr_override_still_honoured_on_top_of_new_base_caps(conn):
    """Regression guard (same bug class fixed once already in PR #1): the
    owner's per-PR override must still raise the cap above the new base of 3,
    not be silently discarded by the new constant."""
    pr_key = "strategy-lab#27"  # owner override of 8 in _REVIEW_ROUND_CAP_BY_PR
    assert kbd._REVIEW_ROUND_CAP_BY_PR.get(pr_key) == 8
    task_id = kb.create_task(conn, title=f"review {pr_key}", assignee="reviewer")
    for _ in range(5):  # above the new base default (3), below the override (8)
        _insert_real_round(conn, task_id, "reviewer")

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)

    assert reason is None
    assert kb.get_task(conn, task_id).status != "blocked"


# --- Rule 5: exactly 3-line owner escalation summary ---------------------------

def test_owner_escalation_summary_is_exactly_three_lines(conn):
    task_id = kb.create_task(conn, title="review strategy-lab#250", assignee="reviewer")
    for _ in range(3):
        _insert_real_round(conn, task_id, "reviewer")
    runs = kbd._review_verdict_runs(conn, "reviewer", {"strategy-lab#250"})

    summary = kbd._owner_escalation_summary(
        conn, "reviewer", "strategy-lab#250", "real_rounds", 3, 3, runs,
    )

    lines = summary.split("\n")
    assert len(lines) == 3
    assert "open findings after freeze" in lines[0].lower()
    assert "head sha" in lines[1].lower() and "ci" in lines[1].lower()
    assert "merge" in lines[2].lower() and "waive" in lines[2].lower() and "close" in lines[2].lower()


# --- F2 (gate-2 round): pr_meta is now wired into the live _review_cost_gate --
# call site, so Rule 6's docs-only cap is actually reachable in production. ---

def test_pr_meta_for_review_cap_short_circuits_with_no_published_pr(conn, monkeypatch):
    """No completion_contract on record yet: pr_meta_for_review_cap must return
    {} WITHOUT ever touching the network -- identical behaviour to before this
    was wired in, and no new network dependency for the common no-PR-yet case."""
    def _boom(*_a, **_k):
        raise AssertionError("fetch_pr_files must not be called with no published PR")
    monkeypatch.setattr(kb_pr_acceptance, "fetch_pr_files", _boom)
    task_id = kb.create_task(conn, title="review bff-api#200", assignee="reviewer")

    assert kbd_review_contract.pr_meta_for_review_cap(conn, "bff-api#200", "reviewer") == {}
    assert kb.get_task(conn, task_id).status != "blocked"


def test_pr_meta_for_review_cap_resolves_published_pr(conn, monkeypatch):
    """A published PR URL (completion_contract) resolves through to a real
    fetch_pr_files() call -- the gap F2 flagged as unreachable."""
    pr_key = "bff-api#9141"  # no _REVIEW_ROUND_CAP_BY_PR override -- exercises the base cap
    task_id = kb.create_task(conn, title=f"review {pr_key}", assignee="reviewer")
    conn.execute(
        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
        ("https://github.com/Stockstrading/bff-api/pull/9141", task_id),
    )
    conn.commit()
    monkeypatch.setattr(
        kb_pr_acceptance, "fetch_pr_files",
        lambda *_a, **_k: {"files": ["docs/readme.md"], "additions": 5, "deletions": 5},
    )

    meta = kbd_review_contract.pr_meta_for_review_cap(conn, pr_key, "reviewer")

    assert meta == {"files": ["docs/readme.md"], "additions": 5, "deletions": 5}


def test_live_review_cost_gate_reaches_docs_only_cap_via_pr_meta(conn, monkeypatch):
    """End-to-end: a published docs-only PR now actually hits the docs-only
    cap of 1 at the live _review_cost_gate call site -- before the F2 fix this
    classifier always saw files=[] and never returned True here."""
    pr_key = "bff-api#9241"  # no _REVIEW_ROUND_CAP_BY_PR override -- exercises the docs-only base cap
    task_id = kb.create_task(conn, title=f"review {pr_key}", assignee="reviewer")
    conn.execute(
        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
        ("https://github.com/Stockstrading/bff-api/pull/9241", task_id),
    )
    conn.commit()
    monkeypatch.setattr(
        kb_pr_acceptance, "fetch_pr_files",
        lambda *_a, **_k: {"files": ["docs/readme.md"], "additions": 5, "deletions": 5},
    )
    _insert_real_round(conn, task_id, "reviewer")  # 1 round already meets the docs-only cap of 1

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)

    assert reason == "review_round_cap"
    assert kb.get_task(conn, task_id).status == "blocked"


def test_live_review_cost_gate_stays_non_docs_cap_when_pr_meta_lookup_fails(conn, monkeypatch):
    """fetch_pr_files failing (network/auth error, returns None) must fall back
    to the default non-capital cap of 3, never silently escalate early."""
    pr_key = "bff-api#142"
    task_id = kb.create_task(conn, title=f"review {pr_key}", assignee="reviewer")
    conn.execute(
        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
        ("https://github.com/Stockstrading/bff-api/pull/142", task_id),
    )
    conn.commit()
    monkeypatch.setattr(kb_pr_acceptance, "fetch_pr_files", lambda *_a, **_k: None)
    _insert_real_round(conn, task_id, "reviewer")

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)

    assert reason is None


# --- F3/F4 (gate-2 round): finding identity + fail-closed category --------------

def test_finding_with_no_id_still_freezes_by_fallback_hash():
    """F3: a round-1/2 finding with no explicit id must still be recognised as
    frozen in round 3 -- it does not silently lose its blocking status just
    because the reviewer forgot to set an id."""
    verdict_runs = [
        {"metadata": kb._json_or_null({"findings": [
            {"file": "a.py", "line": 10, "severity": "MAJOR", "text": "leak"},
        ]})},
        {"metadata": kb._json_or_null({"findings": []})},
    ]
    frozen = kbd_review_contract._frozen_finding_ids(verdict_runs)
    assert len(frozen) == 1
    same_finding_round3 = {
        "file": "a.py", "line": 10, "severity": "MAJOR", "text": "leak", "category": "style",
    }
    assert kbd_review_contract._finding_allowed_after_freeze(
        same_finding_round3, round_no=3, frozen_ids=frozen,
    ) is True


def test_finding_identity_survives_line_drift_via_note_not_line():
    """F3: the fallback hash basis is (file, severity, note) -- a legitimate
    upstream edit that shifts the finding's line number must not change its
    identity (unlike the old (file, line, severity, text) basis, which lost
    the finding's frozen status whenever line drifted)."""
    verdict_runs = [
        {"metadata": kb._json_or_null({"findings": [
            {"file": "a.py", "line": 10, "severity": "MAJOR", "note": "sql injection"},
        ]})},
        {"metadata": kb._json_or_null({"findings": []})},
    ]
    frozen = kbd_review_contract._frozen_finding_ids(verdict_runs)
    assert len(frozen) == 1
    drifted_round3 = {
        "file": "a.py", "line": 14, "severity": "MAJOR", "note": "sql injection",
        "category": "style",
    }
    assert kbd_review_contract._finding_allowed_after_freeze(
        drifted_round3, round_no=3, frozen_ids=frozen,
    ) is True


def test_malformed_non_string_category_fails_closed():
    """F4: a malformed category (list/dict from an upstream schema slip) must
    fail closed exactly like a missing/blank category -- not silently fall
    through str() coercion into a never-matching, non-blocking verdict."""
    finding = {"id": "F7", "severity": "MAJOR", "category": ["capital-safety"]}
    assert kbd_review_contract._finding_allowed_after_freeze(
        finding, round_no=3, frozen_ids=set(),
    ) is True
    finding_dict_category = {"id": "F8", "severity": "MAJOR", "category": {"name": "style"}}
    assert kbd_review_contract._finding_allowed_after_freeze(
        finding_dict_category, round_no=3, frozen_ids=set(),
    ) is True


# --- F10 (gate-2 round): PR URL resolves from the review card's OWN body, no --
# completion_contract binding ever required, and a mismatched contract is never
# trusted. --------------------------------------------------------------------

def test_pr_meta_resolves_pr_url_from_review_cards_own_body_no_contract_bound(conn, monkeypatch):
    """F10: the ordinary single-card kanban_request_review/kanban_request_changes
    cycle never binds completion_contract to a PR URL. pr_meta_for_review_cap
    must still reach the docs-only cap by reading the PR URL straight out of
    the review card's own body (the common 'PR: https://...' hand-off shape)."""
    pr_key = "bff-api#9341"
    task_id = kb.create_task(
        conn, title=f"review {pr_key}",
        body="PR: https://github.com/Stockstrading/bff-api/pull/9341 -- docs only change.",
        assignee="reviewer",
    )
    task = kb.get_task(conn, task_id)
    assert task.completion_contract == "local-only"  # never bound -- F10's premise
    monkeypatch.setattr(
        kb_pr_acceptance, "fetch_pr_files",
        lambda *_a, **_k: {"files": ["docs/readme.md"], "additions": 5, "deletions": 5},
    )

    meta = kbd_review_contract.pr_meta_for_review_cap(
        conn, pr_key, "reviewer", card_text=f"{task.title}\n{task.body}",
    )

    assert meta == {"files": ["docs/readme.md"], "additions": 5, "deletions": 5}


def test_pr_url_for_key_never_trusts_contract_bound_to_a_different_pr(conn):
    """F10 part 2: a completion_contract found on a same-repo card must encode
    THIS pr_key's own number before it is trusted -- a contract that resolved
    to a different PR in the same repo (scope changed mid-flight) is treated
    exactly like no contract found, never fed into this PR's classification."""
    pr_key = "bff-api#9341"
    other_task_id = kb.create_task(conn, title=f"dev work for {pr_key}", assignee="backend-dev")
    conn.execute(
        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
        ("https://github.com/Stockstrading/bff-api/pull/8888", other_task_id),
    )
    conn.commit()

    resolved = kbd._pr_url_for_key(conn, pr_key)

    assert resolved is None


def test_live_review_cost_gate_reaches_docs_only_cap_via_card_body_pr_url(conn, monkeypatch):
    """End-to-end F10: a docs-only PR whose dev task never bound a
    completion_contract still hits the docs-only cap of 1 at the live
    _review_cost_gate call site, because the PR URL is read from the review
    card's own body."""
    pr_key = "bff-api#9441"
    task_id = kb.create_task(
        conn, title=f"review {pr_key}",
        body="PR: https://github.com/Stockstrading/bff-api/pull/9441",
        assignee="reviewer",
    )
    monkeypatch.setattr(
        kb_pr_acceptance, "fetch_pr_files",
        lambda *_a, **_k: {"files": ["docs/readme.md"], "additions": 5, "deletions": 5},
    )
    _insert_real_round(conn, task_id, "reviewer")  # 1 round already meets the docs-only cap of 1

    reason = kbd._review_cost_gate(conn, task_id, "reviewer", dry_run=False)

    assert reason == "review_round_cap"
    assert kb.get_task(conn, task_id).status == "blocked"


# --- F5 (gate-2 round): deterministic round ordering on an ended_at tie -------

def test_review_verdict_runs_tiebreaks_deterministically_on_ended_at_tie(conn):
    """Two verdict runs ending in the exact same second must not swap which
    one is round 1 vs round 2 -- r.id is the deterministic secondary key."""
    task_id = kb.create_task(conn, title="review bff-api#300", assignee="reviewer")
    now = int(time.time())
    _insert_real_round(
        conn, task_id, "reviewer", started_at=now - 600, ended_at=now - 400,
        metadata={"findings": [{"id": "first"}]},
    )
    _insert_real_round(
        conn, task_id, "reviewer", started_at=now - 600, ended_at=now - 400,
        metadata={"findings": [{"id": "second"}]},
    )

    runs = kbd._review_verdict_runs(conn, "reviewer", {"bff-api#300"})

    assert len(runs) == 2
    assert kb._json_dict(runs[0]["metadata"])["findings"][0]["id"] == "first"
    assert kb._json_dict(runs[1]["metadata"])["findings"][0]["id"] == "second"
