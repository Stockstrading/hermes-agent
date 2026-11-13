"""Review contract v2 helpers (ADR-review-contract-v2.md, card t_cca086d6 /
t_3183cd6d): round-cap classification, evidence-gated hand-off, and the
frozen-findings rule. Split out of ``hermes_cli.kanban_db_dispatch`` (that
facade's ``FILE_LINES`` budget is already over target) -- origin-resident
predicates (``_is_capital_review``, ``_findings_from_metadata``) are reached
late-bound via a function-local import of the facade (import-cycle breaking),
same convention the facade itself uses for ``kanban_db``.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as _kb

# Rule 7 (single source of truth): three named round caps replace the single
# flat ``_REVIEW_ROUND_CAP`` that used to live in kanban_db_dispatch.py. Every
# other place that states a cap (SOUL.md prose, the review-round-cap.py cron)
# must read one of these three, never restate a digit of its own -- numbers
# pending explicit owner sign-off, see the ADR.
REVIEW_ROUND_CAP_DEFAULT = 3    # non-capital PR: round 3 unapproved -> owner queue
REVIEW_ROUND_CAP_CAPITAL = 2    # _CAPITAL_REPOS / [CAPITAL] marker: round 2 unapproved -> owner queue
REVIEW_ROUND_CAP_DOCS_ONLY = 1  # Rule 6 (_is_docs_only_review): one gate, one round


@dataclass
class _TaskLike:
    """Minimal (id, title, body) shim so dispatch-time code holding only a raw
    sqlite3.Row can reuse the Task-shaped predicates below without
    constructing a real ``Task``."""

    id: Optional[str] = None
    title: str = ""
    body: str = ""


_DOCS_ONLY_PATH_RE = re.compile(r"(^|/)docs/|\.md$", re.IGNORECASE)


def _is_docs_only_review(task, pr_meta: dict) -> bool:
    """Rule 6: True only for a non-capital PR touching EXCLUSIVELY docs/** or
    *.md files -- size-independent (stricter than ``_is_low_risk_review``,
    which also allows tests/CI paths and caps at ``_SMALL_PR_LINES``).
    Capital-sensitive PRs never qualify regardless of file list; an unknown
    file list fails closed (never docs-only), same convention as
    ``_is_low_risk_review``."""
    from hermes_cli.kanban_db_dispatch import _CAPITAL_DENY_SUBSTRINGS, _is_capital_review

    if _is_capital_review(task):
        return False
    files = pr_meta.get("files") or []
    if not files:
        return False
    if not all(_DOCS_ONLY_PATH_RE.search(f) for f in files):
        return False
    return not any(marker in f.lower() for f in files for marker in _CAPITAL_DENY_SUBSTRINGS)


def _review_round_cap_for(task, pr_meta: Optional[dict] = None) -> int:
    """Rule 7: the base round cap for this PR's class, before any
    ``_REVIEW_ROUND_CAP_BY_PR`` owner override is applied on top."""
    from hermes_cli.kanban_db_dispatch import _is_capital_review

    if _is_docs_only_review(task, pr_meta or {}):
        return REVIEW_ROUND_CAP_DOCS_ONLY
    if _is_capital_review(task):
        return REVIEW_ROUND_CAP_CAPITAL
    return REVIEW_ROUND_CAP_DEFAULT


def pr_meta_for_review_cap(
    conn: sqlite3.Connection, pr: str, assignee: str, *, card_text: Optional[str] = None,
) -> dict:
    """Real ``{"files", "additions", "deletions"}`` for Rule 6's
    ``_is_docs_only_review`` at the live ``_review_cost_gate`` call site (F2,
    hermes-agent PR #5 gate-2): before this, no ``pr_meta`` was ever fetched
    there, so ``_is_docs_only_review`` always saw ``files=[]`` and the
    docs-only cap was unreachable in production. Resolves the published PR
    URL the same way :func:`hermes_cli.kanban_db_dispatch._owner_escalation_summary`
    already does -- falling back to ``card_text`` (the review card's OWN
    title/body, F10) when no task's ``completion_contract`` has ever been
    bound to this PR, which is the normal state for an ordinary single-card
    review-lane cycle -- then one GraphQL round trip via ``kanban_pr_acceptance``.
    Fails closed to ``{}`` (identical behaviour to before this was wired in)
    whenever no PR URL resolves at all or the lookup fails for any
    reason -- both classifiers already treat an empty file list as "never
    docs-only/low-risk"."""
    from hermes_cli import kanban_db_dispatch as _kbd
    from hermes_cli import kanban_pr_acceptance as _kb_pr_acceptance

    pr_url = _kbd._pr_url_for_key(conn, pr, fallback_text=card_text)
    if not pr_url:
        return {}
    try:
        return _kb_pr_acceptance.fetch_pr_files(pr_url, pr_url, assignee=assignee) or {}
    except Exception as exc:  # health: allow BLE001 -- classifier input must never stop the dispatcher
        _kb._log.warning("pr file-list lookup skipped for %s: %s", pr, exc)
        return {}


_EVIDENCE_SCOPE_RE = re.compile(r"(?im)^\s*scope\s*:")
_EVIDENCE_FILES_RE = re.compile(r"(?im)^\s*files?\s*:")
_EVIDENCE_COMMANDS_RE = re.compile(r"(?im)^\s*commands?[ -]?run\s*:")
_EVIDENCE_BASE_RE = re.compile(r"base\s*:\s*origin/main@[0-9a-f]{6,40}", re.IGNORECASE)

# Rule 1 cutover (ADR-review-contract-v2.md §3: "Do not make the evidence gate
# retroactive -- only applies to request_review calls made after this ships").
# ``None`` means the gate is fully disabled -- the state this constant must
# ship in, because the SOUL.md diffs that teach backend-dev/frontend-dev to
# actually produce a Scope/Files/Commands-run/base hand-off are "written out
# in the PR for the owner to apply," not live yet (ground rule 9: SOUL.md is
# owner-edited, never live-patched by an agent). Enforcing today would bounce
# every card already sitting in the reviewer/release-reviewer queue across
# all repos -- those hand-offs were written under the OLD contract and have
# no way to know the new fields exist. The owner flips this to
# ``int(time.time())`` (its own commit, once the SOUL diffs are confirmed
# live) so only hand-offs written AFTER that moment are held to the new bar;
# anything already in flight is grandfathered in by comparing its
# ``review_requested`` run's ``started_at`` against the cutoff.
EVIDENCE_GATE_ROLLOUT_AT: Optional[int] = None


def _handoff_has_evidence(conn: sqlite3.Connection, task_id: str) -> tuple[bool, Optional[str]]:
    """Rule 1: the ``kanban_request_review`` hand-off that put this card in the
    review lane must carry a scope line, a file list, a commands-run block
    with output, and a ``base: origin/main@<sha>`` freshness statement -- else
    the reviewer spends a paid round on a hand-off it cannot act on. Returns
    ``(True, None)`` when all four are present, when the gate is not yet
    rolled out (``EVIDENCE_GATE_ROLLOUT_AT`` is ``None``), when the hand-off
    predates the rollout cutover (not retroactive -- ADR §3), or when no
    hand-off is on record at all (e.g. an owner/test-created card -- never
    bounce on absence of data this gate cannot see). Else ``(False, <missing
    item>)`` naming the FIRST missing item for an actionable bounce reason.
    Fails open on any lookup error: this is a hand-off quality gate, not a
    safety check."""
    if EVIDENCE_GATE_ROLLOUT_AT is None:
        return True, None
    try:
        row = conn.execute(
            "SELECT summary, metadata, started_at FROM task_runs WHERE task_id = ? "
            "AND outcome = 'review_requested' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if row is None:
            return True, None
        started_at = row["started_at"]
        if started_at is None or started_at < EVIDENCE_GATE_ROLLOUT_AT:
            return True, None
        text = "\n".join(filter(None, [
            _kb._lossy_text(row["summary"]) or "",
            _kb._lossy_text(row["metadata"]) or "",
        ]))
        if not _EVIDENCE_SCOPE_RE.search(text):
            return False, "a scope line (Scope: ...)"
        if not _EVIDENCE_FILES_RE.search(text):
            return False, "a file list (Files: ...)"
        if not _EVIDENCE_COMMANDS_RE.search(text):
            return False, "a commands-run block with output (Commands run: ...)"
        if not _EVIDENCE_BASE_RE.search(text):
            return False, "a freshness statement (base: origin/main@<sha>)"
        return True, None
    except Exception as exc:  # health: allow BLE001 -- evidence gate must never stop the dispatcher; fails open
        _kb._log.warning("handoff evidence check skipped for %s: %s", task_id, exc)
        return True, None


# --- Rule 2/3 extension + Rule 4: proof-of-fix and frozen findings ------------
# Two independent gaps on top of the existing delta-only review: an
# "addressed" finding needs proof (test_ref), and after round 2 a brand-new
# finding only reopens the PR's blocking list when it is genuinely
# safety-critical -- everything else is a follow-up.
_FREEZE_ROUND_CUTOFF = 2
_FREEZE_ALLOWED_CATEGORIES = {"capital-safety", "data-loss", "security"}


def _finding_is_addressed(finding: dict) -> bool:
    """Rule 2/3: an 'addressed' finding only counts as resolved when it
    carries a non-blank ``test_ref`` (the test that failed before, passes
    now); missing ``test_ref`` keeps it open regardless of the reported
    ``status``."""
    if str(finding.get("status", "open")).lower() != "addressed":
        return False
    return bool(str(finding.get("test_ref") or "").strip())


def _finding_key(finding: dict) -> str:
    """Stable identity for a finding across review rounds: its explicit
    ``id`` when the reviewer set one, else a deterministic fallback hash of
    ``(file, severity, note)`` (F3, hermes-agent PR #5 gate-2 round 2). The
    findings schema actually used throughout this pipeline is ``id, severity,
    file, line, status, category, found_by, note`` -- there is no ``text``
    field, so hashing it always contributed an empty string. ``line`` is
    deliberately excluded from the basis too: a legitimate upstream edit that
    shifts a line number must not change an id-less finding's identity and
    silently drop its frozen/blocking status by round 3."""
    fid = str(finding.get("id") or "").strip()
    if fid:
        return fid
    basis = "|".join(str(finding.get(k) or "") for k in ("file", "severity", "note"))
    return "fallback:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()


def _frozen_finding_ids(verdict_runs: list) -> set[str]:
    """Keys (see :func:`_finding_key`) reported in round 1 and round 2 (the
    first two verdict runs) -- the frozen list Rule 4 locks in after round 2."""
    from hermes_cli.kanban_db_dispatch import _findings_from_metadata

    ids: set[str] = set()
    for run in verdict_runs[:_FREEZE_ROUND_CUTOFF]:
        for f in _findings_from_metadata(run["metadata"]):
            ids.add(_finding_key(f))
    return ids


def _finding_allowed_after_freeze(finding: dict, round_no: int, frozen_ids: set[str]) -> bool:
    """Rule 4: for ``round_no`` > 2, a finding not already present in the
    round-1/round-2 frozen list (matched by :func:`_finding_key`, so a
    finding missing an explicit ``id`` still freezes/unfreezes consistently
    instead of silently losing its status -- F3) only counts (stays eligible
    to block THIS PR) if its ``category`` is capital-safety, data-loss, or
    security. A frozen-list finding is always eligible -- Rule 4 gates NEW
    findings only. A ``category`` that is missing/blank, OR not a string at
    all (e.g. a list/dict from an upstream schema slip), fails closed (treated
    as blocking, same convention as ``_is_docs_only_review``/
    ``_is_low_risk_review``'s "unknown data is never safe") until a reviewer
    explicitly classifies it as a recognised string category (F4)."""
    if round_no <= _FREEZE_ROUND_CUTOFF:
        return True
    if _finding_key(finding) in frozen_ids:
        return True
    raw_category = finding.get("category")
    if raw_category is None or not isinstance(raw_category, str) or not raw_category.strip():
        return True  # unclassified or malformed (non-string): fail closed, stays blocking
    category = raw_category.strip().lower()
    return category in _FREEZE_ALLOWED_CATEGORIES


def filter_blocking_findings(findings: list, round_no: int, frozen_ids: set[str]) -> tuple[list, list]:
    """Rule 4: split a round's findings into (still eligible for
    ``should_block_on_findings``, follow-up only -- never reopens this PR's
    blocking list, surfaced as a ``Follow-ups:`` line / separate card
    instead)."""
    eligible: list = []
    followups: list = []
    for f in findings:
        (eligible if _finding_allowed_after_freeze(f, round_no, frozen_ids) else followups).append(f)
    return eligible, followups
# --- end Rule 2/3 extension + Rule 4 -------------------------------------------
