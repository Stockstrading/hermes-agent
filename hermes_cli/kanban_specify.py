"""Kanban triage specifier — flesh out a one-liner into a real spec.

``hermes kanban specify [task_id | --all]`` asks the auxiliary LLM for a
tightened title + concrete body for a Triage task, then flips it
``triage -> todo`` via ``kanban_db.specify_triage_task``.

Mirrors ``hermes_cli/goals.py``: same aux-client pattern, same "empty config
=> skip, don't crash" tolerance. One shot, no retry loop. JSON mode is not
requested (works on providers without it); the parse is lenient and falls
back to "whole reply is the body" so a malformed reply never strands a task.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

from utils import env_int

HERMES_KANBAN_SPECIFY_MAX_TOKENS = max(1500, env_int("HERMES_KANBAN_SPECIFY_MAX_TOKENS", 6000))

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Kanban triage specifier for the Hermes Agent board.
A user dropped a rough idea into the Triage column. Your job is to turn it
into a concrete, actionable task spec that an autonomous worker can pick up
and execute without further clarification.

Output a single JSON object with exactly two keys:

  {
    "title": "<tightened task title, <= 80 chars, imperative voice>",
    "body":  "<multi-line spec, see structure below>"
  }

The body MUST include these sections, each prefixed with a bold markdown
heading, in this order:

  **Goal** — one sentence, user-facing outcome.
  **Approach** — 2-5 bullets on how a worker should tackle it.
  **Acceptance criteria** — checklist of concrete, verifiable conditions.
  **Out of scope** — short list of things NOT to touch (omit if nothing
      obvious; never invent scope creep).

Rules:
  - Keep the tightened title close in meaning to the original idea — do
    NOT invent a different project.
  - If the original idea is already detailed, preserve its substance and
    just reformat into the sections above.
  - Never add invented requirements the user didn't hint at.
  - No preamble, no closing remarks, no code fences around the JSON.
  - Output only the JSON object and nothing else.
"""


_USER_TEMPLATE = """Task id: {task_id}
Current title: {title}
Current body:
{body}
"""


@dataclass
class SpecifyOutcome:
    """Result of specifying a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    new_title: Optional[str] = None


# ---------------------------------------------------------------------------
# Rule 6 routing guard (ADR card t_f526b652, review-loop-convergence): a
# log-only detector for an assignee that crosses the dev/review-only line.
# Shared by both kanban_specify.specify_task and kanban_decompose (which
# imports these from here). Per the ADR's open question 3 / owner decision,
# this phase only comments + logs; it does NOT rewrite the assignee yet.
# ---------------------------------------------------------------------------

_DEV_PROFILE_NAMES = frozenset({"backend-dev", "frontend-dev", "devops-sre"})

_DEFAULT_REVIEW_ONLY_PROFILES = ("reviewer", "release-reviewer")

_REVIEW_TITLE_RE = re.compile(r"^\s*(review|design note|audit)\b", re.IGNORECASE)
_PR_REF_RE = re.compile(r"pull/\d+|pull\s*request\s*#?\d+|\bpr\s*#\d+", re.IGNORECASE)


def _is_review_request(title: Optional[str], body: Optional[str]) -> bool:
    """True when a card's text marks it as a review request: the title opens with a
    review verb (Review/Design note/Audit), or the body names an existing PR to review.
    Mirrors the ADR's rule-6 heuristic (card t_f526b652)."""
    if isinstance(title, str) and _REVIEW_TITLE_RE.match(title):
        return True
    if isinstance(body, str) and _PR_REF_RE.search(body):
        return True
    return False


def review_only_profiles_from_config(cfg: dict) -> set[str]:
    """``kanban.review_only_profiles`` as a set, falling back to the code default when the
    config value is missing or malformed (never raises)."""
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    raw = kanban_cfg.get("review_only_profiles")
    if isinstance(raw, list) and raw:
        names = {p.strip() for p in raw if isinstance(p, str) and p.strip()}
        if names:
            return names
    return set(_DEFAULT_REVIEW_ONLY_PROFILES)


def _routing_guard_mismatch(
    assignee: Optional[str], title: Optional[str], body: Optional[str], *,
    review_only_profiles: set[str], default_assignee: str,
) -> Optional[str]:
    """Rule 6 (ADR t_f526b652): log-only detector for an LLM-picked assignee that crosses the
    dev/review-only line in either direction. Returns an audit-comment body, or None when the
    pick is consistent. Does NOT rewrite the assignee — rewriting is a deliberate follow-up once
    the log-only false-positive rate has been observed (owner decision on t_4f89767e)."""
    if not isinstance(assignee, str) or not assignee.strip():
        return None
    is_review = _is_review_request(title, body)
    if assignee in review_only_profiles and not is_review:
        msg = (
            f"routing-guard mismatch: `{assignee}` picked for a non-review card, "
            f"would reassign to `{default_assignee}` (log-only, kanban.review_only_profiles)."
        )
    elif assignee in _DEV_PROFILE_NAMES and is_review:
        msg = (
            f"routing-guard mismatch: `{assignee}` picked for a review-request card, "
            "would reassign to a review-only profile (log-only, kanban.review_only_profiles)."
        )
    else:
        return None
    logger.info(
        "kanban routing-guard mismatch: assignee=%r is_review=%s title=%r",
        assignee, is_review, (title or "")[:120],
    )
    return msg


def _truncate(text: str, limit: int) -> str:
    # Plain length clamp for LLM prompt fields; these never reach a terminal, so no escape stripping here.
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_blob(raw: str, fence_re: re.Pattern = _FENCE_RE) -> Optional[dict]:
    """Lenient JSON object extraction: strip code fences, take the first ``{``
    to the last ``}``. None if nothing parses to a dict."""
    if not raw:
        return None
    stripped = fence_re.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        val = json.loads(stripped[first : last + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return val if isinstance(val, dict) else None


def _nonblank(v) -> Optional[str]:
    return v if isinstance(v, str) and v.strip() else None


def _title_body(parsed: dict) -> tuple[Optional[str], Optional[str]]:
    """``(title, body)`` from an LLM reply: title stripped, body verbatim,
    either None when missing/blank."""
    title = _nonblank(parsed.get("title"))
    return (title.strip() if title else None), _nonblank(parsed.get("body"))


def _profile_author(default: str = "specifier") -> str:
    """Same identity contract as ``hermes_cli.kanban._profile_author``; ``$USER`` as the last
    resort for a human running the CLI outside any profile."""
    from hermes_cli.profiles import current_profile_name
    return current_profile_name() or os.environ.get("USER") or default


def _load_triage_task(task_id: str) -> tuple[Optional[kb.Task], str]:
    """``(task, "")`` when the task exists and is in triage, else ``(None, reason)``."""
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    if task is None:
        return None, "unknown task id"
    if task.status != "triage":
        return None, f"task is not in triage (status={task.status!r})"
    return task, ""


def _task_prompt_fields(task: kb.Task) -> dict[str, str]:
    """Bounded ``task_id``/``title``/``body`` for the user prompt templates."""
    return {
        "task_id": task.id,
        "title": _truncate(task.title or "", 400),
        "body": _truncate(task.body or "(no body)", 4000),
    }


def _call_aux(verb: str, task_id: str, *, aux_task: str, system: str, user: str,
              max_tokens: int, timeout: int, log: logging.Logger = logger) -> tuple[Optional[str], str]:
    """One auxiliary LLM call; ``(reply_text, "")`` or ``(None, reason)``.

    ``call_llm`` applies all ``auxiliary.<aux_task>.*`` config (provider/model/
    base_url, extra_body, reasoning_effort, retries). Imported lazily so a
    missing aux client degrades to a skip instead of an import-time crash.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:  # pragma: no cover — import smoke test
        log.debug("%s: auxiliary client import failed: %s", verb, exc)
        return None, "auxiliary client unavailable"
    # Specify/decompose run outside any agent turn (CLI, dashboard route, gateway watcher), so no
    # conversation affinity scope is bound and the relay-affinity headers (x-opencode-session, the
    # OpenRouter/Portal sticky key) are omitted — the OpenCode Go relay rejects that with 400
    # MissingSessionID (#112043). Declare a per-task scope, but only when none is already bound so an
    # in-turn caller keeps its conversation's key.
    from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
    affinity_token = None if get_affinity_scope() else set_affinity_scope(f"kanban:{task_id}")
    try:
        # Route through call_llm so auxiliary.triage_specifier.* config (provider/model/base_url,
        # extra_body, reasoning_effort, retries) all apply — the direct-create path dropped extra_body
        # (#35566).
        resp = call_llm(
            task=aux_task,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as exc:
        suffix = " — skipping" if verb == "specify" else ""
        log.info("%s: API call failed for %s (%s)%s", verb, task_id, exc, suffix)
        return None, f"LLM error: {type(exc).__name__}"
    finally:
        if affinity_token is not None:
            reset_affinity_scope(affinity_token)
    try:
        return resp.choices[0].message.content or "", ""
    except Exception:
        return "", ""


def specify_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> SpecifyOutcome:
    """Specify one triage task and promote it to ``todo``. Expected failures
    (not in triage, no aux client, API error, malformed reply) surface as
    ``ok=False`` so an ``--all`` sweep continues."""
    task, reason = _load_triage_task(task_id)
    if task is None:
        return SpecifyOutcome(task_id, False, reason)

    raw, reason = _call_aux(
        "specify", task_id, aux_task="triage_specifier", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(**_task_prompt_fields(task)),
        max_tokens=HERMES_KANBAN_SPECIFY_MAX_TOKENS, timeout=timeout or 120,
    )
    if raw is None:
        return SpecifyOutcome(task_id, False, reason)
    raw = raw.strip()

    parsed = _extract_json_blob(raw)
    if parsed is None:
        # Whole reply becomes the body; the user can edit afterward.
        if not raw:
            return SpecifyOutcome(task_id, False, "LLM returned an empty response")
        new_title, new_body = None, raw
    else:
        new_title, new_body = _title_body(parsed)
        if new_body is None and new_title is None:
            return SpecifyOutcome(task_id, False, "LLM response missing title and body")

    audit_author = author or _profile_author()
    with kbc.connect_closing() as conn:
        ok = kb.specify_triage_task(
            conn,
            task_id,
            title=new_title,
            body=new_body,
            author=audit_author,
        )
        if ok:
            # Rule 6 routing guard (log-only phase): specify_task never changes the
            # assignee itself, so check the assignee the task already carries against
            # the (possibly just-tightened) title/body.
            try:
                from hermes_cli.config import load_config_readonly
                cfg = load_config_readonly()
            except Exception:
                cfg = {}
            kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
            default_assignee = (kanban_cfg.get("default_assignee") or "").strip() or "the default profile"
            mismatch = _routing_guard_mismatch(
                task.assignee,
                new_title if new_title is not None else task.title,
                new_body if new_body is not None else task.body,
                review_only_profiles=review_only_profiles_from_config(cfg),
                default_assignee=default_assignee,
            )
            if mismatch:
                kb.add_comment(conn, task_id, audit_author, mismatch)
    if not ok:
        # Race: promoted/archived between our read and the write.
        return SpecifyOutcome(task_id, False, "task moved out of triage before promotion")
    return SpecifyOutcome(task_id, True, "specified", new_title=new_title)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Task ids in the triage column; ``tenant`` narrows the sweep.

    Excludes cards whose most recent event is ``block_loop_detected`` -- that
    escalation is for a human (or ``kanban_unblock``), not for auto-specify. A
    card unblocked and later re-escalated for an unrelated, newer reason is
    still returned (see ``kb.list_tasks(exclude_block_loop=True)``).
    """
    with kbc.connect_closing() as conn:
        tasks = kb.list_tasks(
            conn, status="triage", tenant=tenant, include_archived=False, exclude_block_loop=True,
        )
    return [t.id for t in tasks]
