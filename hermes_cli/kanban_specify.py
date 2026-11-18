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
  - If the current title or body names a repository + PR (e.g. "etoro-foo PR #12"
    or a github.com/.../pull/12 URL), keep that exact repo + PR reference in the
    new title and keep every PR URL in the body. Keep tags such as [CAPITAL],
    [CA] and [needs-human-ok] in the title.
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
    """``(task, "")`` when the task exists, is in triage, and carries no
    unresolved ``block_loop_detected`` escalation, else ``(None, reason)``.

    Re-applies the same exclusion ``list_triage_ids()`` used at enumeration
    time, right before the (slow) auxiliary LLM call -- the enumeration can be
    minutes stale by the time an ``--all`` sweep reaches this id, and a
    `block_loop_detected` escalation that landed in between must still refuse
    here, not just at the DB CAS after the LLM call (#PR12 release-gate
    MAJOR 2)."""
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        if task is not None and task.status == "triage" and kb.task_has_unresolved_block_loop(conn, task_id):
            return None, "task has an unresolved block_loop_detected escalation"
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


_KEEP_TAG_RE = re.compile(r"\[(?:CAPITAL|CA|CA-adjacent|needs-human-ok|Board-approved)\]", re.IGNORECASE)
_GH_PR_URL_RE = re.compile(r"https?://github\.com/[^/\s]+/([^/\s]+)/pull/(\d+)", re.IGNORECASE)
# PR12 release-gate round-6 MAJOR 1: `(?<![A-Za-z])` is not a token boundary --
# a digit or underscore right before the token ("3D PR #12", "123repo PR #8",
# "_internal PR #4") still let the match start mid-word and carve a fake repo
# out of the tail (the round-5 defect verbatim). `(?<![\w.-])` is the real
# boundary: nothing in `[\w.-]` (letters, digits, `_`, `.`, `-`) may precede the
# captured token's first character.
_TITLE_PR_REF_RE = re.compile(
    r"(?<![\w.-])([A-Za-z][\w.-]*)\s+PR\s*#(\d+)\b", re.IGNORECASE)
# Round-7 MAJOR 1: an explicit ``owner/repo`` prefix is actual evidence (a human or tool
# wrote a path, not a sentence), so it bypasses the ambiguous-word filter entirely --
# fixes the gate-2 finding that even ``Stockstrading/capital PR #6`` yielded nothing,
# because only the bare post-slash token (``capital``, a denied word) ever reached the
# matcher. Both segments are ``[\w.-]+`` (no further slashes) so greedy matching can't
# eat a third path component.
_OWNER_REPO_PR_REF_RE = re.compile(
    r"(?<![\w./-])([A-Za-z0-9][\w.-]*)/([A-Za-z0-9][\w.-]*)\s+PR\s*#(\d+)\b", re.IGNORECASE)


def _regular_inflections(base: str) -> frozenset[str]:
    """One base verb's regular English inflections: -s/-es/-ed/-ing, the y->i shift
    (``verify``->``verified``), and CVC consonant doubling for a short, closed set of
    single-syllable verbs that need it (``stop``->``stopped``/``stopping``, never
    ``stoped``/``stoping`` -- the round-7 NIT). Doubling only fires for the verbs that
    are actually single-syllable (stress always on the final syllable), so it can't
    mis-double a multi-syllable word like ``monitor`` (->``monitored``, not
    ``monitorred``) the way a blind CVC regex would."""
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        stem = base[:-1]
        return frozenset({base, stem + "ied", base + "ing", stem + "ies"})
    if base.endswith("e"):
        stem = base[:-1]
        return frozenset({base, base + "d", stem + "ing", base + "s"})
    if base in _DOUBLING_SINGLE_SYLLABLE_VERBS:
        doubled = base + base[-1]
        return frozenset({base, doubled + "ed", doubled + "ing", base + "s"})
    s_form = base + "es" if base.endswith(("s", "x", "z", "ch", "sh")) else base + "s"
    return frozenset({base, base + "ed", base + "ing", s_form})


# Single-syllable verbs ending consonant-vowel-consonant whose final consonant doubles
# before -ed/-ing (``stop``->``stopped``). Deliberately a short, explicit set rather than
# a CVC-shape regex: a shape-only rule would also (wrongly) double multi-syllable verbs
# like ``monitor``/``confirm`` whose stress isn't on the final syllable.
_DOUBLING_SINGLE_SYLLABLE_VERBS = frozenset({"stop", "ship"})


def _verb_forms(base: str) -> frozenset[str]:
    """Regular inflections for one base verb, plus the opposite -ize/-ise spelling
    (``analyze``/``analyse``, ``finalize``/``finalise``) so a single American-spelled
    entry in ``_PR_REF_NON_REPO_VERBS`` also covers the British form, instead of a
    second hand-typed entry that round-7 proved gets missed (``Summarise PR #3``)."""
    forms = _regular_inflections(base)
    if base.endswith("ize"):
        forms |= _regular_inflections(base[:-3] + "ise")
    elif base.endswith("yze"):
        forms |= _regular_inflections(base[:-3] + "yse")
    elif base.endswith("ise"):
        forms |= _regular_inflections(base[:-3] + "ize")
    elif base.endswith("yse"):
        forms |= _regular_inflections(base[:-3] + "yze")
    return forms


# Round-6/round-7 MAJOR 1: measured against the real card titles in ~/.hermes/kanban.db,
# ordinary English verbs that commonly precede "PR #N" in a PR-workflow sentence
# fabricate a repo (verify#560, rebase#103, re-review#210, resolve#324, update#319,
# deploy#1, triage#1, analyze#312, finish#440, final#217 via "finalize", ...). These are
# the PR-workflow verbs that measurement surfaced, each expanded to its regular
# inflections below (including the opposite -ize/-ise spelling, see `_verb_forms`).
# "keep"/"kept" and the irregular doublers "split" (splitting, not splitted) and
# "re-run" (reran, not re-runned) are listed as literal words instead -- regular
# inflection would mangle them.
_PR_REF_NON_REPO_VERBS = frozenset({
    "merge", "rebase", "revert", "backport", "verify", "test", "open", "land",
    "close", "inspect", "reconcile", "resolve", "push", "summarize", "approve",
    "retarget", "fold", "prepare", "refresh", "assess", "audit", "complete",
    "correct", "harden", "rework", "stack", "block", "gate", "update",
    # Round-7: verbs measured as still fabricating a key (20 verbs from the gate-2/3
    # finding, minus "split"/"re-run", listed below as literal irregular forms).
    "deploy", "triage", "document", "unblock", "finish", "ship", "reject", "draft",
    "amend", "squash", "finalize", "confirm", "recheck", "sync", "monitor",
    "validate", "analyze",
    # Round-7: further verbs the same measurement caught fabricating a key over
    # real titles (checkout#103, port#114, reduce#232, clear#115, duplicate#431,
    # exceed#221) -- not in the required 20 but measured all the same.
    "checkout", "port", "reduce", "clear", "duplicate", "exceed",
})
# Mandatory whitespace plus a real token boundary still leaves every ordinary
# English word/verb/preposition that happens to sit right before "PR #N" in a
# sentence ("Review PR #123", "... comments on PR #45", "... regression in PR
# #5") indistinguishable from a real repo name using regex shape alone -- there
# is no configured repo list to validate against here (this module is
# deliberately self-contained, see `_card_pr_keys`'s docstring), so the
# practical fix is a closed set of words that are never plausible repo names
# and commonly precede "PR" in a title: function words, plus every PR-workflow
# verb's regular inflections, plus a short list of PR-workflow nouns/jargon
# that aren't verbs (github, upstream, qa, follow-up, ...).
#
# Round-7 design decision (MAJOR 1, "Capital PR #6"/"Test PR #88"/"Open PR #99"/
# "Update PR #319" -- gate-2's "legitimate repo name" examples): these stay in the
# ambiguous/denied set ON PURPOSE. A bare, unqualified single dictionary word
# immediately before "PR #N" is indistinguishable from an English sentence using
# that word ("capital" the adjective, "test"/"open"/"update" the verb) -- there is
# no in-card evidence that disambiguates it, and minting a key from it anyway is
# exactly the failure this function exists to prevent (a wrong, fabricated repo is
# no safer than a dropped one -- it just fails quietly instead of loudly). The
# actual fix for a card that DOES name one of these as a real repo is to say so
# unambiguously: `_OWNER_REPO_PR_REF_RE` below always wins over this list, so
# "Stockstrading/capital PR #6" (or any `owner/repo PR #N`) produces its key
# regardless of what the repo segment spells -- the exact case gate-2 found
# broken ("Even Stockstrading/capital PR #6 yields nothing"). An unqualified
# "capital PR #6" with no owner prefix and no corroborating PR URL elsewhere on
# the card is never minted into a key; `_preserve_pr_refs` still re-attaches it
# VERBATIM ("PR #6", no invented repo) so the reference itself is never silently
# dropped, which is the safety property the round-cap actually depends on.
_PR_REF_NON_REPO_WORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those",
    "on", "in", "at", "of", "to", "for", "with", "by", "from", "via", "re",
    "and", "or", "about", "regarding", "before", "after", "into", "onto",
    "as", "during", "past", "vs",
    "review", "reviewing", "reviewed", "reviews",
    "address", "addressing", "addressed", "addresses",
    "investigate", "investigating", "investigated", "investigates",
    "fix", "fixing", "fixed", "fixes",
    "check", "checking", "checked", "checks", "see",
    "keep", "keeping", "kept", "keeps",
    "split", "splitting", "splits",
    "re-run", "re-running", "reran", "re-ran", "reruns", "re-runs",
    "cherry-pick", "cherry-picked", "cherry-picking", "cherry-picks",
    "follow-up", "follow-ups", "re-review", "re-reviews",
    "post-merge", "re-land",
    "github", "upstream", "capital", "qa", "apv", "addendum", "stale", "orphaned",
    "parent", "final",
}) | frozenset().union(*(_verb_forms(v) for v in _PR_REF_NON_REPO_VERBS))


def _norm_repo(name: str) -> str:
    """Lowercase, strip ``.git``/the ``etoro-`` prefix, so the same repo matches
    whichever spelling a title or URL happens to use."""
    n = (name or "").lower().removesuffix(".git")
    return n.removeprefix("etoro-")


def _title_pr_ref_matches(text: str) -> list[tuple[str, str]]:
    """Plausible ``(repo, n)`` pairs for one field: ``owner/repo PR #N`` matches are
    ALWAYS confirmed (an explicit owner prefix is evidence, not a guess -- round-7
    MAJOR 1); a bare ``word PR #N`` match is confirmed only when it is not one of
    ``_PR_REF_NON_REPO_WORDS`` AND it is not inside the span an owner/repo match
    already consumed (so the post-slash repo token in ``Stockstrading/capital PR #6``
    is never also re-evaluated — and potentially denied — as a second, bare match).
    The single seam both ``_title_pr_keys`` and ``_preserve_pr_refs`` go through so
    neither can drift out of sync on what counts as a plausible repo token."""
    t = text or ""
    owner_spans: list[tuple[int, int]] = []
    matches: list[tuple[str, str]] = []
    for m in _OWNER_REPO_PR_REF_RE.finditer(t):
        matches.append((m.group(2), m.group(3)))
        owner_spans.append((m.start(), m.end()))
    for m in _TITLE_PR_REF_RE.finditer(t):
        if any(start <= m.start() < end for start, end in owner_spans):
            continue
        r, n = m.group(1), m.group(2)
        if r.lower() not in _PR_REF_NON_REPO_WORDS:
            matches.append((r, n))
    return matches


def _title_pr_keys(text: Optional[str]) -> set[str]:
    """``repo#n`` keys a single field (title OR body) names on its own, with no
    cross-field fallback -- the building block for both halves of
    ``_card_pr_keys``'s contract."""
    t = text or ""
    keys = {f"{_norm_repo(r)}#{n}" for r, n in _GH_PR_URL_RE.findall(t)}
    keys |= {f"{_norm_repo(r)}#{n}" for r, n in _title_pr_ref_matches(t)}
    return keys


def _ambiguous_pr_numbers(text: Optional[str]) -> set[str]:
    """PR numbers from a bare ``word PR #N`` match this module deliberately refuses to
    key (an ambiguous, denied word with no owner qualification -- see the round-7
    design note above ``_PR_REF_NON_REPO_WORDS``), so ``_preserve_pr_refs`` can still
    re-attach the number VERBATIM ("PR #N", no invented repo) instead of silently
    dropping it."""
    t = text or ""
    owner_spans = [(m.start(), m.end()) for m in _OWNER_REPO_PR_REF_RE.finditer(t)]
    out: set[str] = set()
    for m in _TITLE_PR_REF_RE.finditer(t):
        if any(start <= m.start() < end for start, end in owner_spans):
            continue
        r, n = m.group(1), m.group(2)
        if r.lower() in _PR_REF_NON_REPO_WORDS:
            out.add(n)
    return out


def _card_pr_keys(title: Optional[str], body: Optional[str]) -> set[str]:
    """``repo#n`` keys for the PR a card names: the title decides; the body is
    only consulted when the title names none. Deliberately self-contained
    (no import from ``kanban_db_dispatch``'s review-cost-gate helpers, which
    this card does not depend on) — a minimal, local re-derivation is enough
    for the one thing ``_preserve_pr_refs`` needs: never silently drop the
    repo+PR a card is about."""
    keys = _title_pr_keys(title)
    return keys if keys else _title_pr_keys(body)


def _preserve_pr_refs(old_title: Optional[str], old_body: Optional[str],
                      new_title: Optional[str], new_body: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Never let a rewrite drop the repo + PR a card is about (or its capital tags).

    The dispatcher's per-PR review round cap keys on ``repo#n`` from the title (or a
    PR URL in the body). An LLM rewrite to "Review PR #329: ..." silently removed the
    card from the cap on 2026-10-18, so the original reference is re-attached here.

    The title-rewrite step and the body-restore step deliberately use DIFFERENT key
    sets: the title step only ever looks at keys the OLD TITLE itself named (never a
    body-only key — promoting a body-only reference into a rewritten title would
    contradict ``_card_pr_keys``'s own "the title decides" contract, round-2 finding
    2); the body step unions EVERY PR reference the old body named (not just the
    title-decided subset ``_card_pr_keys`` returns), so a second body URL the title
    never mentioned survives a body rewrite too (round-2 finding 1)."""
    title = new_title if new_title is not None else old_title
    body = new_body if new_body is not None else old_body
    old_title_keys = _title_pr_keys(old_title)
    old_all_keys = old_title_keys | _title_pr_keys(old_body)
    old_text = f"{old_title or ''}\n{old_body or ''}"
    display: dict[str, str] = {}
    urls: dict[str, str] = {}
    for m in _GH_PR_URL_RE.finditer(old_text):
        k = f"{_norm_repo(m.group(1))}#{m.group(2)}"
        urls.setdefault(k, m.group(0))
        display.setdefault(k, f"{m.group(1)} PR #{m.group(2)}")
    for r, n in _title_pr_ref_matches(old_text):
        display.setdefault(f"{_norm_repo(r)}#{n}", f"{r} PR #{n}")
    if title is not None and new_title is not None:
        missing = sorted(k for k in old_title_keys if k not in _title_pr_keys(title))
        prefix = ", ".join(display.get(k, k.replace("#", " PR #")) for k in missing)
        # Round-7: an ambiguous bare-word reference (denied, no key minted) must still
        # never be silently dropped -- re-attach the bare number verbatim, no invented
        # repo, distinct from (and appended after) any keyed references above.
        ambiguous_missing = sorted(
            n for n in _ambiguous_pr_numbers(old_title)
            if f"PR #{n}" not in title and n not in {k.split("#", 1)[1] for k in missing}
        )
        ambiguous_prefix = ", ".join(f"PR #{n}" for n in ambiguous_missing)
        full_prefix = ", ".join(p for p in (prefix, ambiguous_prefix) if p)
        tags = [t for t in dict.fromkeys(_KEEP_TAG_RE.findall(old_title or ""))
                if t.lower() not in title.lower()]
        if full_prefix:
            title = f"{full_prefix}: {title}"
        if tags:
            title = "".join(tags) + " " + title
    if new_body is not None and old_all_keys:
        body_text = body or ""
        lost = [k for k in sorted(old_all_keys)
                if (urls.get(k) and urls[k] not in body_text)
                or (not urls.get(k) and display.get(k, "") not in body_text)]
        if lost:
            refs = ", ".join(urls.get(k) or display.get(k, k) for k in lost)
            body = f"{body_text.rstrip()}\n\n**PR** (kept from the original card): {refs}"
    return (title if new_title is not None else None), (body if new_body is not None else None)


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

    try:
        new_title, new_body = _preserve_pr_refs(task.title, task.body, new_title, new_body)
    except Exception:  # pragma: no cover - never let this guard fail open
        logger.warning(
            "specify %s: PR-ref preservation failed, keeping original title/body",
            task_id, exc_info=True,
        )
        # None, None means "don't touch" at the specify_triage_task call site below;
        # writing the pre-LLM-call snapshot explicitly could clobber a concurrent
        # edit to the same row made during the LLM call window.
        new_title, new_body = None, None

    audit_author = author or _profile_author()
    reason_on_fail = "task moved out of triage before promotion"
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
            # the (possibly just-tightened) title/body. Audit-only: it must never
            # change the primary outcome (the card is already promoted above), so the
            # whole phase is one try/except -- a bad config value or a failed
            # add_comment must not raise past an already-successful promotion and
            # abort the rest of an ``--all`` sweep (round-2 finding 3).
            try:
                from hermes_cli.config import load_config_readonly
                cfg = load_config_readonly()
            except Exception:
                logger.debug("specify %s: config load failed for routing guard", task_id, exc_info=True)
                cfg = {}
            try:
                kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
                raw_default_assignee = kanban_cfg.get("default_assignee") if isinstance(kanban_cfg, dict) else None
                default_assignee = (
                    raw_default_assignee.strip() if isinstance(raw_default_assignee, str) and raw_default_assignee.strip()
                    else "the default profile"
                )
                mismatch = _routing_guard_mismatch(
                    task.assignee,
                    new_title if new_title is not None else task.title,
                    new_body if new_body is not None else task.body,
                    review_only_profiles=review_only_profiles_from_config(cfg),
                    default_assignee=default_assignee,
                )
                if mismatch:
                    kb.add_comment(conn, task_id, audit_author, mismatch)
            except Exception:
                logger.warning(
                    "specify %s: routing guard phase failed (card already promoted, "
                    "continuing)", task_id, exc_info=True,
                )
        elif kb.task_has_unresolved_block_loop(conn, task_id):
            # MINOR 2 (round-5): the CAS refusal is a real state change, not a
            # stale read -- say what actually happened instead of blaming a
            # promotion/archive that never occurred.
            reason_on_fail = "escalation landed during the call"
    if not ok:
        # Race: promoted/archived between our read and the write (or an
        # escalation that landed mid-call -- see reason_on_fail above).
        return SpecifyOutcome(task_id, False, reason_on_fail)
    return SpecifyOutcome(task_id, True, "specified", new_title=new_title)


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Task ids in the triage column; ``tenant`` narrows the sweep.

    Excludes a card carrying an unresolved ``block_loop_detected`` escalation
    (one with no resolving event -- see ``kb.list_tasks(exclude_block_loop=True)``
    -- AFTER it; NOT merely "the most recent event is block_loop_detected", so a
    later unrelated comment can't un-hide it) -- that escalation is for a human
    (or ``kanban_unblock``), not for auto-specify. A card unblocked and later
    re-escalated for an unrelated, newer reason is excluded again.
    """
    with kbc.connect_closing() as conn:
        tasks = kb.list_tasks(
            conn, status="triage", tenant=tenant, include_archived=False, exclude_block_loop=True,
        )
    return [t.id for t in tasks]
