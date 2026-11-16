"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.

``gh`` runs as the card's assignee profile (``profile_home``), not the ambient
login: :func:`_gh_env` resolves that profile's own credentials/config for the
subprocess — a multi-profile host's default ``gh`` login cannot read another
org's private repos (#122689).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
# Outcomes that are not failures for a check with no app pin to compare against -- only
# used in the branch-rules-fallback path (see collect_acceptance), never for a repo-declared
# required check (those must still resolve to an actual "success").
_PASSING_WHEN_UNPINNED = {"neutral", "skipped"}


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False,
         profile_home: str | None = None):
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30,
                                check=True, env=_gh_env(profile_home))
    except subprocess.CalledProcessError as exc:
        # 401/403/404 = the login cannot see this repository (wrong profile identity
        # or missing grant), not a transient API failure. Persist only the status
        # code + endpoint, never gh's stderr (credentials/host details).
        denied = re.search(r"HTTP (40[134])", exc.stderr or "")
        if denied:
            raise _GateAuthError(f"HTTP {denied[1]} on {endpoint.split('?')[0]}") from None
        if exc.returncode == 4:  # gh's authentication-required exit: this profile has no login
            raise _GateAuthError(f"gh has no login for {endpoint.split('?')[0]}") from None
        raise
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


class _GateAuthError(RuntimeError):
    """gh was refused at HTTP 401/403/404 (or GraphQL returned no repository):
    this profile's login cannot see the repo — an identity problem to fix, not
    an infrastructure blip to retry."""


def _resolve_contract_pr(contract: str, published_pr: str | None) -> tuple[re.Match[str], str | None] | None:
    """The PR ``match`` and resolved ``url`` for ``(contract, published_pr)``, or ``None``.

    Shared by :func:`fetch_pr_head_sha` and :func:`collect_acceptance`: a ``contract`` that is
    itself a declared PR URL must agree with a non-empty ``published_pr``; otherwise the task's
    own declared contract decides the URL and ``published_pr`` is informational only.
    """
    declared = _PR.fullmatch(contract)
    url = contract if declared else published_pr
    match = _PR.fullmatch(url or "")
    if not match or (not declared and match[1] != contract) or (declared and published_pr and published_pr != contract):
        return None
    return match, url



def _gh_env(profile_home: str | None) -> dict[str, str] | None:
    """Child env for ``gh``: the card's profile identity when one is resolvable.

    The completion boundary runs in the worker (assignee), the CLI, or a
    reviewer/dispatcher turn, so an ambient ``gh`` login is whichever process
    happened to call it (#122689). ``served_profile_child_env(inherit_credentials=True)``
    is the seam for "this child acts for that profile": it scrubs the launch
    profile's credential residue and overlays the target profile's own
    ``GH_TOKEN``/``GH_CONFIG_DIR`` (its ``.env`` + external secret sources).
    ``None`` keeps the ambient env — unassigned cards behave exactly as before.
    """
    if not profile_home:
        return None
    from tools.environments.local import _is_routed_home, hermes_subprocess_env, served_profile_child_env
    base = hermes_subprocess_env(inherit_credentials=True)
    routed = _is_routed_home(profile_home)
    if routed:
        # gh's config dir decides which login `gh api` uses, yet it is a path, not a
        # credential, so no scrub list sees it; the target's own value is overlaid from its .env.
        base.pop("GH_CONFIG_DIR", None)
    env = served_profile_child_env(base=base, target_home=profile_home, inherit_credentials=True)
    if routed and not (env.keys() & {"GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR"}):
        # HOME/XDG_CONFIG_HOME are still the launch process's: without a login of its own the
        # child would fall through to ~/.config/gh/hosts.yml — the ambient login. Pin gh's config
        # to a profile-owned dir so it fails "not logged in" (exit 4 -> auth) instead.
        env["GH_CONFIG_DIR"] = str(Path(profile_home) / "gh")
    return env


def _assignee_profile_home(assignee: str | None) -> str | None:
    """Home whose ``gh`` login must read the contract repo — the assignee's, resolved
    exactly as the dispatcher resolves the worker's home — or None (unassigned) so the
    ambient login is used. An assigned card whose profile cannot be resolved is an
    identity failure (``auth``), never a silent fall-through to the ambient login."""
    if not assignee:
        return None
    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
    try:
        return resolve_profile_env(normalize_profile_name(assignee))
    except (FileNotFoundError, ValueError):
        raise _GateAuthError(f"assignee profile {assignee!r} cannot be resolved") from None


def fetch_pr_head_sha(contract: str, published_pr: str | None = None,
                      assignee: str | None = None) -> str | None:
    """Live head SHA for the PR named by ``contract``/``published_pr``, or ``None``.

    Minimal sibling of :func:`collect_acceptance`: a single GraphQL round trip for just
    ``headRefOid``, skipping the required-checks machinery entirely. Used by the review-dedupe
    guard (``kanban_db_dispatch``) to decide whether a prior reviewer verdict is still current
    -- that guard must fail CLOSED (spawn a real reviewer) whenever this cannot prove the head
    SHA, so every failure mode here (bad contract, private repo, closed PR, network/auth error)
    returns ``None`` rather than raising.
    """
    try:
        profile_home = _assignee_profile_home(assignee)
        resolved = _resolve_contract_pr(contract, published_pr)
        if resolved is None:
            return None
        match, _url = resolved
        repo, number = match[1], int(match[2])
        owner, name = repo.split("/")
        query = (
            f'{{repository(owner:{json.dumps(owner)},name:{json.dumps(name)}){{'
            f'pullRequest(number:{number:d}){{headRefOid state}}}}}}'
        )
        repository = _api("graphql", query=query, profile_home=profile_home)["data"]["repository"]
        if repository is None:
            return None
        pr = repository.get("pullRequest")
        if pr is None:
            return None
        sha = pr.get("headRefOid")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            return None
        if pr.get("state") not in {"OPEN", "MERGED"}:
            return None
        return sha
    except (_GateAuthError, OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError,
            IndexError, AttributeError):
        # Never raise into a dispatcher tick: any resolution failure here must fail
        # closed into "spawn for real", never into a silent skip. AttributeError covers a
        # malformed GraphQL shape (e.g. {"data": {"repository": []}}) where .get()/dict
        # access on a non-dict value would otherwise escape -- the docstring promises this
        # never raises.
        return None


def _resolve_required_checks(required: set[tuple[str, int | None]], graphql_required: bool,
                             rules_denied_exc: Exception | None,
                             ran_checks: set[tuple[str, None]]) -> tuple[set, str | None]:
    """The final required-checks set and its ``required_source`` label (``None`` when the
    set is exactly what GraphQL + the rulesets REST call declared, with nothing to disclose).

    Raises ``rules_denied_exc`` unchanged when the rules/branches denial was not the
    GitHub-Free-plan 403/404 shape: a 401 or a rate-limited 403 on the SAME login that just
    read the PR fine is an identity/infra problem, never "no governance configured", and
    must fail closed rather than be read as a plan limitation.
    """
    if rules_denied_exc is not None:
        if not re.search(r"HTTP 40[34]", str(rules_denied_exc)):
            raise rules_denied_exc
        if graphql_required:
            # GraphQL already declared >=1 required check: the denied rulesets call may have
            # required more. Never drop that possibility silently -- merge every check that
            # actually ran on the head in ADDITIVELY on top of the known GraphQL set, so a
            # real failure on an unlisted check is still caught.
            return required | ran_checks, ("graphql-required checks + fallback: all checks that ran on "
                                           "the head (rules/branches denied on this plan)")
        return set(ran_checks), "fallback: all checks that ran on the head (no branch rules available on this plan)"
    if not required:
        return set(ran_checks), "fallback: all checks that ran on the head (no branch rules available on this plan)"
    return required, None


def collect_acceptance(contract: str, published_pr: str | None,
                       assignee: str | None = None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": published_pr, "checks": [],
               "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                           "Use kanban_block if human input is needed; receipts remain on the task event log."}
    try:
        profile_home = _assignee_profile_home(assignee)
        resolved = _resolve_contract_pr(contract, published_pr)
        if resolved is None:
            receipt["detail"] = "Supply metadata.published_pr matching the persisted completion contract."
            return receipt
        match, url = resolved
        repo, number = match[1], int(match[2])
        receipt["pr_url"] = url
        owner, name = repo.split("/")
        query = f'{{repository(owner:{json.dumps(owner)},name:{json.dumps(name)}){{pullRequest(number:{number:d}){{headRefOid baseRefName state\n            baseRef{{branchProtectionRule{{requiredStatusChecks{{context app{{databaseId}}}}}}}}}}}}}}'
        repository = _api("graphql", query=query, profile_home=profile_home)["data"]["repository"]
        if repository is None:
            # A private repo the login cannot read resolves to null, not an error.
            raise _GateAuthError(f"HTTP 404 on graphql {repo}")
        pr = repository["pullRequest"]
        sha, branch = pr["headRefOid"], pr["baseRefName"]
        receipt["head_sha"] = sha
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        protection = (pr.get("baseRef") or {}).get("branchProtectionRule") or {}
        required = {(r["context"], (r.get("app") or {}).get("databaseId")) for r in protection.get("requiredStatusChecks", [])}
        graphql_required = bool(required)
        rules_denied_exc: _GateAuthError | None = None
        try:
            rules = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100",
                         paginate=True, profile_home=profile_home)
        except _GateAuthError as exc:
            # Record the denial instead of silently discarding it: whether it is safe to
            # treat as "this plan has no rulesets" (vs. an identity failure to fail closed
            # on) is decided below, once we know the HTTP status and what GraphQL already
            # declared required.
            rules_denied_exc = exc
            rules = []
        for page in rules:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    required.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest",
                     paginate=True, profile_home=profile_home)
        runs = [run for page in pages for run in page["check_runs"]]
        if len({r["id"] for r in runs}) != pages[0]["total_count"]:
            raise ValueError("Incomplete check-run pagination")
        statuses = [{**s, "sha": sha} for page in _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100",
                                                       paginate=True, profile_home=profile_home) for s in page]
        ran_checks = {(r["name"], None) for r in runs} | {(s["context"], None) for s in statuses}
        required, required_source = _resolve_required_checks(required, graphql_required, rules_denied_exc, ran_checks)
        if required_source:
            receipt["required_source"] = required_source
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        if not required:
            receipt["detail"] = ("No repository-required checks are configured and no checks ran on the head; "
                                 "explicitly use a local-only contract for non-CI tasks.")
            return receipt
        outcomes = []
        for context, app_id in sorted(required, key=str):
            matching = [r for r in runs if r["name"] == context and
                        (app_id in (None, -1) or r["app"]["id"] == app_id)]
            # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
            legacy = [s for s in statuses if s["context"] == context] if app_id in (None, -1) else []
            selected = matching + ([max(legacy, key=lambda s: s["id"])] if legacy else [])
            if not selected:
                outcomes.append("missing")
                receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
            for check in selected:
                is_run = "conclusion" in check
                outcome = check.get("conclusion") if is_run else check["state"]
                classification = _classify(check, sha, outcome, is_run)
                if app_id is None and classification == "infra" and outcome in _PASSING_WHEN_UNPINNED:
                    # A required context with no app pin to compare against has nothing for
                    # _classify's conservative default (infra on an unrecognised-but-non-failing
                    # outcome -- "neutral"/"skipped") to check, so it would block every
                    # branch-rules-denied or no-rules repo on its own non-blocking checks;
                    # treat those as passing. A pinned required check never takes this path.
                    classification = "success"
                outcomes.append(classification)
                receipt["checks"].append({"name": context, "id": check["id"],
                    "url": check.get("html_url") or check.get("target_url"),
                    "head_sha": check.get("head_sha", check.get("sha")),
                    "classification": classification, "conclusion": outcome})
        # Re-read after all pages: old-head successes are never transferable.
        current = _api(f"repos/{repo}/pulls/{number}", profile_home=profile_home)
        if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
            receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
            return receipt
        receipt["classification"] = next((x for x in outcomes if x != "success"), "missing" if not outcomes else "success")
        receipt["ok"] = receipt["classification"] == "success"
        return receipt
    except _GateAuthError as exc:
        login = f"assignee profile {assignee!r}'s gh login" if assignee else "the ambient gh login"
        receipt.update(classification="auth",
                       detail=f"GitHub refused the acceptance read ({exc}) as {login}; "
                              "fix that profile's GitHub credentials/access to the repository, then retry completion.")
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # Never persist gh stderr (credentials/host details); the failed phase is actionable.
        receipt.update(classification="infra", detail="GitHub acceptance evidence unavailable or incomplete; check gh authentication/API access and retry.")
        return receipt


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")
