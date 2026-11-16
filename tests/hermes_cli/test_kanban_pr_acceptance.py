"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = [[]]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                value = [{"total_count": 100 + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)]}, {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "print(urllib.request.urlopen(u).read().decode())\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.platforms("linux")
def test_pr_completion_requires_current_required_evidence(github):
    with connect() as conn:
        for conclusion in ("failure", "pending", "cancelled", "timed_out", "action_required", "neutral", "skipped", None, "success"):
            github.update(conclusion=conclusion, head="a" * 40)
            tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
            ok = kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert ok is (conclusion == "success")
            task = kb.get_task(conn, tid)
            assert (task.status == "done") is ok
            receipts = [json.loads(r[0]) for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
            assert receipts and receipts[-1]["head_sha"] == "a" * 40
            if not ok:
                assert task.status in {"running", "ready", "blocked", "review"}
                assert "retry" in receipts[-1]["recovery"]
                assert receipts[-1]["checks"][0]["id"] == 42
        for fault in ("missing", "stale", "head_change"):
            github.update(conclusion="success", head="a" * 40)
            github[fault] = True
            tid = kb.create_task(conn, title=fault, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).status != "done"
            github.pop(fault)
        # Omission and a sibling repository cannot downgrade the stored declaration.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, summary="local green")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/other/repo/pull/7"})
        before = len(github["requests"])
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="https://github.com/acme/repo/pull/7 is background context")
        assert len(github["requests"]) == before


@pytest.mark.platforms("linux")
def test_acceptance_receipts_and_terminal_write_share_run_ownership(github):
    with connect() as conn:
        for conclusion in ("success", "failure"):
            tid = kb.create_task(conn, title="race", completion_contract="acme/repo")
            owner = kb.claim_task(conn, tid)
            run_id = owner.current_run_id
            def reclaim():
                with connect() as rival:
                    assert kb.block_task(rival, tid, reason="Reassigned during acceptance")
                    assert kb.unblock_task(rival, tid)
                    github["replacement"] = kb.claim_task(rival, tid).current_run_id
            github.update(conclusion=conclusion, race=reclaim)
            assert not kb.complete_task(conn, tid, result="done", expected_run_id=run_id,
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


# --- #122689: acceptance must read the repo as the ASSIGNEE profile's gh login ---

@pytest.mark.platforms("posix")
def test_acceptance_runs_gh_as_the_assignee_profile(tmp_path, monkeypatch):
    """The gh child env carries the assignee's own GH credentials (its .env),
    never the ambient/launch residue, and an invisible repo is classified
    `auth` naming the repository — not a retryable `infra` failure."""
    from pathlib import Path

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GH_TOKEN=b-token\n", encoding="utf-8")
    # Ambient residue that must NOT decide the login.
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport json, os\n"
                  f"json.dump(dict(os.environ), open({str(env_dump)!r}, 'w'))\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        receipts = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
        assert receipts[-1]["classification"] == "auth"
        assert "acme/repo" in receipts[-1]["detail"]
    captured = json.loads(env_dump.read_text())
    assert captured["GH_TOKEN"] == "b-token"
    assert captured.get("GH_CONFIG_DIR") != "/nonexistent/launch/gh"
    assert "credentials" in (kb.get_task(conn, tid).last_failure_error or "")


@pytest.mark.platforms("posix")
def test_assignee_without_own_gh_login_never_falls_through_to_ambient_login(tmp_path, monkeypatch):
    """An assignee profile with no GH_TOKEN/GH_CONFIG_DIR of its own must not inherit the
    launch user's ~/.config/gh (HOME/XDG_CONFIG_HOME stay the launch process's): gh is pinned
    to a profile-owned config dir, its 'not logged in' exit is classified `auth` naming the profile."""
    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "launch-xdg"))

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Real gh: GH_CONFIG_DIR wins; a config dir without hosts.yml means "not logged in" (exit 4).
    gh.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
                  f"pathlib.Path({str(env_dump)!r}).write_text(json.dumps(dict(os.environ)), encoding='utf-8')\n"
                  "if 'GH_CONFIG_DIR' in os.environ and not os.path.exists(os.environ['GH_CONFIG_DIR']):\n"
                  "    sys.exit(4)\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'b'" in receipt["detail"] and "no login" in receipt["detail"]
    captured = json.loads(env_dump.read_text(encoding="utf-8-sig"))
    assert captured["GH_CONFIG_DIR"] == str(assignee_home / "gh")
    assert "GH_TOKEN" not in captured and "GITHUB_TOKEN" not in captured


def test_assigned_card_with_unresolvable_profile_is_auth_not_ambient(tmp_path, monkeypatch):
    """A card assigned to a profile that no longer exists must not run gh as the completing
    process's ambient login: classification `auth` naming the profile, gh never invoked."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # any gh spawn would fail as infra
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-ghost", completion_contract="acme/repo", assignee="ghost")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'ghost'" in receipt["detail"] and "cannot be resolved" in receipt["detail"]


# --- branch-rules 403/404 fallback: GitHub Free private repos expose no branch rules ---

@pytest.fixture
def github_no_branch_rules(tmp_path, monkeypatch):
    """Same shape as ``github`` but the rules/branches endpoint answers 404 (no branch
    protection configured, the GitHub Free private-repo plan), forcing the fallback path."""
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                # No branchProtectionRule at all -- the declared-required-checks path is empty,
                # so collect_acceptance must fall through to the branch-rules endpoint below.
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": None}}}}}
            elif "/rules/branches/" in self.path:
                self.send_error(404)
                return
            elif "/check-runs" in self.path:
                runs = [] if state.get("missing") else [
                    {"id": 42, "name": "ci", "head_sha": sha, "app": {"id": 1},
                     "status": "completed", "conclusion": state["conclusion"],
                     "html_url": "https://github.com/acme/repo/actions/runs/42"},
                ]
                if state.get("extra_optional"):
                    runs.append({"id": 43, "name": "optional-advisory", "head_sha": sha,
                                 "app": {"id": 2}, "status": "completed", "conclusion": "neutral"})
                value = [{"total_count": len(runs), "check_runs": runs}]
            elif "/statuses" in self.path:
                value = [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request,urllib.error\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "try:\n    print(urllib.request.urlopen(u).read().decode())\n"
                  "except urllib.error.HTTPError as e:\n    sys.stderr.write(f'HTTP {e.code}\\n'); sys.exit(1)\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.platforms("linux")
def test_branch_rules_404_falls_back_to_every_check_on_head(github_no_branch_rules):
    """A 404 on rules/branches is the GitHub Free private-repo plan, not an identity failure:
    completion still succeeds off the checks that actually ran, with required_source recorded."""
    with connect() as conn:
        tid = kb.create_task(conn, title="fallback-ok", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, result="done",
                                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,)).fetchone()[0])
    assert receipt["classification"] == "success"
    assert receipt["ok"] is True
    assert "fallback" in receipt["required_source"]
    assert {"ci"} <= {c["context"] for c in receipt["required"]}


@pytest.mark.platforms("linux")
def test_branch_rules_404_fallback_still_fails_on_a_real_failure(github_no_branch_rules):
    """The fallback treats ran checks as required -- it does not launder a genuine failure."""
    github_no_branch_rules["conclusion"] = "failure"
    with connect() as conn:
        tid = kb.create_task(conn, title="fallback-fail", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"


@pytest.mark.platforms("linux")
def test_branch_rules_404_fallback_accepts_neutral_and_skipped_unpinned_checks(github_no_branch_rules):
    """An extra check that ran on the head but concluded neutral/skipped (no app pin to
    compare, so _classify's conservative default is `infra`) must not block a fallback repo
    on its own non-blocking checks."""
    github_no_branch_rules["extra_optional"] = True
    with connect() as conn:
        tid = kb.create_task(conn, title="fallback-neutral", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, result="done",
                                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,)).fetchone()[0])
    assert receipt["classification"] == "success"
    neutral_check = next(c for c in receipt["checks"] if c["name"] == "optional-advisory")
    assert neutral_check["classification"] == "success"


@pytest.mark.platforms("linux")
def test_branch_rules_404_fallback_with_no_checks_at_all_stays_missing(github_no_branch_rules):
    """No branch rules AND no checks ran on the head: nothing to fall back to, so the
    outcome is the pre-existing 'declare local-only explicitly' guidance, not a false success."""
    github_no_branch_rules["missing"] = True
    with connect() as conn:
        tid = kb.create_task(conn, title="fallback-empty", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,)).fetchone()[0])
    assert receipt["classification"] == "missing"
    assert receipt["required_source"] == "fallback: all checks that ran on the head (no branch rules available on this plan)"
    assert "local-only" in receipt["detail"]


# --- branch-rules denial must never launder a check GraphQL did not already know about ---

@pytest.fixture
def github_required_rules_denied(tmp_path, monkeypatch):
    """GraphQL branch protection already declares >=1 required check (``required``,
    success), AND a second real check (``other-ci``) ran on the head and failed, AND the
    rules/branches REST endpoint is denied. The denial must never make ``other-ci``
    disappear: a repo with real CI governance cannot have a failing check silently
    laundered just because the rulesets endpoint is inaccessible."""
    state = {"conclusion": "failure", "head": "a" * 40, "status_code": 403, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                self.send_error(state["status_code"])
                return
            elif "/check-runs" in self.path:
                runs = [
                    {"id": 1, "name": "required", "head_sha": sha, "app": {"id": 1},
                     "status": "completed", "conclusion": "success",
                     "html_url": "https://github.com/acme/repo/actions/runs/1"},
                    {"id": 2, "name": "other-ci", "head_sha": sha, "app": {"id": 2},
                     "status": "completed", "conclusion": state["conclusion"],
                     "html_url": "https://github.com/acme/repo/actions/runs/2"},
                ]
                value = [{"total_count": len(runs), "check_runs": runs}]
            elif "/statuses" in self.path:
                value = [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request,urllib.error\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "try:\n    print(urllib.request.urlopen(u).read().decode())\n"
                  "except urllib.error.HTTPError as e:\n    sys.stderr.write(f'HTTP {e.code}\\n'); sys.exit(1)\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.platforms("linux")
def test_graphql_required_plus_denied_rules_never_launders_a_failed_check(github_required_rules_denied):
    """Finding 1 (#9 release-gate reject): GraphQL already declares 'required' required and
    passing; rules/branches is denied (403); 'other-ci' also ran on the same head and failed.
    Pre-fix this returned ok=True (the denial silently dropped whatever rules/branches would
    have required, and 'fallback' never engaged because the merged required set was non-empty).
    The fix must catch 'other-ci' by merging the fallback ran-checks set additively."""
    with connect() as conn:
        tid = kb.create_task(conn, title="denied-rules-real-failure", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,)).fetchone()[0])
    assert receipt["ok"] is False
    assert receipt["classification"] == "failure"
    assert {"required", "other-ci"} <= {c["context"] for c in receipt["required"]}
    failed = next(c for c in receipt["checks"] if c["name"] == "other-ci")
    assert failed["classification"] == "failure"


@pytest.mark.platforms("linux")
def test_graphql_required_plus_denied_rules_still_succeeds_when_everything_passed(github_required_rules_denied):
    """Same denial shape, but both checks that ran actually passed: completion must still
    succeed -- the fix must not turn every rules/branches denial into a false failure."""
    github_required_rules_denied["conclusion"] = "success"
    with connect() as conn:
        tid = kb.create_task(conn, title="denied-rules-all-pass", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, result="done",
                                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,)).fetchone()[0])
    assert receipt["ok"] is True
    assert receipt["classification"] == "success"


@pytest.mark.platforms("linux")
def test_graphql_required_plus_rules_401_fails_closed_not_laundered(github_required_rules_denied):
    """Finding 2/4: a 401 (or rate-limited 403) on rules/branches is an identity/infra problem
    on the SAME login that just read the PR fine -- it must fail closed (classification
    'auth'), never be treated as 'this plan has no rulesets configured'."""
    github_required_rules_denied["status_code"] = 401
    github_required_rules_denied["conclusion"] = "success"
    with connect() as conn:
        tid = kb.create_task(conn, title="denied-rules-401", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"


# --- fetch_pr_head_sha: fail-closed on every ambiguous case ---

def test_fetch_pr_head_sha_returns_live_head_on_success(github):
    from hermes_cli.kanban_pr_acceptance import fetch_pr_head_sha
    sha = fetch_pr_head_sha("https://github.com/acme/repo/pull/7")
    assert sha == "a" * 40


def test_fetch_pr_head_sha_fails_closed_on_malformed_contract(github):
    from hermes_cli.kanban_pr_acceptance import fetch_pr_head_sha
    assert fetch_pr_head_sha("not-a-pr-or-repo") is None
    assert fetch_pr_head_sha("acme/repo") is None  # bare OWNER/REPO has no PR number to resolve


def test_fetch_pr_head_sha_fails_closed_on_mismatched_published_pr(github):
    from hermes_cli.kanban_pr_acceptance import fetch_pr_head_sha
    assert fetch_pr_head_sha("https://github.com/acme/repo/pull/7",
                             published_pr="https://github.com/acme/repo/pull/9") is None


def test_fetch_pr_head_sha_fails_closed_on_private_repo_or_closed_pr(tmp_path, monkeypatch):
    """Null repository (no read access) and a repository with no matching PR both return
    None -- the review-dedupe guard must fall back to spawning a real reviewer, never skip one."""
    from hermes_cli.kanban_pr_acceptance import fetch_pr_head_sha
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport json\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    assert fetch_pr_head_sha("https://github.com/acme/repo/pull/7") is None


def test_fetch_pr_head_sha_fails_closed_on_gh_error(tmp_path, monkeypatch):
    """A gh subprocess failure (network, auth) must never raise into the dispatcher tick."""
    from hermes_cli.kanban_pr_acceptance import fetch_pr_head_sha
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(1)\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    assert fetch_pr_head_sha("https://github.com/acme/repo/pull/7") is None
