"""Task claims, not caller-supplied launch policy, authorize Kanban execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_claim_freezes_task_policy_and_rejects_forgery(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'pyproject.toml').write_text('[project]\nname = "task"\n')  # a code worktree
    from contextlib import closing
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned acceptance', body='Goal criteria', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['owned-skill'], goal_mode=True,
            goal_max_turns=3, model_override='loop-model', provider_override='custom')
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    from gateway.session_kanban import build_kanban_policy
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal', 'file']},
              'agent': {'coding_context': 'focus'}}  # the worktree must not collapse a worker into the coding posture
    policy, _secrets = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    assert policy.source == 'kanban' and policy.cwd == str(workspace)
    assert policy.model == task.model_override and policy.provider == task.provider_override
    assert {'kanban', 'terminal', 'file'} <= set(policy.toolsets)
    assert context['task_id'] == task.id and context['run_id'] == task.current_run_id
    assert context['board'] == 'owned' and context['claim_lock'] == task.claim_lock
    assert context['skills'] == ['owned-skill'] and context['goal_mode'] and context['goal_max_turns'] == 3
    assert context['accept_hooks'] is True and 'Goal criteria' in context['goal_text']
    for changed in ({'run_id': task.current_run_id + 1}, {'board': 'foreign'}, {'claim_lock': 'forged'}, {'cwd': '/tmp'}):
        with pytest.raises(RuntimeStoreError):
            build_kanban_policy(connection, params | changed, config)
    connection.native_owner = False
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        build_kanban_policy(connection, params, config)
    from gateway.session_policy import build_policy
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        build_policy({'source': 'kanban', 'cwd': str(workspace)}, config)


@pytest.mark.parametrize('mode', ['complete', 'timeout', 'rate_limit', 'billing', 'crash'])
def test_real_dispatcher_lifecycle(tmp_path, mode):
    import os
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    from tests.gateway.fixtures.local_recovery_probe import child_env
    env = child_env()
    env.update(HOME=str(tmp_path / 'home'), USERPROFILE=str(tmp_path / 'home'), HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=str(repo), KANBAN_PROBE_MODE=mode)
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_owner_probe.py')],
        env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt))
    if mode != 'complete':
        assert receipt['task_status'] == 'ready'
        return
    assert receipt['task_status'] == 'done' and receipt['source'] == 'kanban'
    assert receipt['tools'] and receipt['task_context'] and receipt['skill_context'] and receipt['hook_effect']
    assert receipt['admissions'] == 1 and receipt['retry_same_session']
    assert receipt['goal_continuation'] and receipt['stable_prefix']
    print(json.dumps(receipt))


def test_review_lane_forces_sdlc_review_tracked_separately(tmp_path, monkeypatch):
    """The review lane adds ``sdlc-review`` to the preload list (unchanged behaviour) but also
    records it as ``forced_review_skill`` so ``_run_task_turns`` can treat it as soft (#t_3d8b302d:
    a reviewer worker whose copy of the skill could not resolve exited 1 before any turn ran)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from contextlib import closing
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Reviewed deliverable', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['own-skill'])
        kb.recompute_ready(conn)
        implementer_run = kb.claim_task(conn, task_id)
        kb.request_review(conn, task_id, summary='done', expected_run_id=implementer_run.current_run_id)
        task = kb.claim_review_task(conn, task_id)
        assert task is not None
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    from gateway.session_kanban import build_kanban_policy, REVIEW_FORCED_SKILL
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal', 'file']}}
    policy, _secrets = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    assert context['skills'] == ['own-skill', REVIEW_FORCED_SKILL]
    assert context['forced_review_skill'] == REVIEW_FORCED_SKILL


def test_review_lane_overlap_keeps_own_requested_forced_skill_as_hard_requirement(tmp_path, monkeypatch):
    """A task that itself requests ``sdlc-review`` (not just the review lane's auto-add) must keep
    it in ``own_skills`` so it stays a hard requirement — #t_3d8b302d round 1's value-equality
    filter on the merged list silently reclassified this overlap case as the fail-open forced
    copy, which would have hidden a genuinely broken preload for a task that asked for it."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from gateway.session_kanban import REVIEW_FORCED_SKILL
    from contextlib import closing
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Reviewed deliverable', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=[REVIEW_FORCED_SKILL])
        kb.recompute_ready(conn)
        implementer_run = kb.claim_task(conn, task_id)
        kb.request_review(conn, task_id, summary='done', expected_run_id=implementer_run.current_run_id)
        task = kb.claim_review_task(conn, task_id)
        assert task is not None
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    from gateway.session_kanban import build_kanban_policy
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal', 'file']}}
    policy, _secrets = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    # Deduped union still shows one entry; own_skills (pre-union) still carries it as the task's
    # own hard requirement, independent of forced_review_skill also naming it.
    assert context['skills'] == [REVIEW_FORCED_SKILL]
    assert context['own_skills'] == [REVIEW_FORCED_SKILL]
    assert context['forced_review_skill'] == REVIEW_FORCED_SKILL


def test_forced_review_skill_failure_is_fail_open_but_own_skill_failure_still_raises(monkeypatch):
    """A reviewer whose task carries no skills of its own never hard-fails on a missing
    ``sdlc-review`` (fail-open); a task with its OWN unresolvable skill still raises so a genuinely
    broken preload is not silently swallowed."""
    import agent.skill_commands as sc
    from gateway import session_kanban

    class _Agent:
        session_id = 'sess-1'

        def run_conversation(self, prompt, conversation_history=None):
            return {'completed': True, 'final_response': prompt}

    monkeypatch.setattr('agent.kanban_turn_recovery.recover_failed_kanban_turns', lambda *a, **kw: None)
    frame = {'text': 'work kanban task t1'}

    # Case 1: only the force-loaded review skill is missing -> no raise, turn still runs.
    def fake_build_only_forced_missing(identifiers, task_id=None, excluded_loaded_names=None):
        if identifiers == ['sdlc-review']:
            return '', [], ['sdlc-review']
        return '', [], []
    monkeypatch.setattr(sc, 'build_preloaded_skills_prompt', fake_build_only_forced_missing)
    context = {'skills': ['sdlc-review'], 'own_skills': [], 'forced_review_skill': 'sdlc-review',
               'context': '', 'task_id': 't1', 'goal_mode': False}
    result = session_kanban._run_task_turns(_Agent(), frame, [], context)
    assert result['completed'] is True

    # Case 2: the task's OWN requested skill is unresolvable -> still a hard failure.
    def fake_build_own_missing(identifiers, task_id=None, excluded_loaded_names=None):
        if identifiers == ['own-skill']:
            return '', [], ['own-skill']
        return '', [], []
    monkeypatch.setattr(sc, 'build_preloaded_skills_prompt', fake_build_own_missing)
    context2 = {'skills': ['own-skill'], 'forced_review_skill': None, 'context': '',
                'task_id': 't1', 'goal_mode': False}
    with pytest.raises(ValueError, match='kanban_skills_unavailable'):
        session_kanban._run_task_turns(_Agent(), frame, [], context2)

    # Case 3: overlap — the task's own requested skills ALSO include the forced review skill
    # name. own_skills (pre-union, as build_kanban_policy now stores it) must still be treated
    # as a hard requirement even though forced_review_skill names the same skill.
    def fake_build_overlap_missing(identifiers, task_id=None, excluded_loaded_names=None):
        if identifiers == ['sdlc-review']:
            return '', [], ['sdlc-review']
        return '', [], []
    monkeypatch.setattr(sc, 'build_preloaded_skills_prompt', fake_build_overlap_missing)
    context3 = {'skills': ['sdlc-review'], 'own_skills': ['sdlc-review'],
                'forced_review_skill': 'sdlc-review', 'context': '', 'task_id': 't1', 'goal_mode': False}
    with pytest.raises(ValueError, match='kanban_skills_unavailable'):
        session_kanban._run_task_turns(_Agent(), frame, [], context3)


def test_late_launcher_cannot_stamp_replacement_claim(tmp_path, monkeypatch):
    from contextlib import closing
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='Owned race', assignee='default')
        old = kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason='replace claim', expected_run_id=old.current_run_id)
        kb.unblock_task(conn, tid)
        new = kb.claim_task(conn, tid)
        dispatch._set_worker_pid(conn, tid, 424242, run_id=old.current_run_id, claim_lock=old.claim_lock)
        assert kb.get_task(conn, tid).worker_pid is None
        assert conn.execute('SELECT worker_pid FROM task_runs WHERE id=?', (new.current_run_id,)).fetchone()[0] is None


def test_claim_reclaimed_before_admission_marker_is_refused(tmp_path, monkeypatch):
    """A claim that stops being current between policy/session creation and the marker
    transaction is refused: no owner_admitted row, no resume, no submit."""
    import asyncio
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned race', assignee='default', workspace_kind='dir', workspace_path=str(workspace))
        kb.recompute_ready(conn)
        old = kb.claim_task(conn, task_id)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    calls = []
    def create_local_session(authority, actor, params, *, trusted_policy=None, trusted_secrets=None):
        # Another process reclaims the task while the owner session is being created.
        with closing(connect(board='owned')) as conn:
            kb.block_task(conn, task_id, reason='replace claim', expected_run_id=old.current_run_id)
            kb.unblock_task(conn, task_id)
            kb.claim_task(conn, task_id)
        return SimpleNamespace(session_id='owner-session', profile_id=str(tmp_path))
    monkeypatch.setattr('gateway.session_local.create_local_session', create_local_session)
    async def submit(actor, submission, **kwargs):
        calls.append(('submit', submission))
    async def resume(ref, params):
        calls.append(('resume', ref))
    db = SimpleNamespace(get_session=lambda sid: None, db_path=str(tmp_path / 'state.db'))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, submit=submit)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    params = dict(board='owned', task_id=task_id, run_id=old.current_run_id, claim_lock=old.claim_lock)
    from gateway.session_kanban import run_task
    with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    with closing(connect(board='owned')) as conn:
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='owner_admitted'", (task_id,)).fetchone()[0] == 0
