"""Tests for the specifier module + `hermes kanban specify` CLI surface.

The auxiliary LLM client is mocked — these tests don't hit any network or
real provider. They exercise the prompt plumbing, response parsing, DB
writes, and CLI flag surface.
"""

from __future__ import annotations

import argparse
import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    """Build a minimal object shaped like an OpenAI chat.completions result.

    The specifier only reads ``resp.choices[0].message.content``, so we
    avoid importing the openai SDK and build the tree with MagicMock.
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    """Patch call_llm at its source module — specify_task now routes through
    it (#35566) instead of building a raw client. Returns (patcher, mock) so
    callers can still assert on the call.
    """
    mock_fn = MagicMock(return_value=_fake_aux_response(content))
    return patch("agent.auxiliary_client.call_llm", mock_fn), mock_fn


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# specify_task (module-level entry point)
# ---------------------------------------------------------------------------

def test_specify_task_happy_path(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rough", triage=True)

    content = jsonlib.dumps({
        "title": "Refined rough",
        "body": "**Goal**\nA concrete goal.",
    })
    p, _ = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="ace")

    assert outcome.ok is True
    assert outcome.task_id == tid
    assert outcome.new_title == "Refined rough"

    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    # Parent-free → recompute_ready promotes to ready.
    assert task.status == "ready"
    assert task.title == "Refined rough"
    assert "**Goal**" in (task.body or "")






# ---------------------------------------------------------------------------
# CLI wiring — argparse + _cmd_specify
# ---------------------------------------------------------------------------

def _run_cli(*argv: str) -> int:
    """Invoke the `hermes kanban …` argparse surface directly."""
    root = argparse.ArgumentParser()
    subp = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subp)
    ns = root.parse_args(["kanban", *argv])
    return kanban_cli.kanban_command(ns)




def test_cli_specify_tenant_filter(kanban_home, capsys):
    with kbc.connect() as conn:
        outside = kb.create_task(conn, title="outside", triage=True)
        inside = kb.create_task(
            conn, title="inside", triage=True, tenant="proj-a",
        )

    content = jsonlib.dumps({"title": "spec", "body": "body"})
    p, _ = _patch_aux_client(content)
    with p:
        rc = _run_cli("specify", "--all", "--tenant", "proj-a", "--json")
    assert rc == 0
    lines = [
        jsonlib.loads(l)
        for l in capsys.readouterr().out.strip().splitlines()
        if l
    ]
    ids = {row["task_id"] for row in lines}
    assert ids == {inside}

    # The outside task stays in triage.
    with kbc.connect() as conn:
        assert kb.get_task(conn, outside).status == "triage"
        # The inside task was promoted.
        assert kb.get_task(conn, inside).status in {"todo", "ready"}


# ---------------------------------------------------------------------------
# _preserve_pr_refs: an auto-specify rewrite must not drop the repo+PR key
# or capital-safety tags.
# ---------------------------------------------------------------------------


def test_preserve_pr_refs_keeps_dropped_title_pr_key():
    """An LLM rewrite that drops the repo+PR reference gets it re-prefixed."""
    old_title = "strategy-api PR #329: fix zero-bid loop"
    new_title, new_body = spec._preserve_pr_refs(
        old_title, "body", "Fix the zero-bid collect loop", None,
    )
    assert new_title is not None
    assert "strategy-api" in new_title and "329" in new_title
    assert new_body is None


def test_preserve_pr_refs_keeps_capital_tags_in_title():
    old_title = "[CAPITAL][needs-human-ok] strategy-api PR #392: fix gate"
    new_title, _ = spec._preserve_pr_refs(old_title, "body", "Fix the release gate", None)
    assert new_title is not None
    assert "[CAPITAL]" in new_title
    assert "[needs-human-ok]" in new_title


def test_preserve_pr_refs_keeps_pr_url_in_rewritten_body():
    old_body = "See https://github.com/Stockstrading/strategy-api/pull/329 for context."
    _, new_body = spec._preserve_pr_refs("t", old_body, None, "A totally rewritten body.")
    assert new_body is not None
    assert "pull/329" in new_body


def test_preserve_pr_refs_untouched_fields_pass_through_as_none():
    """When the specifier didn't rewrite a field (None), the function must not
    invent a value for it — the caller's "don't touch" contract at the DB
    write site depends on this."""
    new_title, new_body = spec._preserve_pr_refs("old t", "old b", None, None)
    assert new_title is None
    assert new_body is None


def test_preserve_pr_refs_no_op_when_old_card_names_no_pr():
    new_title, new_body = spec._preserve_pr_refs("plain idea", "body", "Tightened idea", "New body")
    assert new_title == "Tightened idea"
    assert new_body == "New body"


# ---------------------------------------------------------------------------
# list_triage_ids must not resurrect block_loop_detected escalations
# ---------------------------------------------------------------------------


def _running_task(conn, title="t"):
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def test_list_triage_ids_excludes_block_loop_detected(kanban_home):
    """A triage card whose latest event is block_loop_detected is a human-gated
    escalation; auto-specify must not re-specify it back to todo (that is
    exactly the storm this fix stops)."""
    with kbc.connect() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        kb.block_task(conn, tid, reason="x", kind="capability")
        assert kb.get_task(conn, tid).status == "triage"

    assert tid not in spec.list_triage_ids()


def test_list_triage_ids_includes_fresh_triage_card(kanban_home):
    """A plain one-liner idea card in triage (no block_loop_detected event at
    all) must still be picked up — only the escalation kind is excluded."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="idea", triage=True)
        assert kb.get_task(conn, tid).status == "triage"

    assert tid in spec.list_triage_ids()


# ---------------------------------------------------------------------------
# PR12 release-gate MAJOR 2 — specify_task must re-check for an unresolved
# block_loop_detected escalation right before the LLM call, not trust a
# minutes-stale list_triage_ids() enumeration from an --all sweep.
# ---------------------------------------------------------------------------


def test_specify_task_refuses_and_skips_llm_call_when_block_loop_lands_after_enumeration(kanban_home):
    """Mirrors the exact repro trace from the release-gate finding: a sweep
    enumerates the card as eligible, a block_loop_detected escalation then
    lands on it, and the (stale) call into specify_task must still refuse --
    and never even reach the auxiliary LLM, since the escalation is caught by
    _load_triage_task before _call_aux runs."""
    with kbc.connect() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        kb.block_task(conn, tid, reason="x", kind="capability")
        assert kb.get_task(conn, tid).status == "triage"

    # The stale enumeration a sweep would have made BEFORE the escalation
    # landed (simulated here by not even calling list_triage_ids() again).
    content = jsonlib.dumps({"title": "Sneaked past the gate", "body": "b"})
    p, mock_fn = _patch_aux_client(content)
    with p:
        outcome = spec.specify_task(tid, author="sweep")

    assert outcome.ok is False
    mock_fn.assert_not_called()
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "triage"
    assert task.title != "Sneaked past the gate"


def test_specify_task_reason_names_escalation_landed_during_the_call(kanban_home):
    """Round-6 MINOR 2: the CAS refusal that fires when an escalation lands
    strictly DURING the (slow) LLM call -- after ``_load_triage_task`` already
    passed -- must report ``"escalation landed during the call"``, not the
    generic "task moved out of triage before promotion" used for every other
    CAS miss. A mutation that replaces that string with a no-op silently
    passes every other test in this file, so it needs its own pin."""
    with kbc.connect() as conn:
        tid = _running_task(conn)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))

    def _escalate_during_the_call(*args, **kwargs):
        # Simulates another worker's block_task hitting the recurrence limit
        # and landing block_loop_detected on this card while the LLM call this
        # process is making is still in flight.
        with kbc.connect() as conn2:
            with kb.write_txn(conn2):
                kb._append_event(conn2, tid, "block_loop_detected", None)
        return _fake_aux_response(jsonlib.dumps({"title": "Sneaked past the gate", "body": "b"}))

    mock_fn = MagicMock(side_effect=_escalate_during_the_call)
    with patch("agent.auxiliary_client.call_llm", mock_fn):
        outcome = spec.specify_task(tid, author="sweep")

    mock_fn.assert_called_once()
    assert outcome.ok is False
    assert outcome.reason == "escalation landed during the call"
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "triage"
    assert task.title != "Sneaked past the gate"


# ---------------------------------------------------------------------------
# PR12 release-gate MAJOR 1 — _TITLE_PR_REF_RE requires an explicit "PR"
# marker; a bare hyphenated token + "#N" must never be treated as a PR ref.
# ---------------------------------------------------------------------------


def test_title_pr_ref_does_not_match_hyphenated_token_without_pr_marker():
    assert spec._card_pr_keys("Investigate phase-2 #3 rollout", None) == set()


def test_title_pr_ref_does_not_match_commit_style_hyphenated_token():
    assert spec._card_pr_keys("fix(kanban): block-loop #2 follow-up", None) == set()


def test_title_pr_ref_matches_explicit_pr_marker_on_non_hyphenated_repo():
    assert spec._card_pr_keys("numpy PR #123 bump", None) == {"numpy#123"}


# ---------------------------------------------------------------------------
# PR12 release-gate ROUND 5 MAJOR 1 — mandatory whitespace stops the regex
# backtracking into a word (APR/SPR), and a small non-repo-word filter stops
# an ordinary English word immediately before "PR" from being treated as a
# repo. All five measured false positives from the round-5 gate finding.
# ---------------------------------------------------------------------------


def test_title_pr_ref_does_not_split_a_word_ending_in_pr():
    assert spec._card_pr_keys("APR #12 planning", None) == set()


def test_title_pr_ref_does_not_split_a_capitalized_word_ending_in_pr():
    assert spec._card_pr_keys("The SPR #7 rollout", None) == set()


def test_title_pr_ref_does_not_treat_review_as_a_repo():
    assert spec._card_pr_keys("Review PR #123 before merge", None) == set()


def test_title_pr_ref_does_not_treat_preposition_on_as_a_repo():
    assert spec._card_pr_keys("Address CodeRabbit comments on PR #45", None) == set()


def test_title_pr_ref_does_not_treat_preposition_in_as_a_repo():
    assert spec._card_pr_keys("Investigate the regression in PR #5", None) == set()


# ---------------------------------------------------------------------------
# PR12 release-gate ROUND 6 MAJOR 1 — the deny-list measured against the 8209
# real card titles in ~/.hermes/kanban.db still fabricated a key from 80 of
# them (verify#560, rebase#103, re-review#210, resolve#324, update#319, ...).
# Every word from that measurement and the round-6 finding's own text must
# now produce no key; the real-repo cases must keep working.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("title", [
    "Merge PR #123", "Rebase PR #12", "Revert PR #8", "Backport PR #45",
    "Follow-up PR #12", "Re-review PR #12", "Verify PR #2", "Test PR #88",
    "GitHub PR #42", "Upstream PR #7", "Open PR #99", "Land PR #100",
    # the 80-fabrication measurement over real titles
    "Verify PR #560 test evidence and static gates", "rebase PR #103",
    "re-review PR #210", "resolve PR #324", "update PR #319",
    "Inspect PR #6", "Reconcile PR #6", "Push PR #6", "Summarize PR #6",
    "Approve PR #6", "Retarget PR #6", "Fold PR #6", "Prepare PR #6",
    "Refresh PR #6", "Assess PR #6", "Audit PR #6", "Complete PR #6",
    "Correct PR #6", "Harden PR #6", "Rework PR #6", "Stacked PR #6",
    "Blocking PR #6", "Gate PR #6", "Keep PR #6", "QA PR #6", "vs PR #6",
    "capital PR #6", "post-merge PR #6", "re-land PR #6",
])
def test_round6_pr_ref_fabrication_words_produce_no_key(title):
    assert spec._card_pr_keys(title, None) == set()


@pytest.mark.parametrize("title", [
    "3D PR #12", "123repo PR #8", "_internal PR #4",
])
def test_round6_pr_ref_non_letter_prefix_never_starts_a_match(title):
    """`(?<![A-Za-z])` is not a token boundary -- a digit or underscore right
    before the token still let the match start mid-word and carve a fake
    repo out of the tail (the round-5 defect verbatim)."""
    assert spec._card_pr_keys(title, None) == set()


@pytest.mark.parametrize("title,expected", [
    ("strategy-api PR #528", {"strategy-api#528"}),
    ("hermes-agent PR #12", {"hermes-agent#12"}),
    ("numpy PR #123", {"numpy#123"}),
    ("AI-StockTrading PR #617", {"ai-stocktrading#617"}),
])
def test_round6_real_repo_pr_refs_still_produce_their_key(title, expected):
    assert spec._card_pr_keys(title, None) == expected


# ---------------------------------------------------------------------------
# PR12 release-gate MINOR 1/2 — the title-prefix step and the body-restore
# step must use different key sets (title-only vs. all body URLs).
# ---------------------------------------------------------------------------


def test_preserve_pr_refs_keeps_second_body_url_on_body_rewrite():
    """A second PR URL that the title never named must survive a full body
    rewrite too, not just the one the title decided on."""
    old_title = "strategy-api PR #536: fix"
    old_body = (
        "See https://github.com/Stockstrading/strategy-api/pull/536 and "
        "https://github.com/Stockstrading/position-service/pull/7 for context."
    )
    _, new_body = spec._preserve_pr_refs(old_title, old_body, None, "Rewritten body.")
    assert new_body is not None
    assert "pull/536" in new_body
    assert "pull/7" in new_body


def test_preserve_pr_refs_title_rewrite_does_not_promote_body_only_url():
    """A PR URL that only ever lived in the body must never get promoted into
    a rewritten title -- that would contradict _card_pr_keys' own 'the title
    decides' contract (round-2 finding 2)."""
    old_title = "Investigate recurring failure"
    old_body = "Caused by https://github.com/Stockstrading/strategy-api/pull/536."
    new_title, _ = spec._preserve_pr_refs(old_title, old_body, "Tightened investigation", None)
    assert new_title == "Tightened investigation"


