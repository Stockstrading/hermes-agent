# Local Hermes patches → fork branch: switch-over doc (card t_02e1f65f, finding F11)

## What this is

`~/.hermes/hermes-agent` (the live install, v0.22.0, HEAD detached) has 15 modified
files that were hand-patched in place over time (several via the scripts in
`~/work/_patches/`), plus 2 stale autostashes from past `hermes update` runs.
`updates.non_interactive_local_changes: "stash"` means the NEXT `hermes update`
auto-stashes these changes to apply the new version — silently dropping them
unless someone remembers to `git stash pop` afterward.

Branch `chore/local-patches-fork-t_02e1f65f` on `Stockstrading/hermes-agent`
(fork remote) reproduces the live diff **exactly** (verified byte-for-byte,
see "Verification" below), built on top of `fork/main` via a 3-way merge
against the files fork/main has since evolved independently.

PR: https://github.com/Stockstrading/hermes-agent/pull/2
(branch pushed; open the PR link above or run `gh pr create` if it wasn't
auto-opened by the time you read this)

## Commits (6, not 5 — see "Scope note" below)

1. `feat(kanban): config defaults ... + scheduled resume_at timer + block_recurrence_limit override`
   — `config_defaults.py`, `kanban_db.py`, `kanban_db_connect.py` + their tests.
2. `feat(kanban): respawn-guard PR-dedup fix + review cost/round-cap gate + review dedupe + Rules 3/5 ...`
   — `kanban_db_dispatch.py` (one commit: this file interleaves the respawn-guard
   fix, the review cost gate, the per-PR round cap, review-dedupe, and ADR Rules 3/5 too
   tightly to split without rewriting code by hand — see "Scope note").
3. `feat(kanban): PR acceptance fallback ...` — `kanban_pr_acceptance.py`.
4. `feat(kanban): card PR-key preservation on specify + Rule 6 routing guard + block_loop exclusion ...`
   — `kanban_specify.py`, `kanban_decompose.py` + their tests.
5. `feat(kanban): kanban_schedule SCHEDULED_UNTIL=<iso> wake timer + kanban_request_changes metadata`
   — `tools/kanban_tools.py`, `tools/kanban_tools_schemas.py` + test.
6. `docs(sdlc-review): structured findings schema + head_sha metadata guidance ...`
   — `skills/devops/sdlc-review/SKILL.md`.

## Scope note — this is bigger than "5 patches"

The card named 5 patches (respawn-guard fix, review cost gate, review round
cap, PR acceptance fallback, card PR-key resolver). Reading the live diff
end to end, the actual local-patch surface also contains, inseparably mixed
into the same file as the cost gate:

- **Review dedupe** (replay a prior verdict when the PR head SHA and
  findings are unchanged — skips a real model spawn).
- **ADR "review-loop convergence" Rules 3, 5, 6** (card `t_f526b652`):
  structured findings + delta-only re-review (Rule 3), fallback model for
  low-risk PRs when the required gate is confirmed unavailable (Rule 5),
  and a log-only routing guard for dev-vs-review-only assignee mismatches
  (Rule 6).
- **Cost gate D**: capital-affecting PRs force Opus for release-reviewer.
- **Block-recurrence-limit config override** and the **scheduled `resume_at`
  429-backoff timer** — infrastructure the above features share, not named
  individually in the card.

"Review round cap" turned out not to be its own patch: it is step 3 inside
`_review_cost_gate`. There is **also already an open PR #1**
(`feat/review-round-cap-quota-stalls-t_480b842d-v2`, "generalize review
round cap — 3 real rounds OR 3 quota stalls → owner escalation") on the
fork for a **different, quota-stall-based** round-cap mechanism — do not
confuse the two; #1 was left untouched by this work.

## Test gaps found (not fixed here — scope was "reproduce the diff", not "add missing coverage")

- No pytest file covers `kanban_pr_acceptance.py`'s branch-rules fallback or
  `fetch_pr_head_sha`'s fail-closed paths directly (commit 3).
- No pytest file covers `kanban_specify._preserve_pr_refs` directly (commit 4);
  it's only exercised transitively.
- The original respawn-guard patch's own standalone verifier
  (`~/work/_patches/verify_guard.py`, 6/6 cases) was never ported into the
  pytest suite as `tests/hermes_cli/test_kanban_respawn_guard_known_pr.py`
  (the patch's own patch file includes that test, but it's absent from the
  live tree — whoever applied `apply_guard_fix.py`/`apply_guard_fix2.py`
  didn't carry the test file forward).
- Recommend a follow-up `backend-dev` card to add these three.

## Verification performed

- `git diff` of every one of the 15 modified files in this branch against
  the corresponding file in the live install: **zero differences** (ran a
  byte-for-byte `diff` over all 15 paths after the last commit).
- `.venv` built with `uv venv --python 3.14` (the project pins all real deps
  to `python_version >= '3.14'`; a 3.12 venv installs almost nothing and
  everything ImportErrors on `ruamel`/ruamel-yaml-backed modules).
- `scripts/run_tests.sh`-equivalent (`pytest` directly, since the wrapper
  needs a clean non-worktree checkout) on the 5 affected test files: **130
  passed, 1 skipped, 4 failed**. All 4 failures are a pre-existing
  environment artifact, NOT caused by this patch set — reproduced
  identically on a clean `fork/main` checkout before any of these commits
  landed: `tests/home_io_guard.py` refuses file I/O under
  `/home/ubuntu/.hermes/hermes-agent/.git/worktrees/<name>/...` because this
  worktree's `.git` lives inside the real `HERMES_HOME`'s install. Running
  the same tests from a worktree outside `~/.hermes/` (or from a plain
  clone) would not hit this.
- `ruff check hermes_cli/ tools/ tests/`: all checks passed.
- The pre-push lint hook (`scripts/check`'s lint-python stage) ran clean on
  every commit.

## Switch-over steps (for the owner to run — this agent made NO live changes)

1. **Rebase the fork branch onto the latest `fork/main` / upstream `main`**
   before merging, in case either moved further since this was built:
   ```
   cd <a fresh clone or worktree of Stockstrading/hermes-agent>
   git fetch fork origin
   git checkout chore/local-patches-fork-t_02e1f65f
   git rebase fork/main     # or origin/main, per your merge target
   ```
2. **Run the full suite** (not just the 5 affected files) before merging:
   ```
   uv venv .venv --python 3.14 --clear
   uv pip install -p .venv -e . --group dev --group test
   HERMES_HOME=$(mktemp -d) .venv/bin/python -m pytest tests/ -q
   ```
3. **Merge the PR** (https://github.com/Stockstrading/hermes-agent/pull/2)
   into whatever branch the fork treats as its integration branch.
4. **Switch the live install over**, on the VM, as root/ubuntu — NOT done by
   this agent (ground rule: no live Hermes install/config changes):
   ```
   cd ~/.hermes/hermes-agent
   git remote -v                     # confirm 'fork' still points at
                                      # Stockstrading/hermes-agent
   git fetch fork
   git checkout fork/main            # or whatever branch now carries the merge
   git stash list                    # inspect/drop the 2 old autostashes
                                      # (hermes-update-autostash-20261007,
                                      # -20261019) — they predate this work
                                      # and are very likely superseded by it;
                                      # diff them before dropping.
   ```
   Then `hermes update` (or restart the gateway/dispatcher) so new worker
   processes pick up the merged code from a clean branch HEAD instead of a
   detached, hand-patched tree.
5. **After switch-over**, `updates.non_interactive_local_changes: "stash"`
   stops being a silent-data-loss risk for THESE patches (they're in git
   history on a real branch now) — any *future* local hotfix still needs
   its own commit on a branch before the next `hermes update`, or it will
   be auto-stashed again.

## Upstream candidates (self-contained enough to send to NousResearch/hermes-agent)

Good candidates — generic bugs/gaps, not Stockstrading-specific policy:

- **Respawn-guard PR-dedup fix** (commit 2, part 1): a real correctness bug
  (livelock) in generic dispatcher logic, with its own regression tests
  already written. Best candidate for upstreaming as-is.
- **PR acceptance branch-rules fallback + `fetch_pr_head_sha`** (commit 3):
  the GitHub-Free-private-repo branch-rules-403 fallback is a genuine gap
  (anyone using a private repo without branch protection hits the same
  `ValueError`); `fetch_pr_head_sha` is a generically useful minimal
  GraphQL helper. Needs the missing test file (see "Test gaps") before
  sending upstream.
- **Review dedupe** (commit 2, part 3): generically useful (skip a spawn
  when nothing changed) and config-gated (`kanban.review_dedupe`) so it's
  opt-out, not a behaviour change forced on other installs.
- **`resume_at` scheduled-task wake timer + `recompute_scheduled`**
  (commit 1): generic scheduler feature, backward compatible (NULL = old
  behaviour unchanged), with tests.

Keep local (Stockstrading-specific policy, not generic Hermes behaviour):

- **Review cost gate's round cap + `_REVIEW_ROUND_CAP_BY_PR`** (commit 2,
  part 2): the override dict is literally per-PR owner decisions for this
  org's repos — meaningless anywhere else, and would need a config-file
  rewrite (not a hardcoded dict) to even be upstream-shaped.
- **Rule 5 fallback model polic**y (commit 2, part 5) and **Rule 6 routing
  guard** (commit 4): both encode this org's specific profile names
  (`reviewer`, `release-reviewer`, `backend-dev`, ...) and ADR decisions
  (card `t_f526b652`) rather than generic Hermes behaviour.
- **Cost gate D's capital-PR Opus override** (commit 2, part 6): hardcodes
  `_CAPITAL_REPOS` / `_CAPITAL_DENY_SUBSTRINGS` specific to this trading
  platform.
