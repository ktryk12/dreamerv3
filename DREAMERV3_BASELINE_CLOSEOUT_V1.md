# DreamerV3 — baseline closeout V1

```text
REPOSITORY:   E:\repos\dreamerv3  (ktryk12/dreamerv3, fork of danijar/dreamerv3)
BRANCH:       fix/test-baseline-v1 @ 79f0c48  (local only, not pushed)
BASELINE IN:  0e7332a0 on feat/podman-cpu-v1
DATE:         2026-10-23
AUTHORIZED:   a minimal correctness fix to selectors.Uniform.__delitem__;
              a test-only solution for the parallel tests
NOT TOUCHED:  RL algorithms, training semantics, production architecture
ON HOLD:      push, merge, upstream PR, deployment
```

Prior records: `OVERORDENT_REPOSITORY_CANONICAL_AUDIT_V1.md`,
`DREAMERV3_BASELINE_RECOVERY_V1.md`,
`DREAMERV3_SECURITY_RECOMMENDATIONS_V1.md`,
`DREAMERV3_VIRKORA_ADAPTER_DESIGN_V1.md`.

---

## 1. Test result and reproducibility

| | Audit (`0e7332a0`) | Previous tranche | **Now** |
|---|---|---|---|
| Collection | 2 errors, suite aborts | 0 errors, 201 tests | **0 errors, 226 tests** |
| Passed | 141 | 188 | **222** |
| Failed | 55 | 9 | **0** |
| Skipped | 0 | 4 | 4 (unchanged, section 4) |
| Wall time | aborted | 34 s | 78 s |

```text
RUN 1 summary: 222 passed, 4 skipped in 79.04s     RUN 1 failures: 0
RUN 2 summary: 222 passed, 4 skipped in 78.10s     RUN 2 failures: 0
REPRODUCIBLE=yes   (identical result lines)
```

Both runs inside the container, repository mounted read-only, `--network none`.
The suite grew from 34 s to 78 s because the new lifecycle tests take 45 s of
real parallel execution; that is the price of the coverage and it is bounded.

---

## 2. Task A — replay correctness

### 2.1 Reproduced first

Before any change: `9 failed, 188 passed, 4 skipped in 35.17s`, all nine in
`test_replay.py`, all the `capacity=1` parameterizations.

### 2.2 The selector's invariants, documented

`Uniform` keeps two containers so that both sampling and removal are O(1):

```text
keys    : list[key]          sampled by random index
indices : dict[key, int]     position of each key in `keys`

invariant:  indices[keys[i]] == i   for every i
            set(keys) == set(indices)
            len(keys) == len(indices) == len(selector)
```

Removal swaps the last element into the freed slot:

```python
index = self.indices.pop(key)
last = self.keys.pop()
if index != len(self.keys):      # the removed key was not already last
    self.keys[index] = last
    self.indices[last] = index
```

`__call__`, `__setitem__` and `__delitem__` all hold `self.lock`, so the three
are mutually exclusive. `__len__` does not take the lock.

`Uniform` was the only selector carrying a minimum-size guard: `Fifo`,
`Recency`, `Prioritized` and `Mixture` have none.

### 2.3 Delete-before-insert

`Replay._insert` evicts first and inserts after:

```python
while self.capacity and len(self.items) >= self.capacity:
    self._remove()
```

so at eviction time the selector holds exactly `capacity` items. With
`capacity == 1` that is one item, and the guard demanded two. The guard was
protecting nothing: measured directly, the swap-pop path with a single element
leaves `keys == []`, `indices == {}`, `len == 0` — correct, because the freed
index equals the new length so the swap is skipped.

### 2.4 Concurrency risk, measured not assumed

`Replay.add` holds `rwlock.reading` and `Replay.sample` takes no rwlock at all,
so eviction and sampling are **not** mutually exclusive — only the individual
selector calls are, and `Uniform.lock` does not span the delete/insert pair in
`_insert`.

That matters only at `capacity == 1`, because any larger capacity evicts from
`capacity` down to `capacity - 1 >= 1`. At `capacity == 1` the selector is
briefly empty, and:

```text
np.random.default_rng(0).integers(0, 0)  -> ValueError: high <= 0
Uniform().__call__() on empty            -> ValueError: high <= 0
Replay._sample retry handler             -> except KeyError: continue
```

So a concurrent sampler that lands in that window sees a `ValueError` the retry
loop does not catch. The old assertion hid this by making `capacity == 1`
unusable altogether.

**Net effect of the fix:** `capacity == 1` goes from "always fails" to "works,
with a narrow race under concurrent sampling". Strictly better, and the
remaining race is documented in `test_selectors.py::test_call_on_empty_selector`
rather than left implicit. Raising `KeyError` from `__call__` would route it
into the existing retry path — one line — but `__call__` was outside this
tranche's authorization, so it is a recommendation, not a change.

### 2.5 The fix

One production line, plus the comment explaining it:

```diff
-      assert 2 <= len(self), len(self)
+      assert 1 <= len(self), len(self)
```

The guard is lowered rather than removed: deleting from an empty selector would
make `indices.pop` raise `KeyError`, and that is a caller bug worth catching.
`test_selectors.py::test_delete_from_empty_raises` pins that behaviour.

### 2.6 Regression tests

`embodied/tests/test_selectors.py`, new, 17 tests. An `invariants()` helper
asserts `indices[keys[i]] == i` and both set and length equality, and is called
after every mutation in:

| Test | Covers |
|---|---|
| `test_insert_invariants[1,2,5,20]` | invariant after each insert |
| `test_delete_single_item` | the capacity=1 case that used to assert |
| `test_delete_then_reinsert` | five evict/insert cycles, the capacity=1 window |
| `test_delete_from_empty_raises` | the guard that is kept |
| `test_delete_in_insertion_order[1,2,5,20]` | invariant after each delete |
| `test_delete_in_random_order[0,1,2]` | invariant under permuted deletes, 20 keys |
| `test_sampling_only_returns_live_keys` | 500 draws stay inside the live set and cover it |
| `test_sampling_after_capacity_one_cycle` | sampling right after each evict/insert |
| `test_call_on_empty_selector` | the documented limitation from 2.4 |

`embodied/tests/test_replay.py`, three added at replay level:

| Test | Covers |
|---|---|
| `test_evict_and_sample_at_capacity[1,2,10]` | eviction and sampling at capacity 1, 2 and normal; only the most recent `capacity` steps are sampleable |
| `test_capacity_one_holds_only_the_last_step` | 20 add/sample cycles at capacity 1 |
| `test_eviction_is_fifo` | at capacity 2 the older of the two is evicted |

### 2.7 Existing behaviour verified

The nine previously failing tests now pass, and no other test changed state:
the suite went from 188 passed / 9 failed to 222 passed / 0 failed, where the
extra 25 are the new tests (17 selector + 3 replay parameterized to 5 + 3
lifecycle). `test_sample_uniform`, `test_sample_sequences`, `test_restore_*`,
`test_threading` and the save/load round-trips all pass unchanged.

---

## 3. Task B — the parallel tests

### 3.1 The design question

A correct fix to `test_parallel.py` needs production to change, which is not
authorized: `combined()` ends in `portal.run(workers)`, every worker loop is
`while True`, and `args.duration` is read nowhere in
`embodied/run/parallel.py` since `f8817c40` replaced
`distr.run(workers, args.duration, exit_after=True)`.

So the four skips stay, as instructed, and the coverage was rebuilt from the
outside in a new module.

### 3.2 The time-bounded lifecycle design

`embodied/tests/test_parallel_lifecycle.py`. `portal.Process` makes it
possible: it cloudpickles its target, so a closure works; its `kill()`
terminates the process *and all nested children*; and its `running` flag is
reliable.

```text
test process                          worker process (portal.Process)
------------                          -------------------------------
portal.Server('report') ........<.... TestAgent.policy reports stats
start worker                          combined() -> portal.run(workers)
poll stats, bounded by BUDGET=120s      4 env processes
  env_steps >= 400                      actor + learner threads
  saves     >= 2                        replay server
  reports   >= 1                        logger server
assert on the stats
worker.kill()                         -> process and all children die
assert no descendants left behind
```

Every wait has a deadline, so the module cannot hang — the failure mode is a
test failure with the last observed stats, not a 610 s timeout.

### 3.3 What it actually tests

| Test | Covers | Time |
|---|---|---|
| `test_runs_terminates_and_cleans_up` | real parallel execution; `env_steps`, `replay_steps`, `reports`, `saves` all advance; replay steps track env steps at `train_ratio=32`; `loads == 0` on a fresh logdir; termination; no leftover processes | ~20 s |
| `test_restart_loads_the_checkpoint` | a second run against the same logdir restores the first run's checkpoint (`loads == 1`) | ~20 s |
| `test_without_eval_envs` | the `eval_envs=0` path | ~5 s |

`3 passed in 45.13s`, and they pass as part of the full suite in both runs.

These are the assertions of `test_parallel.py` that do not require
`combined()` to return — including the `loads == 1` restart assertion, which
was the second half of the original test.

### 3.4 Two findings from building it, both mine

**`TestAgent.stream` was wrong.** The previous tranche had it return the stream
unchanged, reasoning that a test agent needs neither the device placement nor
the seeding that the JAX agent's `streams.Prefetch(st, fn)` adds. It does need
the Prefetch. The learner gates reporting on
`should_report(skip=not received['report'])`, and `received[source]` is only
incremented inside `parallel_stream()` — which the Prefetch worker thread is
what advances. Without it the report stream is never primed and `reports` stays
at zero forever, which is exactly what the first lifecycle run showed
(`assert array(0) >= 1`). `TestAgent.stream` now returns
`embodied.streams.Prefetch(st)`. `DREAMERV3_BASELINE_RECOVERY_V1.md` section
3.3 carries the correction.

**The cleanup check counted a false leak.** The first green-assertions run still
failed on `Processes left behind after kill(): [(70, ['/usr/local/bin/python',
'-c', 'from multiprocessing.resource_tracker import main;main(17)'])]`. That is
multiprocessing's resource tracker — one helper per parent interpreter, spawned
because `portal` uses mp barriers, alive as long as pytest. It is excluded by
cmdline, with the reason in the code; anything else still fails the test.

### 3.5 Separate design decision for the owner

Restoring a duration-bounded run in production would let `test_parallel.py`
pass as written:

```diff
-  portal.run(workers)
+  portal.run(workers, duration=args.duration)
```

`portal.run(workers, duration=None)` already accepts it, and `args.duration`
already exists in the test's config. Two reasons to consider it: it is how the
code worked before `f8817c40`, and it would let the original four tests assert
on a completed run rather than on a sampled one. One reason against: it changes
how `combined()` terminates in production, where the current behaviour — run
until killed — is what an operator wants.

My recommendation is to leave production as it is and keep the lifecycle
module, because it tests termination and cleanup, which a duration-bounded run
does not. The four skips then remain permanently, and their reason in the
source points at the lifecycle module.

---

## 4. Remaining skips

Four, all `embodied/tests/test_parallel.py::TestParallel::test_run_loop`
(`[-1-2]`, `[1-2]`, `[1-0]`, `[32-2]`), marked `pytest.mark.skip` with the full
reason in the source, visible in the summary line as `4 skipped`.

No `xfail`, no loosened assertion, nothing skipped to improve a number. The
module collects, its imports and call signatures are current, and its coverage
now exists in `test_parallel_lifecycle.py`.

Separately, `embodied/perf/test_bandwidth.py` and `embodied/perf/test_distr.py`
remain un-collectable: they need `zerofun.StoppableProcess`, and `portal` has
no equivalent (`portal.StoppableProcess` and `portal.get_free_port` are both
absent — verified). They are benchmarks, not part of `embodied/tests`, and
migrating them would mean guessing at cooperative-shutdown semantics.

---

## 5. Production changes

One line in one file, for the whole closeout:

```text
 embodied/core/selectors.py | 9 ++++++++-
 1 file changed, 8 insertions(+), 1 deletion(-)
```

of which 7 lines are the explanatory comment and one is the change:
`assert 2 <= len(self)` → `assert 1 <= len(self)`.

Everything else in this tranche is test code. Across the whole branch against
`main`:

```text
production:  embodied/core/selectors.py          (1 line)
tests:       embodied/tests/{test_selectors,test_parallel_lifecycle}.py  (new)
             embodied/tests/{test_replay,test_driver,test_parallel,test_train,utils}.py
             embodied/perf/test_replay.py
infra/docs:  .gitattributes, .containerignore, Containerfile.podman,
             scripts/podman-cpu.ps1, PODMAN.md, five records
```

No change to `dreamerv3/`, `embodied/jax/`, `embodied/run/` or
`embodied/envs/`.

---

## 6. Security status

| Item | State |
|---|---|
| Secrets in tracked files | none (scan returned only `tokens` in the ML sense) |
| CUDA packages in the CPU image | zero |
| Offline capability | the full suite and the smoke run both pass under `--network none` |
| Line endings | closed, verified from a fresh clone (section 7.3) |
| `pip check` | `No broken requirements found` |
| Container runs as root | **open** — recommendation 4 |
| No resource limits | **open** — recommendation 5 |
| `crafter` unpinned | **open** — recommendation 3, and now measured (section 7.2) |
| Base image by tag | **open** — recommendation 2, and now measured |
| Upstream Dockerfile supply chain | recorded, not our build path |
| `cloudpickle` over `portal` RPC | loopback only by default; a constraint for any future integration |

Nothing in `DREAMERV3_SECURITY_RECOMMENDATIONS_V1.md` sections 2-9 was applied;
that document now carries the drift measurement as section 1b.

---

## 7. Container status

### 7.1 Cached build

`podman build -f Containerfile.podman -t dreamerv3-cpu .` — exit 0, seconds,
mostly layer reuse. Smoke under `--network none`: exit 0, 23 agent-step
reports.

### 7.2 Clean build, and what it proves

`podman build --no-cache --pull=always` — exit 0 in **381 s**, image
`591369769c31`. It works: smoke exit 0, no warnings. But it is **not the same
image**. `pip freeze` against the cached build, nine days apart:

```text
cached: 100 packages        clean: 101 packages

pillow            12.3.0       -> 13.0.0        <- major
python-dateutil   2.9.0.post0  -> 3.0.0         <- major
fastapi           0.143.1      -> 0.161.1
starlette         1.8.0        -> 1.9.0
certifi, filelock, google-api-core, google-cloud-storage,
googleapis-common-protos, granular, pycparser, wcwidth, websockets  -> bumped
tzdata            (absent)     -> 2026.5        <- new
```

Fourteen version changes and one new package in nine days, two of them major.
So "rebuild the image" and "rebuild the same image" are different operations
today, and a training result cannot be attributed to a specific software
state. The build happening to still work is the measurement, not a guarantee —
this is the concrete case for pinning the base image by digest and pinning
`crafter`.

### 7.3 Checkpoint, metrics and line endings

Offline smoke run (`--network none`, named volume):

```text
SMOKE_EXIT=0
metrics lines: 31        scores lines: 9        last step: 1932 of 2000
ckpt entries:  20261023T051644F267461  latest      ckpt size: 176K
```

Line endings from a **fresh clone** of the branch, in a clone that inherited
`core.autocrlf=true`:

```text
entrypoint.sh:  CR bytes = 0, LF bytes = 21     (file: "ASCII text")
README.md:      CR bytes = 123                  (not covered, as intended)
git check-attr: entrypoint.sh -> text: set, eol: lf
```

The `.gitattributes` rule overrides `autocrlf` on a clean checkout, which is
what the GPU image's `COPY . .` depends on. Only `*.sh` is covered, by design —
CRLF on `.py` files is harmless, and the suite passing in-container with CRLF
sources is the proof.

---

## 8. Merge readiness and the exact order

### 8.1 Verified state

```text
main                e01491f   == upstream/main, 0 ahead / 0 behind
feat/podman-cpu-v1  0e7332a   == origin/feat/podman-cpu-v1, fully pushed
fix/test-baseline-v1 79f0c48  4 ahead of feat, 0 behind, local only

main is an ancestor of feat   : yes
feat is an ancestor of work   : yes
merge commits in main..work   : 0
PR #1  feat/podman-cpu-v1 -> main : OPEN, MERGEABLE, mergeStateStatus CLEAN
```

The history is strictly linear:

```text
e01491f (main, upstream/main)
  └─ 3db6521  feat(podman): add CPU-first DreamerV3 workflow
      └─ 0e7332a  docs(podman): correct machine capacity guidance   (feat tip)
          └─ dae9ab9  fix(tests): restore the test baseline
              └─ d24d34a  docs: audit, recovery, security, adapter design
                  └─ 942aadc  fix(replay): capacity=1 eviction
                      └─ 79f0c48  test(parallel): lifecycle test     (work tip)
```

Because every branch tip is an ancestor of the next, **both merges are
fast-forwards**. No merge commits, no cherry-picks, no rebases, no duplicate
commits, and no history rewriting is needed anywhere.

### 8.2 The order

```bash
# 1. fix/test-baseline-v1 -> feat/podman-cpu-v1   (fast-forward, local)
git switch feat/podman-cpu-v1
git merge --ff-only fix/test-baseline-v1          # feat: 0e7332a -> 79f0c48

# 2. publish the feature branch                   [HOLD: needs owner GO]
git push origin feat/podman-cpu-v1                # PR #1 picks up 4 commits

# 3. re-verify on the published branch
podman build -f Containerfile.podman -t dreamerv3-cpu .
podman run --rm --network none -v "$PWD:/src:ro" dreamerv3-cpu-test \
  sh -c "cd /src && python -m pytest embodied/tests -q"

# 4. feat/podman-cpu-v1 -> main                   [HOLD: needs owner GO]
#    Either merge PR #1 on GitHub with "Rebase and merge" / "Squash",
#    or locally:
git switch main
git merge --ff-only feat/podman-cpu-v1            # main: e01491f -> 79f0c48
git push origin main
```

Two notes on step 4. If PR #1 is merged on GitHub with the default **"Create a
merge commit"**, `main` gets a merge commit and `main` and `feat` stop being
identical, which costs nothing here but ends the clean linear history; **"Rebase
and merge"** keeps it linear but rewrites the commit ids, so the local `feat`
would then differ from `origin/main` and should be reset to it afterwards. The
fast-forward in the snippet avoids both. Whichever is chosen, do it once —
merging the PR *and* fast-forwarding locally would be the one way to produce
duplicate commits here.

### 8.3 Divergence from upstream after the merge

```text
18 files changed, 2273 insertions(+), 33 deletions(-)
```

of which 1 line is production, 33 deletions are all in test files, and 1467
insertions are the five markdown records. The fork stays additive, so merges
from `danijar/dreamerv3` remain trivial.

---

## 9. Verdict

```text
TEST BASELINE:            PASS        222 passed, 4 skipped, 0 failed, reproducible
REPRODUCIBILITY (tests):  PASS        two identical runs
REPRODUCIBILITY (image):  FAIL        14 package versions drifted in 9 days
PRODUCTION CHANGES:       1 line      selectors.py, with regression tests
REMAINING SKIPS:          4           documented, coverage relocated
SECURITY:                 PARTIAL     line endings closed; pinning, non-root,
                                      resource limits open
CONTAINER:                PASS        cached and clean builds both green; smoke
                                      green offline; checkpoint and metrics written
MERGE PATH:               CLEAN       two fast-forwards, no rewriting
GIT HISTORY:              PRESERVED

INTEGRATION TO MAIN:      GO
```

**GO**, with one caveat that is not a blocker: the image is not reproducible.
The test baseline is green and reproducible, the single production change is
one line backed by 20 new tests, the container builds from scratch and trains,
and the merge is two fast-forwards with nothing to rewrite.

The image drift does not block integration because it affects what a *future*
rebuild produces, not what this branch contains. It should be closed before any
training result is used as evidence for anything — at that point "which
software produced this number" becomes a question, and today there is no
answer. That is recommendation 2 and 3 in the security document, roughly two
lines in `Containerfile.podman`.

### 9.1 What still needs the owner

1. **Push and merge** — steps 2 and 4 in section 8.2. Still on HOLD.
2. **Pin the image** — base image by digest
   (`sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce`)
   and `crafter==1.8.3`. Two lines, closes section 7.2.
3. **`Uniform.__call__` on an empty selector** — raise `KeyError` so
   `Replay._sample`'s existing retry handles it. One line, closes the last
   `capacity=1` race from section 2.4.
4. **The parallel run loop** — accept the four permanent skips plus the
   lifecycle module (my recommendation), or restore `duration` in production.
5. **Upstream PR** — the test fixes, `.gitattributes` and the selectors fix all
   apply to `danijar/dreamerv3`. Still on HOLD.

Items 2 and 3 are each about one line and would both be verified by the
existing suite plus a rebuild.
