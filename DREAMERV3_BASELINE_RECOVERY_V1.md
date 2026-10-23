# DreamerV3 — baseline recovery V1

```text
REPOSITORY:   E:\repos\dreamerv3  (ktryk12/dreamerv3, fork of danijar/dreamerv3)
BASELINE IN:  0e7332a0  on feat/podman-cpu-v1
WORK BRANCH:  fix/test-baseline-v1  (local only, not pushed)
DATE:         2026-10-23
SCOPE:        test and compatibility fixes plus .gitattributes. No production
              code, no algorithm, no replay semantics, no container change,
              no merge, no push, no upstream PR, no Virkora integration.
```

Companion documents: `DREAMERV3_SECURITY_RECOMMENDATIONS_V1.md` (sections 2-9
are recommendations, not applied) and
`DREAMERV3_VIRKORA_ADAPTER_DESIGN_V1.md` (design proposal, nothing
implemented).

---

## 1. Result in one table

| | Before (`0e7332a0`) | After (`fix/test-baseline-v1`) |
|---|---|---|
| Collection | **2 errors**, suite aborts; 196 collected | **0 errors**, 201 collected |
| Passed | 141 | **188** |
| Failed | 55 | **9** |
| Skipped | 0 | 4 (documented, section 5) |
| Wall time | 14.8 s *with the two broken modules excluded* | 34.1-34.8 s for the whole suite |
| Production code changed | — | **0 files** |

Both numbers are reproducible: two consecutive full runs gave
`9 failed, 188 passed, 4 skipped` in 34.80 s and 34.09 s.

The 9 remaining failures are one pre-existing **production** defect, not a test
defect, and fixing it is outside this tranche's authorization (section 4).

---

## 2. Phase 1 — state verified before any change

| Check | Measured |
|---|---|
| HEAD | `0e7332a0`, branch `feat/podman-cpu-v1`, clean tracked tree |
| Worktree | one, `E:\repos\dreamerv3`, no submodules |
| `main` vs `upstream/main` | 0 ahead, 0 behind (`e01491fa` both) after `git fetch --all` |
| PR #1 | `state=OPEN mergeable=MERGEABLE feat/podman-cpu-v1->main` |
| Our two commits | `A .containerignore`, `A Containerfile.podman`, `A PODMAN.md`, `A scripts/podman-cpu.ps1` — **0 `.py` files** |
| Work branch | `fix/test-baseline-v1` created from `0e7332a0` |

### 2.1 Audit findings reproduced

The canonical audit left attribution as NOT_VERIFIED. It is now **verified by
execution**. `git archive upstream/main` was extracted to a scratch directory
and the suite was run against that tree inside the container, with the import
path proven:

```text
embodied imported from: /up/embodied

collect-only:  196 tests collected, 2 errors
               ERROR embodied/tests/test_parallel.py  ModuleNotFoundError: zerofun
               ERROR embodied/tests/test_train.py     same, via utils.py:4
full run:      55 failed, 141 passed  (24.44 s)
```

Identical to the numbers measured on our branch. The defects are upstream's;
our four added files cannot have caused them.

### 2.2 Dependency map: `zerofun`, `portal`, replay, driver

Everything traces to **one upstream commit, `f8817c40` ("Update")**, which
moved production forward and left the tests behind:

| Surface | Before `f8817c40` | After | Tests still assumed |
|---|---|---|---|
| RPC package | `zerofun` | `portal` (declared `portal>=3.5.0`) | `zerofun` — never declared anywhere |
| Replay sampling | `replay.dataset(batch)` → iterator | `replay.sample(batch, mode)` → one batch | `dataset()` |
| Agent data input | `agent.dataset(generator)` | `agent.stream(st) -> st` | `dataset()` |
| Run-loop signature | `train(agent, replay, env, logger, args)` | `train(agent, replay, env, **stream**, logger, args)` | the old arity |
| Transition contents | `trans` included `'reset'` | `'reset'` moved to `Driver.acts` only | `seq['reset']` |
| Parallel termination | `distr.run(workers, args.duration, exit_after=True)` | `portal.run(workers)`, every loop `while True` | that `combined()` returns |

The last row is the one that could not be fixed from the test side; see
section 5.

Address format changed with the package: `portal` asserts `'://' not in addr`
(`portal/client_socket.py:39`, `portal/server_socket.py:28`), so `ipc://` and
`tcp://` prefixes are gone. Production already uses `localhost:{auto}`
(`configs.yaml:61-63`).

---

## 3. Phase 2 — what was changed

Six files, all tests. `git diff --name-only 0e7332a0 -- dreamerv3/ embodied/core/
embodied/jax/ embodied/run/ embodied/envs/` returns **0 files**.

```text
 embodied/perf/test_replay.py    | 12 +++++++++---
 embodied/tests/test_driver.py   | 11 +++++++++--
 embodied/tests/test_parallel.py | 31 ++++++++++++++++++++++++-------
 embodied/tests/test_replay.py   | 33 ++++++++++++++++++++-------------
 embodied/tests/test_train.py    |  7 +++++--
 embodied/tests/utils.py         | 33 ++++++++++++++++++++++++++++-----
 6 files changed, 95 insertions(+), 32 deletions(-)
```

Plus one new file, `.gitattributes`.

### 3.1 `zerofun` → `portal`

- `embodied/tests/utils.py`: `import zerofun` → `import portal`;
  `zerofun.Client(addr, connect=True)` → `portal.Client(addr)`. The eager
  connect is now the default (`ClientSocket(..., start=True)`), so the kwarg is
  gone rather than renamed.
- `embodied/tests/test_parallel.py`: same import change; the stats server
  address `'ipc:///tmp/teststats'` → `f'localhost:{portal.free_port()}'`;
  `server.start()` → `server.start(block=False)` (portal's `start` blocks by
  default); the three `'tcp://localhost:{auto}'` args → `'localhost:{auto}'`,
  matching `configs.yaml`.

`embodied/perf/test_bandwidth.py` and `embodied/perf/test_distr.py` were **not**
migrated: they need `zerofun.StoppableProcess`, and `portal` has no
`StoppableProcess` and no `get_free_port`. The authorization required verified
API compatibility before migrating, and there is none to verify — a replacement
would be a guess at cooperative-shutdown semantics, in two benchmark modules
whose timing assertions cannot be validated here. They remain un-collectable,
unchanged, and out of scope.

### 3.2 Replay: `dataset()` → `sample()`

One helper added to `embodied/tests/test_replay.py`, and the same to
`embodied/perf/test_replay.py`:

```python
def samples(replay, batch=1, mode='train'):
  # Replay.dataset() was replaced by Replay.sample() in f8817c40, which returns
  # one batch per call instead of an iterator.
  while True:
    yield replay.sample(batch, mode)
```

All 13 call sites in `tests/` and all 3 in `perf/` were `replay.dataset(1)`
verbatim, so the substitution is mechanical. **No assertion was changed**: the
existing `unbatched()` wrapper and every `assert` are untouched. This took
`test_replay.py` from 53 failures to 9.

### 3.3 Agent: `dataset()` → `stream()`

`base.Agent.dataset(generator)` became `stream(st) -> st`. `TestAgent` now
implements `stream`.

> **Corrected on 2026-10-23.** This section originally said `TestAgent.stream`
> passes the stream through because a test agent needs neither the device
> placement nor the seeding the JAX agent adds. That was wrong in one respect:
> the JAX agent returns `streams.Prefetch(st, fn)`, and the parallel learner
> gates reporting on `should_report(skip=not received['report'])` while
> `received[source]` is only incremented inside `parallel_stream()`, which the
> Prefetch worker thread is what advances. Without the Prefetch the report
> stream is never primed and `reports` stays at zero forever.
> `TestAgent.stream` now returns `embodied.streams.Prefetch(st)`. See
> `DREAMERV3_BASELINE_CLOSEOUT_V1.md`.

A shared `make_stream(args, replay, mode)` was added to
`embodied/tests/utils.py`, mirroring `dreamerv3/main.py:make_stream` (Stateless
→ Consec, with `length` switching on mode and `consec=1`, the production
default from `configs.yaml:13-14`). `test_train.py` and `test_parallel.py` pass
it as the new sixth argument, and `test_train.py` gained `consec_report=1`,
which `train()` reads.

`TestAgent.train` asserted an exact key set. It was updated, not weakened —
still exact equality, with the contract change encoded:

```python
expected = set(self.obs_space) | set(self.act_space)
expected |= {'stepid', 'consec'}   # Consec stream adds 'consec'
expected.discard('reset')          # f8817c40: not recorded in transitions
```

### 3.4 Driver: the `'reset'` contract

`f8817c40` changed

```diff
-    acts['reset'] = obs['is_last'].copy()
-    self.acts = acts
-    trans = {**obs, **acts, **outs}
+    self.acts = {**acts, 'reset': obs['is_last'].copy()}
+    trans = {**obs, **acts, **outs, **logs}
```

so `'reset'` is kept only for the next env step and is no longer recorded.
Corroborating: `embodied/jax/agent.py:52` asserts `'reset' not in act_space`,
and `dreamerv3/main.py:132` strips it.

The two assertions were **ported, not deleted**. The old expectation
(`seq['reset'] == [0,0,0,0,0,1,...]`) was byte-identical to the `is_last`
expectation on the line above it, so the signal is still asserted; what was
added is an explicit check of the new contract plus the driver's own flag:

```python
assert 'reset' not in seq
assert (driver.acts['reset'] == seq['is_last'][-1]).all()
```

### 3.5 `.gitattributes`

```
*.sh text eol=lf
```

`entrypoint.sh` went from `ASCII text, with CRLF line terminators` to
`ASCII text`. `git check-attr` confirms `text: set`, `eol: lf`. The blob in git
was already LF, so the index did not change — the defect was only in how
Windows checkouts materialize the file, and `COPY . .` into the GPU image no
longer carries a script `sh` cannot run.

Deliberately narrow: only `*.sh`. The other tracked files remain CRLF in this
working tree, which is harmless for Python — the suite passes in-container with
CRLF sources, which is the proof.

---

## 4. The 9 remaining failures are a production defect

Not a test problem. `embodied/core/selectors.py:52`:

```python
def __delitem__(self, key):
    with self.lock:
      assert 2 <= len(self), len(self)
```

and `embodied/core/replay.py:_insert` evicts **before** inserting:

```python
while self.capacity and len(self.items) >= self.capacity:
    self._remove()
```

So at eviction time `len == capacity`. With `capacity == 1` the assertion
requires 2 and sees 1. Reproduced in four lines, without pytest:

```python
import embodied
replay = embodied.replay.Replay(length=1, capacity=1)
replay.add({'step': 0})     # len = 1
replay.add({'step': 1})     # AssertionError: 1
```

```text
    del self.sampler[itemid]
  File "/src/embodied/core/selectors.py", line 52, in __delitem__
    assert 2 <= len(self), len(self)
AssertionError: 1
```

Exactly the 9 failing parameterizations are the `capacity=1` ones
(`[1-1-1-*]`, `[1-1-128-*]`). The defect was previously masked: those tests
failed earlier on `dataset()` and never reached the eviction path.

Impact in practice: low. Eviction only trips the assertion when `capacity == 1`,
and the default is `replay.size: 5e6`. But it is a genuine correctness defect —
the guard is off by one for the evict-before-insert order.

The one-line fix would be `assert 1 <= len(self), len(self)`. It was **not
applied**: this tranche puts replay semantics under HOLD, and the rule is not
to change production behaviour to make tests pass. It needs an owner decision.

---

## 5. The 4 skipped tests, and the coverage question

`embodied/tests/test_parallel.py::TestParallel` (4 parameterizations) is marked
`pytest.mark.skip` with the full reason in the source. Summary:

`combined()` ends in `portal.run(workers)` and every worker loop is
`while True`. `args.duration` is no longer read anywhere in
`embodied/run/parallel.py`. The test asserts on stats collected **after**
`combined()` returns, which it now never does. Measured: one parameterization
ran 610.27 s before `pytest-timeout` killed it, then unwound with
`RuntimeError: cannot schedule new futures after interpreter shutdown`.

Two ways out, both outside this tranche:

1. Restore a duration-bounded run in production
   (`portal.run(workers, duration=args.duration)`) — a production change.
2. Redesign the test to run `combined()` in a subprocess and kill it — that
   invents a test contract rather than restoring one.

**Was coverage lost?** No, measured against the baseline: at `0e7332a0` this
module could not be imported at all and contributed nothing. It now collects,
its imports and call signatures are current, and the single remaining blocker
is named precisely. The honest statement is that **the parallel run loop is
untested, before and after** — the difference is that we now know exactly why.

No `xfail` was added, no assertion was loosened, and nothing was skipped to
improve a number. The skip is visible in the summary line (`4 skipped`) and
carries its reason in the test output.

---

## 6. Phase 4 — verification

All runs inside the container, no network, repository mounted read-only.

### 6.1 Commands

```powershell
# image: dreamerv3-cpu-test = dreamerv3-cpu + pytest 8.3.4 + pytest-timeout 2.3.1
podman run --rm --network none -v "E:\repos\dreamerv3:/src:ro" dreamerv3-cpu-test sh -c `
  "cd /src && python -m pytest embodied/tests -q --collect-only -p no:cacheprovider"
podman run --rm --network none -v "E:\repos\dreamerv3:/src:ro" dreamerv3-cpu-test sh -c `
  "cd /src && python -m pytest embodied/tests -q --tb=no -p no:cacheprovider --timeout=900"
podman build -f Containerfile.podman -t dreamerv3-cpu .
.\scripts\podman-cpu.ps1 smoke
```

### 6.2 Results

| Step | Result |
|---|---|
| 1. Collection, whole suite, no `--ignore` | **201 tests collected, 0 errors** (2.25 s) |
| 2. Full suite, run 1 | **9 failed, 188 passed, 4 skipped** (34.80 s) |
| 3. Full suite, run 2 | **9 failed, 188 passed, 4 skipped** (34.09 s) |
| 4. Failures by module | 9 × `embodied/tests/test_replay.py`, all `capacity=1` |
| 5. CPU container build | exit 0, `localhost/dreamerv3-cpu:latest` → `4b90afb2c8e2` |
| 6. Smoke on the rebuilt image | `podman-cpu.ps1 smoke` exit 0; 32 metric records in `/logdir/smoke/metrics.jsonl`, last step 1968 of 2000; checkpoint written |
| 7. Offline | every test run above executed under `--network none` |
| 8. Line endings | `entrypoint.sh`: LF; `git check-attr` → `text: set`, `eol: lf` |
| 9. Scope audit | 6 test files + `.gitattributes`; 0 production files |

### 6.3 Environment-dependent tests

- `test_parallel.py` (4, skipped): spawns processes and binds loopback ports;
  the measured 610 s hang is a contract issue, not flakiness, but this module
  would in any case be the most environment-sensitive in the suite.
- `test_train.py` (1, passing, 12.66 s): asserts `stats['lifetime'] >= 1` and
  uses wall-clock intervals (`save_every=0.2`), so it is timing-sensitive by
  construction. It passed on every run here; on a loaded host it could become
  flaky.
- The other 187 passing tests are pure computation with no timing or network
  dependence.

### 6.4 API contracts changed

None in production. The contracts listed in section 2.2 were changed by
upstream in `f8817c40`; this work only brought the tests up to them.

---

## 7. Remaining risks

| Risk | Severity | Owner decision needed |
|---|---|---|
| `Uniform.__delitem__` off-by-one makes `capacity=1` replay unusable | P1 correctness, low practical impact | yes — one-line production fix |
| The parallel run loop is untested | P1 coverage | yes — restore duration, or redesign the test |
| `embodied/perf/{test_bandwidth,test_distr}.py` un-collectable | P2 | no — leave until `portal` offers an equivalent |
| `args.from_checkpoint_regex` is read by `train.py:89` but absent from `configs.yaml` | P2 latent | yes — `--run.from_checkpoint <path>` would fail in production; unverified, since the branch is only taken with a checkpoint path |
| `pandas` undeclared, so `plot.py` cannot run | P2 | no — one line in `requirements.txt` |
| Container runs as root, no resource limits, unpinned `crafter`, base image by tag | P2 | yes — see the security document, sections 2-5 |

### 7.1 Correction to the canonical audit

The audit (section I-2) stated that `matplotlib`, `pandas` and `ruamel.yaml`
were all missing. Measured in the image: `matplotlib==3.11.2`,
`ruamel.yaml==0.19.1` and `tqdm==4.70.1` are present transitively; **only
`pandas` is missing**, and it alone blocks `plot.py`:

```text
plot.py imports: FAILED - ModuleNotFoundError No module named 'pandas'
```

The point about declaration still stands — these packages are relied on
transitively and are not in `requirements.txt` — but the audit's claim that
three were absent was wrong, and it is corrected here.

---

## 8. Recommendation

```text
GO / CONDITIONAL / NO-GO:   CONDITIONAL
```

Conditional, not GO, for one reason only: the suite is not green, and the two
items keeping it from green are both owner decisions, not engineering gaps.

What is now true that was not before:

- The suite collects completely and runs in 34 seconds instead of aborting.
- Every remaining failure has a named, reproduced root cause in production code.
- The one hang is bounded by a documented skip, so the suite is usable as a
  regression gate.
- The GPU container path is no longer broken by this checkout.
- Nothing in production, in the algorithm, in replay semantics or in the
  container was touched: `git diff` against `0e7332a0` is six test files and
  `.gitattributes`.

This is enough to develop on. It is not yet a clean gate.

### 8.1 Next decisions for the owner

1. **`selectors.py:52`** — approve `assert 1 <= len(self)`? One line, closes 9
   failures and a real defect. (My recommendation: yes.)
2. **The parallel run loop** — restore `duration` in production, or redesign
   the test, or accept it as untested? (My recommendation: restore `duration`,
   since the loop is otherwise untestable and the parameter already exists in
   the test's args.)
3. **Commit and push this branch, and PR #1** — both still on HOLD. The work is
   committed locally on `fix/test-baseline-v1` only.
4. **The security tranche** (security document sections 2-5: pin `crafter`,
   pin the base image by digest, non-root user, resource limits) — roughly ten
   lines across two files, all verifiable by rebuild plus smoke.

### 8.2 What JCode can do without a new GO

Nothing further in this tranche: everything authorized has been done. Items 1,
2 and 4 above each need a decision first, and items 1 and 2 touch production.
