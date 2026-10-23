# DreamerV3 — dependency reproducibility V1

```text
BRANCH:   chore/dependency-lock-v1, from canonical main @ b8df67a
DATE:     2026-10-23
SCOPE:    base image digest, a hash-pinned Python lock, a controlled crafter
          pin, and verification by two independent clean builds
```

The problem this closes was measured, not assumed:
`DREAMERV3_BASELINE_CLOSEOUT_V1.md` section 7.2 recorded 14 package version
changes and one new package across nine days, including `pillow 12.3.0 →
13.0.0` and `python-dateutil 2.9.0 → 3.0.0`, with no change to the repository.

---

## 1. The documented environment

```text
Python:        CPython 3.11.17
Platform:      linux x86_64 (manylinux wheels), glibc 2.41
Base image:    docker.io/library/python:3.11-slim
               @sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377
Resolver:      pip 26.2.1
Manifest:      requirements.lock.txt, 101 packages, every one with a sha256
Install:       pip install --require-hashes --no-deps -r requirements.lock.txt
```

The base image tag moved under us during this work: the audit recorded
`sha256:e88e9763…` on 2026-10-14 and `python:3.11-slim` resolved to
`sha256:b7a9e5f6…` on 2026-10-23, with the interpreter's own build date moving
from Oct 6 to Oct 21. That is the drift, visible in one line. The digest above
is the one now pinned.

---

## 2. What changed

| File | Change |
|---|---|
| `requirements.lock.txt` | new, generated: 101 packages, each `name==version` plus `--hash=sha256:…` |
| `scripts/lock-deps.sh` | new: regenerates the lock from `requirements.txt` plus the CPU pins |
| `Containerfile.podman` | base image pinned by digest; the three `pip install` steps replaced by one `--require-hashes --no-deps` install from the lock |

No production Python changed. No test changed.

### 2.1 Upstream inheritance is preserved, but made explicit

The previous `Containerfile.podman` read `requirements.txt` directly, so bumps
from `danijar/dreamerv3` arrived automatically — and so did everything else,
which is precisely why the image drifted.

`requirements.txt` is still the input, but now to `scripts/lock-deps.sh` rather
than to the build:

```text
requirements.txt  (upstream, minus the CUDA-only lines)
      +
CPU pins in scripts/lock-deps.sh:
      jax[cpu]==0.4.33, chex==0.1.87, optax==0.2.3, crafter==1.8.3
      |
      v  pip resolve --report, in CPython 3.11 on linux x86_64
requirements.lock.txt  (101 packages, hashed)
      |
      v
Containerfile.podman
```

Upstream bumps now reach us as a reviewable diff from one command instead of
silently at build time. That is a deliberate trade: inheritance becomes a step
someone performs and reads, which is the only form of inheritance compatible
with a reproducible build.

### 2.2 Why `--require-hashes --no-deps`

`--require-hashes` makes pip refuse any requirement without a hash and any
requirement that is not pinned exactly, so a lock that is missing a transitive
dependency fails the build instead of silently resolving it. `--no-deps` then
guarantees nothing outside the 101 listed packages can enter. Together they
turn "pinned" into "complete" — and the smoke tests in section 4 are what prove
the set is actually sufficient.

### 2.3 The crafter pin

`crafter==1.8.3`, pinned inside the lock. It is the environment the agent
trains against, so an unpinned version means the reward signal can change
between runs. This is the single highest-value pin in the file.

---

## 3. Verification: two independent clean builds

Both with `--no-cache`, from the same commit:

```text
build A:  exit 0, 219 s, image 9b668fb70fc6, 1.73 GB
build B:  exit 0, 313 s, image dc870103ce2e, 1.73 GB

pip freeze A: 101 packages
pip freeze B: 101 packages
diff A B    : PACKAGE SETS IDENTICAL

pip check (A): No broken requirements found
python -VV (A): Python 3.11.17 (main, Oct 21 2026, 01:07:38) [GCC 14.2.0]
```

Also measured: the lock matches the earlier unpinned clean build exactly
(101 vs 101 packages, zero version differences), so pinning introduced **no
dependency upgrade** — it froze the set that was already being produced.

### 3.1 What is *not* claimed

**The images are not bit-identical.** Measured: build A is
`9b668fb70fc6`, build B is `dc870103ce2e`. Identical inputs produced different
image ids, as expected — the `apt-get` layer and filesystem timestamps are not
reproducible, and nothing in this change attempts to make them so.

What is verified is weaker and is the thing that matters for attributing a
result: **the installed Python package set is identical, by name and version,
across independent clean builds, and every wheel is fetched by hash.** Bit
identical images would need a reproducible-builds effort (pinned apt snapshots,
fixed timestamps, `SOURCE_DATE_EPOCH`) that was not authorized and is not
needed to answer "which software produced this number".

---

## 4. Offline smoke tests of the finished images

Both images, `--network none`, named volume:

```text
dreamerv3-lock-a:  SMOKE_EXIT=0,  4 agent-step reports over 600 steps
dreamerv3-lock-b:  SMOKE_EXIT=0,  5 agent-step reports over 600 steps
```

The report counts differ because logging is driven by wall clock
(`log_every`), not by step count; the runs are the same length. This is a
logging cadence difference, not nondeterminism in training.

Full test suite against a locked image (`dreamerv3-lock-a` plus
`pytest==8.3.4`, `pytest-timeout==2.3.1`):

```text
222 passed, 4 skipped in 70.99s
```

So the lock neither breaks the runtime nor the test path.

---

## 5. Regenerating the lock

```powershell
podman run --rm -v "${PWD}:/repo" docker.io/library/python:3.11-slim `
  sh /repo/scripts/lock-deps.sh
git diff requirements.lock.txt      # read it
```

Then rebuild and re-run the suite and the smoke test before committing. The
script refuses to write a partial lock: if any resolved package has no sha256
it exits rather than emitting something `--require-hashes` would later accept
by accident.

Hashes are wheel hashes and therefore platform specific. Regenerating on a
different platform or Python version will produce a different lock, which is
why section 1 documents both.

---

## 6. Remaining limitations

| Limitation | Severity | Note |
|---|---|---|
| Images are not bit-identical | accepted | section 3.1; out of scope |
| The `apt-get` layer is unpinned | P3 | `ffmpeg`, `git`, `libgl1`, `libglib2.0-0` come from Debian's current index. Pinning needs a snapshot mirror |
| The base digest must be refreshed by hand | by design | it is the point; refresh it together with the lock |
| Container still runs as root, no resource limits | P2, open | `DREAMERV3_SECURITY_RECOMMENDATIONS_V1.md` sections 4 and 5 |
| The test image is not defined in the repository | P3 | pytest pins live in an ad-hoc Containerfile; worth adding if CI is set up |
