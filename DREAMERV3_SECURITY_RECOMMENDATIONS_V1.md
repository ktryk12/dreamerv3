# DreamerV3 — security and reproducibility recommendations V1

```text
SCOPE:     E:\repos\dreamerv3, branch fix/test-baseline-v1 (from 0e7332a0)
DATE:      2026-10-22
AUTHORITY: documentation only. Nothing in sections 2-7 was applied; the
           container architecture was left unchanged as instructed.
APPLIED:   section 1 only (.gitattributes), which was explicitly authorized.
```

Risk profile: a local, single-operator research trainer with no credentials, no
authority surface and no production effect. Every recommendation below is sized
to that, and each one says plainly what it buys.

---

## 0. What was measured

| Check | Result |
|---|---|
| Secrets in tracked files | **none.** A scan for `api_key`, `secret`, `password`, `token`, `BEGIN (RSA\|OPENSSH\|PRIVATE)` across all 99 tracked files returned 4 files, all false positives — `tokens` in the machine-learning sense (`agent.py:119`, `rssm.py:61`, …) |
| CUDA packages in the CPU image | **zero.** `pip freeze \| grep -ciE "^nvidia\|cuda"` returns 0; `jax==0.4.33`, `jaxlib==0.4.33` are CPU wheels |
| Dependency consistency | `pip check` reports `No broken requirements found` |
| Offline capability | the full test suite runs under `podman run --network none` |
| Container user | `uid=0` (root), no `USER` directive |
| Resource limits in our wrapper | none set (`--memory`, `--cpus`, `--pids-limit`, `--read-only`, `--cap-drop` all absent) |
| Outbound network in the Python sources | no `subprocess`, `os.system`, `eval(`, `exec(`, `pickle`, `requests`, `urllib`, `socket` |

---

## 1. Line endings — APPLIED

`core.autocrlf=true` plus no `.gitattributes` meant `entrypoint.sh` was checked
out with CRLF. The upstream `Dockerfile` does `COPY . .` and
`ENTRYPOINT ["sh", "entrypoint.sh"]`, so the GPU image built from this checkout
carried a script `sh` cannot execute.

Applied: a `.gitattributes` containing `*.sh text eol=lf`, followed by a
re-checkout of `entrypoint.sh`.

```text
before:  entrypoint.sh: ASCII text, with CRLF line terminators
after:   entrypoint.sh: ASCII text
         0000000   s   e   t       -   e  \n  \n   ...
```

The blob stored in git was already LF, so the index did not change — the defect
was purely in how Windows checkouts materialize the file, and it is now fixed
for every future checkout and clone.

---

## 1b. Measured: the image is not reproducible

Added after a clean rebuild on 2026-10-23. `podman build --no-cache
--pull=always` took 381 s and produced a working image (`dreamerv3-cpu-clean`,
smoke exit 0, no warnings), but **not the same image**. Comparing
`pip freeze` between the cached build and the clean one, nine days apart:

```text
cached: 100 packages        clean: 101 packages

certifi           2026.7.22    -> 2026.10.19
fastapi           0.143.1      -> 0.161.1
filelock          4.1.2        -> 4.2.4
google-api-core   2.42.0       -> 2.43.0
google-cloud-...  3.17.0       -> 3.17.1
googleapis-...    1.75.5       -> 1.76.0
granular          0.24.1       -> 0.24.2
pillow            12.3.0       -> 13.0.0      <- major
pycparser         3.11         -> 3.12
python-dateutil   2.9.0.post0  -> 3.0.0       <- major
starlette         1.8.0        -> 1.9.0
wcwidth           0.9.2        -> 0.9.3
websockets        17.2         -> 17.3
tzdata            (absent)     -> 2026.5      <- new
```

Fourteen version changes and one new package in nine days, two of them major
version bumps, in an image whose only declared pins are in upstream's
`requirements.txt`. The build happened to still work; that is the measurement,
not a guarantee. This is the concrete argument for sections 2 and 3 below:
without a digest on the base image and a pin on `crafter`, "rebuild the image"
and "rebuild the same image" are different operations, and a training result
cannot be attributed to a specific software state.

## 2. Base image is pinned by tag, not digest (recommended)

`Containerfile.podman` begins `FROM docker.io/library/python:3.11-slim`. The
tag moves; the build is therefore not reproducible across time.

The digest currently in use on this machine:

```text
docker.io/library/python:3.11-slim
  sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
```

Recommended change (one line):

```dockerfile
FROM docker.io/library/python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
```

Buys: an identical base layer on every rebuild. Costs: the digest must be
refreshed deliberately when picking up security updates, which is the point.

---

## 3. `crafter` is installed unpinned (recommended)

`RUN pip install crafter` resolves to whatever is newest at build time. It
currently resolves to `crafter==1.8.3`.

Recommended: `RUN pip install crafter==1.8.3`.

Buys: the environment the agent trains against stops changing underneath us,
which matters because Crafter is our reward signal. This is the single highest
value pin in the file.

---

## 4. The container runs as root (recommended)

```text
$ podman run --rm dreamerv3-cpu id -u
0
```

Impact is limited — Podman runs rootless on this host, so container root maps
to the unprivileged user — but a training process has no need for it, and the
image writes only to `/logdir` and the in-image `/app`.

Recommended, after the `pip install` layers:

```dockerfile
RUN useradd --create-home --uid 1000 dreamer && chown -R 1000:0 /app
USER 1000
```

Verification: `id -u` is non-zero and `podman-cpu.ps1 smoke` still trains. Note
that the named volume `dreamerv3-logs` will need its ownership checked on first
use after the change — that is the one thing that can break, so it belongs in
the same tranche as the test, not before it.

---

## 5. No resource limits on training runs (recommended)

`scripts/podman-cpu.ps1` sets no limits. A long DreamerV3 run on 18 CPUs with
472 GiB of host memory available can starve the rest of the workstation, which
also hosts other Podman workloads.

Recommended additions to the `train` and `smoke` invocations:

```powershell
'--memory', '16g', '--cpus', '8', '--pids-limit', '512'
```

Buys: a runaway or leaking run degrades one container instead of the host. This
is the recommendation with the most practical value for day-to-day use, because
the host is shared.

---

## 6. Upstream Dockerfile supply chain (recorded, not recommended for change)

On the GPU path, which we do not currently build:

| Line | Issue |
|---|---|
| `wget -O - https://gist.githubusercontent.com/.../install-dmlab.sh \| sh` | an unpinned remote script, fetched over the network and piped into a shell at build time, from a gist that can be edited in place |
| `pip install ale_py==0.9.0 autorom[accept-rom-license]==0.6.1` | accepts a licence and downloads Atari ROMs non-interactively |
| MineRL from a GitHub release URL | a release asset, not a package index |
| `FROM ghcr.io/nvidia/driver:7c5f8932-550.144.03-ubuntu24.04` | tag, not digest |
| `entrypoint.sh` | queries the GCP metadata service on every start: `curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/...`. Off GCP it fails and prints `NA`, but it is an unconditional outbound attempt |

These are upstream's decisions. Changing them diverges the fork for no present
benefit, because the CPU image does not use any of them — it has its own base,
installs no envs from remote scripts, and does not run `entrypoint.sh`.

If the GPU path is ever qualified, the gist line is the one to replace first:
vendor the script into the repository so its content is reviewable and
versioned.

---

## 7. `cloudpickle` over `portal` RPC (recorded, conditional)

`embodied/run/parallel.py:32-38` serializes factory functions with
`cloudpickle`; `embodied/core/driver.py:21` does the same for env constructors.
Deserializing cloudpickle executes code by design.

Current exposure: none beyond the host. Default addresses are
`localhost:{auto}` (`dreamerv3/configs.yaml:61-63`), and `portal` resolves a
bare `host:port` — it explicitly rejects any address containing `://`
(`portal/client_socket.py:39`, `portal/server_socket.py:28`). So nothing listens
on a routable interface unless an operator changes those addresses.

This becomes a real trust boundary only in two future cases: multi-node
training, or any Virkora integration where a process outside this trainer can
reach a `portal` port. In both cases the rule is that the RPC surface must stay
on loopback or inside one network namespace, and must never be reachable from a
component with a different trust level.

---

## 8. Data writes

| Path | Written by | Persistence |
|---|---|---|
| `/logdir/<run>/metrics.jsonl`, `scores.jsonl` | logger | named volume `dreamerv3-logs` |
| `/logdir/<run>/ckpt/` | checkpointer | same |
| replay chunks | replay, when `--replay.directory` is set | same |
| `/app` | nothing at runtime, but writable because the process is root | ephemeral |

No host path is bind-mounted by `podman-cpu.ps1`; it uses a named volume, so a
run cannot touch the working tree. The audit's own test runs mounted the
repository explicitly with `:ro`.

A `--read-only` rootfs with a `--tmpfs /tmp` would close the writable `/app`,
and is worth doing in the same tranche as section 4.

---

## 9. Priority

| # | Recommendation | Priority | Authorization needed |
|---|---|---|---|
| 1 | `.gitattributes` for `*.sh` | P1 | **applied** |
| 2 | Pin `crafter==1.8.3` | P2 | owner GO (container change) |
| 3 | Resource limits in `podman-cpu.ps1` | P2 | owner GO |
| 4 | Pin the base image by digest | P2 | owner GO |
| 5 | Non-root `USER` + `--read-only` rootfs | P2 | owner GO |
| 6 | Vendor the DMLab install script | P3 | only if the GPU path is qualified |
| 7 | Keep `portal` on loopback | P3 | a design constraint, not a change |

Items 2-5 are a single coherent tranche of roughly ten lines across two files,
and all four are verifiable by rebuilding and re-running the smoke test. They
are left unapplied because this tranche's authorization covers test fixes,
compatibility fixes and `.gitattributes` only.
