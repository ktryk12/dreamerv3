# DreamerV3 with Podman

This fork keeps upstream's `Dockerfile` untouched and adds a separate,
CPU-first container setup for Podman. The goal is a small image that builds in
minutes on a Windows workstation, so we can run and modify the agent before
dealing with GPU passthrough.

## Why a second file

Upstream's `Dockerfile` starts from the NVIDIA driver image and installs DMLab,
MineRL, Atari, ProcGen, dm_control and memory_maze. That is the right setup for
reproducing the paper benchmarks on a CUDA machine, but it is a large build and
it cannot run without a GPU. `Containerfile.podman` installs CPU JAX, upstream's
`requirements.txt` minus the CUDA pins, and Crafter as a lightweight
environment.

Podman builds Dockerfiles natively, so nothing in DreamerV3 itself is modified.

## Files

| File | Purpose |
| --- | --- |
| `Containerfile.podman` | CPU image definition |
| `.containerignore` | Build context exclusions for Podman |
| `scripts/podman-cpu.ps1` | Build / smoke / train / shell / logs / clean |
| `PODMAN.md` | This document |

## Pinned JAX stack

`requirements.txt` pins `jax[cuda12]==0.4.33` but leaves `chex` and `optax`
unpinned. Their current releases require `jax >= 0.7`, and that JAX rejects the
positional `jax.jit` arguments used in `embodied/jax/transform.py`:

```
TypeError: jit() takes 1 positional argument but 5 were given
```

So the image installs `jax[cpu]==0.4.33`, `chex==0.1.87` and `optax==0.2.3`
after the rest of the requirements. pip prints dependency-conflict warnings for
the newer chex/optax metadata; the pinned combination is the one that runs.
Revisit this when upstream updates its JAX pins.

## Requirements

- Podman 5.x on Windows with a running machine (`podman machine start`)
- The WSL machine needs more than the default 2 GiB of memory for training
  runs. 8 GiB is a reasonable starting point:

  ```powershell
  podman machine stop
  podman machine set --memory 8192 --cpus 8
  podman machine start
  ```

## Usage

```powershell
.\scripts\podman-cpu.ps1 build   # build the image
.\scripts\podman-cpu.ps1 smoke   # short Crafter run on the debug config
.\scripts\podman-cpu.ps1 train   # full Crafter training run
.\scripts\podman-cpu.ps1 shell   # interactive bash in the image
.\scripts\podman-cpu.ps1 logs    # list the log volume
.\scripts\podman-cpu.ps1 clean   # remove image and log volume
```

Extra arguments after `train` are forwarded to `dreamerv3/main.py`:

```powershell
.\scripts\podman-cpu.ps1 train --run.train_ratio 32
```

Logs live in the named volume `dreamerv3-logs`, mounted at `/logdir`. To write
to a host directory instead, swap the volume for a bind mount:

```powershell
podman run --rm -v E:\logdir:/logdir dreamerv3-cpu `
  python dreamerv3/main.py --logdir /logdir/run --configs crafter debug
```

## GPU status

GPU support is **not** qualified in this fork yet. The workstation's Quadro
M6000 is Maxwell-generation; current JAX CUDA wheels target newer compute
capabilities, so GPU runs will likely need either an older JAX/CUDA combination
or upstream's `Dockerfile` on different hardware. Until that is tested, treat
this setup as CPU-only.

## Staying in sync with upstream

```powershell
git fetch upstream
git switch main
git merge upstream/main
git push origin main
```

`Containerfile.podman` deliberately reuses upstream's `requirements.txt`, so
dependency bumps from `danijar/dreamerv3` carry over without editing our files.

Upstream's MIT license and copyright notices are unchanged.
