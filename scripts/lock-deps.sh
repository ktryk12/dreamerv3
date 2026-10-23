#!/bin/sh
# Regenerate requirements.lock.txt.
#
# Upstream's requirements.txt stays the input, so dependency bumps from
# danijar/dreamerv3 still reach us -- but as a reviewable step rather than
# silently at build time. Run this, read the diff, commit it.
#
#   podman run --rm -v "$PWD:/repo" docker.io/library/python:3.11-slim \
#     sh /repo/scripts/lock-deps.sh
#
# Resolution happens in the same Python and platform the image uses: CPython
# 3.11 on linux x86_64. Hashes are wheel hashes and are platform specific, so
# regenerating on another platform will produce a different lock.
set -e

REPO="${REPO:-/repo}"
WORK="$(mktemp -d)"

# The CPU pins. jax, chex and optax are held at the 0.4.33-era line because
# newer chex/optax require jax >= 0.7, which rejects the positional jax.jit
# arguments in embodied/jax/transform.py. crafter is our benchmark env and is
# pinned so the reward signal cannot change under us.
cat > "$WORK/pins.txt" <<'PINS'
jax[cpu]==0.4.33
chex==0.1.87
optax==0.2.3
crafter==1.8.3
PINS

# Upstream requirements minus the lines that have no CPU equivalent, and minus
# the two the pins above own.
grep -v -E '^(jax\[cuda|nvidia-cuda-nvcc|chex|optax)' "$REPO/requirements.txt" \
  > "$WORK/spec.txt"
cat "$WORK/pins.txt" >> "$WORK/spec.txt"

python -m venv "$WORK/venv"
. "$WORK/venv/bin/activate"
pip install -q -U pip setuptools
pip install --dry-run --ignore-installed \
  --report "$WORK/report.json" -r "$WORK/spec.txt" > "$WORK/resolve.log" 2>&1

PIP_VERSION="$(pip --version | cut -d' ' -f2)"
REPORT="$WORK/report.json" PIP_VERSION="$PIP_VERSION" \
  OUT="$REPO/requirements.lock.txt" python - <<'PY'
import json, os, pathlib
rep = json.loads(pathlib.Path(os.environ['REPORT']).read_text(encoding='utf-8'))
rows = []
for item in rep['install']:
    name = item['metadata']['name']
    ver = item['metadata']['version']
    info = (item.get('download_info') or {}).get('archive_info') or {}
    sha = (info.get('hashes') or {}).get('sha256')
    if not sha:
        raise SystemExit(f'no sha256 for {name}=={ver}; refusing to write a '
                         'partial lock')
    rows.append((name, ver, sha))
rows.sort(key=lambda r: r[0].lower())
lines = [
    '# Generated file. Do not edit by hand.',
    '#',
    '# Regenerate with scripts/lock-deps.sh, which resolves requirements.txt',
    '# (minus the CUDA-only lines) together with the CPU pins listed in that',
    '# script, then records every resolved package with its wheel hash.',
    '#',
    f"# Resolved for: CPython 3.11, linux x86_64 (manylinux), "
    f"pip {os.environ['PIP_VERSION']}",
    f'# Packages:     {len(rows)}',
    '#',
    '# Install with: pip install --require-hashes -r requirements.lock.txt',
    '',
]
for name, ver, sha in rows:
    lines.append(f'{name}=={ver} \')
    lines.append(f'    --hash=sha256:{sha}')
pathlib.Path(os.environ['OUT']).write_text('\n'.join(lines) + '\n',
                                           encoding='utf-8')
print(f'wrote {len(rows)} packages')
PY

rm -rf "$WORK"
