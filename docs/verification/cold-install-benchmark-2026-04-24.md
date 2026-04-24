# Cold-Install Benchmark — 2026-04-24

This note records the reproducible command and current evidence for the
Phase 1.1 cold-install gate tracked by issue #54.

## Gate

- Test-plan row: Phase 1.1, editable dev install.
- Target: `pip install -e '.[dev]'` cold install completes in under 180 s.
- Cold-install definition: empty target directory, `--no-cache-dir`, and a fresh
  `TMPDIR` so locally cached wheels are not reused.

## Linux Measurement

Reference host:

- OS: Linux `hkust-Rack-Server`, kernel `6.8.0-60-generic`, `x86_64`.
- Python: `3.12.3` from `.venv/bin/python`.
- pip: `26.0.1` from `.venv`.
- Branch: `issue53_54`.

Command:

```bash
TARGET="/tmp/autoqec-cold-install-target-$(date +%Y%m%d%H%M%S)"
TMPDIR_PATH="/tmp/autoqec-cold-install-tmp-$(date +%Y%m%d%H%M%S)"
mkdir -p "$TARGET" "$TMPDIR_PATH"
TIMEFORMAT=$'real %3R\nuser %3U\nsys %3S'
time env TMPDIR="$TMPDIR_PATH" \
  .venv/bin/python -m pip install -e '.[dev]' \
  --target="$TARGET" \
  --no-cache-dir
```

Observed results:

| Attempt | Network mode | Result | Wall-clock |
|---|---|---|---:|
| Linux A | sandboxed / no network | failed while resolving `setuptools>=70` | 8.048 s |
| Linux B | network escalation allowed | manually interrupted before completion | > 1242.3 s |

The Linux cold-install gate is therefore not demonstrated as passing on this
reference host. The interrupted run already exceeded the 180 s target, so the
current evidence should be treated as a gate failure until a faster dependency
strategy or a completed passing measurement is recorded.

## Blocker Profile

Installed package footprint in the existing development virtualenv:

| Package | Version | Installed footprint |
|---|---:|---:|
| `torch` | 2.11.0 | 1184.1 MB |
| `numpy` | 2.4.4 | 68.3 MB |
| `ldpc` | 2.4.1 | 34.1 MB |
| `ruff` | 0.15.11 | 27.8 MB |
| `stim` | 1.15.0 | 14.5 MB |
| `pymatching` | 2.3.1 | 1.7 MB |

The dominant dependency is `torch>=2.11.0`; any realistic < 180 s cold gate
needs either pre-provisioned wheels, a CPU-only wheel strategy documented for
the target platform, or a revised gate that separates project install time from
third-party wheel download time.

## Windows Measurement

The 2026-04-23 verification report was run on Windows 11 and recorded only a
warm install: 37.4 s. No Windows cold-install run is available from this
Linux-only workspace. Use the same `--target` / `--no-cache-dir` command from a
Windows shell and append the result here before claiming the Windows cold gate
passes.

PowerShell equivalent:

```powershell
$Target = Join-Path $env:TEMP ("autoqec-cold-install-target-" + (Get-Date -Format yyyyMMddHHmmss))
$TmpDir = Join-Path $env:TEMP ("autoqec-cold-install-tmp-" + (Get-Date -Format yyyyMMddHHmmss))
New-Item -ItemType Directory -Force $Target, $TmpDir | Out-Null
Measure-Command {
  $env:TMPDIR = $TmpDir
  .venv\Scripts\python.exe -m pip install -e '.[dev]' --target=$Target --no-cache-dir
}
```

## Gate Decision

Revise the Phase 1.1 gate from a bare `< 180 s` assertion to a reproducible
benchmark record: if a cold install exceeds 180 s, the verification report must
record the wall-clock, platform, pip command, and dominant dependency blocker.
The current Linux evidence exceeds the target and points to `torch` as the
blocker; the project should not advertise a passing cold-install gate until a
completed Linux + Windows cold measurement is under 180 s or a wheel-caching
strategy is explicitly part of the setup instructions.
