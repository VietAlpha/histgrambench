# Compiling model-generated TikZ safely

Contestant answers are untrusted LaTeX written by models. The harness never compiles them on the host. Analysis-only
reproduction (level 1, and level 2 without recompilation) needs **no** compiler and no Docker.

## What protects the host

The answer passes three layers, in this order (code in `hmb/submission/`):

1. **Extraction and policy** (`extract.py`, `wrapper.py: normalize`): exactly one fenced LaTeX block with exactly one
   `tikzpicture`; packages and TikZ libraries from the published allowlist only; commands that read or write files,
   run shell code, include images or select fonts are rejected before compilation (`P_POLICY_REJECT`).
2. **Container** (`sandbox.py: container_args`): a fresh container per answer with `--network none`, `--read-only`
   root file system, a size-limited `tmpfs` at `/job` (`noexec`), a non-root user (uid 10001), `--cap-drop ALL`,
   `no-new-privileges`, a process limit, `--memory` equal to `--memory-swap`, and one CPU.
3. **Engine and job limits** (`sandbox/run-job.sh`, `sandbox/texmf.cnf.hmb`): `lualatex --nosocket
   --no-shell-escape --interaction=nonstopmode --halt-on-error`, `shell_escape = f`, `openout_any = p`, CPU-time and
   file-size `ulimit`s, a page and raster-size cap before `pdftoppm` runs, and its own CPU limit.

`--safer` is not used, because luaotfload aborts under it. The image self-test (`python -m hmb sandbox-selftest`) asserts the
compensating controls: `os.execute` is refused, `io.popen` is unavailable, Lua writes are confined, and the container
confines reads.

## Limits (registered parameters, `frozen/thresholds.json`)

| Parameter | Value | Outcome when exceeded |
|---|---|---|
| `cpu_seconds` | 20 | `C_TIMEOUT` |
| `wall_seconds` | 60 | `C_WALLCLOCK` |
| `memory_bytes` | 1 GiB | `C_RESOURCE` |
| `output_bytes` | 32 MiB | `C_RESOURCE` |
| `log_bytes` | 4 MiB | `C_RESOURCE` |
| `page_cap` | 4 pages | `C_RESOURCE` (more than one page also gives `V_MULTIPAGE`) |
| `page_dim_cap_cm` | 200 cm | `V_OVERSIZE` |
| `render_pixel_cap` | 100,000,000 px | `V_OVERSIZE` |
| `sandbox_pids_limit` | 64 | process limit |
| `raster_cpu_seconds` | 20 | `C_RESOURCE` |

These values define outcomes; they are part of the frozen protocol. Do not raise them to make an answer compile, and
do not run the sandbox with weaker flags. Compile one answer at a time unless you have checked your host: CPU seconds
inside a virtual machine depend on load. The run used 4 concurrent compiles on its cloud host after
`python -m hmb sandbox-cputime-check` showed no outcome change (CPU-time ratio median 1.05).

## Building the image

```bash
docker info                      # Docker Engine with a Linux backend
python -m hmb sandbox-build      # builds hmb-sandbox:v1 from sandbox/Dockerfile and REWRITES sandbox/image.lock.json
python -m hmb sandbox-selftest   # must print ALL PASS
git diff sandbox/image.lock.json # shows how your image differs from the run's
```

The Dockerfile pins the TeX Live base image by digest and the Debian package snapshot by date, but the image
identity (`sha256:e902dd5a...` in the run, measured on Docker's containerd image store) will differ after a rebuild,
so `harness_version` will differ from the run's even when every compile outcome agrees. The image of the run is
not distributed (8.9 GB; it bundles TeX Live, Debian packages and fonts under their own licences). The level-2
recompilation check (`scripts/recompile_sample.py`) measures outcome agreement on a stratified sample.

## Running a recompilation check

```bash
HMB_ROOT=../hgb_root python scripts/recompile_sample.py --per-outcome 3 --work ../recompile_scratch --out recompile.json
```

The sample includes every outcome code that occurs (truncated, no TikZ, policy rejections, compile errors, timeouts,
resource and output faults, valid drawings). Compiling an answer that the original run classified `C_TIMEOUT` may
land on either side of the 20-second limit on another machine; treat a `C_TIMEOUT` disagreement as a host-speed
effect, and any other disagreement as a reproduction failure.
