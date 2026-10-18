r"""§9.6 sandboxed compilation (Docker), post-compile checks, and outcome coding (§9.5).

TeX is NEVER run on the host. `compile_in_sandbox` raises `SandboxUnavailable` when Docker or the
pinned image is missing; there is no host fallback.

Container (every run): `--network none`, read-only root fs, a tmpfs job dir that receives only
`cand.tex` (piped over stdin; there are no bind mounts at all, so no crop, card or item exists in the
namespace), uid 10001, `--cap-drop ALL`, `--security-opt no-new-privileges`, pids limit, Docker's
default seccomp profile, memory limit with swap disabled, one CPU. Outputs come back as a tar stream
on stdout. Engine line: ENGINE_LINE below (also hard-coded in sandbox/run-job.sh; a test asserts equality).

DEVIATION (contradiction C16; owner decision D5 option A, 2026-09-23, recorded with the MS1 form
freeze): the §9.6 line minus `--safer`. Measured in the pinned image (LuaHBTeX 1.22.0, luaotfload v3.29):
luaotfload.lua:103 aborts under --safer ("luaotfload can't run with option --safer"), so fontspec and
the wrapper_v1 fallback chain cannot load at all. Compensating properties, asserted by the image
self-test on every build and by tests/w3 through real compiles: os.execute / io.popen refused
(shell_escape = f); Lua io.open(…,"w") outside TEXMFOUTPUT=/job/out returns nil (LuaTeX applies
openout_any = p to Lua io); os.remove / os.rename / lfs.mkdir outside the tmpfs fail (read-only
root, non-root uid); \write18 not executed; the container remains the read confinement.

Limits (all from Thresholds): cpu_seconds -> RLIMIT_CPU on lualatex (SIGXCPU -> C_TIMEOUT);
wall_seconds -> liveness kill of the container (C_WALLCLOCK, re-run once); memory_bytes -> cgroup
limit (OOM -> C_RESOURCE); output_bytes -> RLIMIT_FSIZE per file (SIGXFSZ -> C_RESOURCE);
log_bytes -> post-hoc log size (C_RESOURCE); page_cap -> C_RESOURCE above it, V_MULTIPAGE for any
other page count != 1; page_dim_cap_cm / render_pixel_cap -> V_OVERSIZE; sandbox_pids_limit;
raster_cpu_seconds -> RLIMIT_CPU on pdftoppm (overrun or failure -> C_RESOURCE).

C_SANDBOX_VIOLATION signatures (detected in the TeX log / terminal output; documented form choice):
  S1 `runsystem(` : a shell-escape (\\write18) request reached the engine and was refused.
  S2 `HMB-SANDBOX-VIOLATION <call> <path>` : logged by the wrapper_v1 Lua hook (installed after the
     font setup in every wrapper class) when the document calls os.execute/exec/spawn or io.popen,
     opens a file for writing outside /job, or calls os.remove/os.rename/lfs.mkdir/rmdir/touch/link/
     chdir on a path outside /job. The hook is telemetry for outcome coding, not a security control
     (a document could bypass it through the debug library); the calls themselves are refused by
     the engine configuration and the container regardless.
  S3 `! I can't write on file `<path>'` with an absolute path outside /job or a path containing
     `..` (kpathsea refused it under openout_any = p; measured: LuaTeX logs only this line), or
     kpathsea's `Not writing to <path> (openout_any = p)`: \\openout outside the output dir refused.
  S4 an opened file path in the log outside the TeX tree, the fonts, /opt/hmb and /job
     (e.g. `\\input{/etc/passwd}`): read outside the job and the TeX tree.
  Not detectable (disclosed): reads via \\openin or Lua io.open(…,"r"), which leave no log trace;
  they are confined by the container (nothing item-related exists in the namespace).
Precedence when several apply: C_WALLCLOCK > C_SANDBOX_VIOLATION > C_TIMEOUT > C_RESOURCE >
C_MISSING_GLYPH > C_ERROR, then V_MULTIPAGE > V_RASTER > V_OVERSIZE > V_EMPTY > OK.
"TeX capacity exceeded" and Lua "not enough memory" are coded C_RESOURCE (memory).
A run that ends with "No pages of output." has page count 0 -> V_MULTIPAGE (§9.5 "pages != 1").
"""
from __future__ import annotations

import io
import json
import math
import re
import shutil
import subprocess
import threading
import tarfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from hmb.hashing import sha256_bytes, sha256_json
from hmb.vocab import Outcome

from .render import canonicalize_image, ink_fraction

SANDBOX_DIR = Path(__file__).resolve().parents[2] / "sandbox"
IMAGE_LOCK = SANDBOX_DIR / "image.lock.json"
IMAGE_TAG = "hmb-sandbox:v1"
RUNTIME_UID = 10001
# C16: §9.6 engine line minus --safer (see module docstring). Hashed into harness_version.
ENGINE_LINE = ("lualatex --nosocket --no-shell-escape --interaction=nonstopmode "
               "--halt-on-error --output-directory=/job/out --jobname=cand /job/cand.tex")
PT_PER_CM = 72.0 / 2.54
TMPFS_SLACK = 8 * 1024 * 1024

SIGXCPU, SIGKILL, SIGXFSZ = 24, 9, 25

_ALLOWED_READ_ROOTS = ("/usr/local/texlive/", "/job/", "/opt/hmb/", "/usr/share/fonts/",
                       "/etc/fonts/", "/var/cache/fontconfig/", "/usr/share/texmf/",
                       "/usr/share/texlive/")
VIOLATION_PATTERNS = {
    "S1_shell_escape": re.compile(r"runsystem\("),
    "S2_lua_io_process": re.compile(r"HMB-SANDBOX-VIOLATION \S+"),
    "S3_openout": re.compile(r"Not writing to .*\(openout_any = p\)|I can't write on file `(?:/(?!job/)|[^']*\.\./)"),
    "S4_read_outside": re.compile(r"\((/[^\s()]+)"),
}
MISSING_GLYPH = re.compile(r"Missing character: There is no (.+?) \((U\+[0-9A-Fa-f]+)\) in font ([^!\n]+)!")
MISSING_GLYPH_ANY = re.compile(r"Missing character: There is no")
RESOURCE_LOG = re.compile(r"TeX capacity exceeded|not enough memory|No space left on device")


class SandboxUnavailable(RuntimeError):
    """Docker or the pinned sandbox image is unavailable. There is no host fallback."""


class SandboxHarnessError(RuntimeError):
    """The runner itself failed (not an item outcome); fails the run."""


# ----------------------------------------------------------------------------- image
def _docker(*args: str, input: bytes | None = None, timeout: float | None = None,
            check: bool = False) -> subprocess.CompletedProcess:
    exe = shutil.which("docker")
    if exe is None:
        raise SandboxUnavailable("docker executable not found")
    try:
        return subprocess.run([exe, *args], input=input, capture_output=True, timeout=timeout,
                              check=check)
    except FileNotFoundError as e:
        raise SandboxUnavailable(str(e)) from e


def read_lock() -> dict:
    if not IMAGE_LOCK.exists():
        raise SandboxUnavailable(f"{IMAGE_LOCK} missing: build the image (python -m hmb sandbox-build)")
    return json.loads(IMAGE_LOCK.read_text(encoding="utf-8"))


def sandbox_image_digest() -> str:
    """The pinned image id (sha256 of the image config) used for every run."""
    return read_lock()["image_id"]


def ensure_image() -> str:
    image_id = sandbox_image_digest()
    try:
        r = _docker("image", "inspect", "--format", "{{.Id}}", image_id, timeout=60)
    except subprocess.TimeoutExpired as e:
        raise SandboxUnavailable("docker daemon not responding") from e
    if r.returncode != 0 or r.stdout.decode().strip() != image_id:
        raise SandboxUnavailable(f"pinned sandbox image {image_id} not present: "
                                 + r.stderr.decode(errors="replace").strip()[:300])
    return image_id


def sandbox_available() -> bool:
    try:
        ensure_image()
        return True
    except SandboxUnavailable:
        return False


def build_image(tag: str = IMAGE_TAG, log_path: Path | None = None) -> dict:
    """docker build sandbox/ -> record image id + in-image manifest in sandbox/image.lock.json."""
    r = _docker("build", "--progress=plain", "-t", tag, str(SANDBOX_DIR), timeout=7200)
    if log_path:
        Path(log_path).write_bytes(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SandboxHarnessError("docker build failed:\n" + r.stderr.decode(errors="replace")[-3000:])
    image_id = _docker("image", "inspect", "--format", "{{.Id}}", tag, check=True).stdout.decode().strip()
    man = _docker("run", "--rm", "--network", "none", image_id, "cat", "/opt/hmb/image-manifest.json",
                  check=True).stdout.decode()
    files = {p.name: sha256_bytes(p.read_bytes()) for p in sorted(SANDBOX_DIR.iterdir())
             if p.is_file() and p.name != IMAGE_LOCK.name}
    lock = {"image_id": image_id, "tag": tag, "manifest": json.loads(man), "context_files_sha256": files,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engine_line": ENGINE_LINE}
    IMAGE_LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return lock


# ----------------------------------------------------------------------------- run
@dataclass
class RunArtifacts:
    wall_killed: bool
    oom_killed: bool
    container_exit: int | None
    wall_seconds: float
    files: dict[str, bytes] = field(default_factory=dict)   # "out/cand.pdf", "meta/exit", ...

    def text(self, name: str) -> str:
        return self.files.get(name, b"").decode("utf-8", errors="replace")


def _limits(th) -> dict:
    return {"cpu": int(th.value("cpu_seconds")), "wall": float(th.value("wall_seconds")),
            "mem": int(th.value("memory_bytes")), "out": int(th.value("output_bytes")),
            "log": int(th.value("log_bytes")), "page_cap": int(th.value("page_cap")),
            "dim_cm": float(th.value("page_dim_cap_cm")), "pix": int(th.value("render_pixel_cap")),
            "pids": int(th.value("sandbox_pids_limit")), "rcpu": int(th.value("raster_cpu_seconds")),
            "dpi": int(th.value("render_dpi_candidate")),
            # the dimension cap in points exactly as passed to run-job.sh (6 decimals)
            "dim_pt": float(f"{float(th.value('page_dim_cap_cm')) * PT_PER_CM:.6f}")}


# ONE shared raster-cap formula. run-job.sh evaluates the identical expression in awk on the raw
# pdfinfo CropBox points (w=x1-x0, h=y1-y0):
RASTER_CAP_AWK = "p==1 && w>0 && h>0 && w<=cap && h<=cap && (w*dpi/72)*(h*dpi/72)<=pix"


def raster_pixels(w_pt: float, h_pt: float, dpi: int) -> float:
    return (w_pt * dpi / 72) * (h_pt * dpi / 72)


def within_raster_caps(w_pt: float, h_pt: float, lim: dict) -> bool:
    """Python mirror of RASTER_CAP_AWK (page count checked separately); zero sizes are handled apart."""
    return (w_pt <= lim["dim_pt"] and h_pt <= lim["dim_pt"]
            and raster_pixels(w_pt, h_pt, lim["dpi"]) <= lim["pix"])


def container_args(image_id: str, lim: dict, name: str, command: list[str]) -> list[str]:
    tmpfs = 2 * lim["out"] + lim["log"] + TMPFS_SLACK
    return ["run", "-i", "--name", name, "--network", "none", "--read-only",
            "--tmpfs", f"/job:rw,nosuid,nodev,noexec,size={tmpfs},mode=1777",
            "--user", f"{RUNTIME_UID}:{RUNTIME_UID}", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", str(lim["pids"]),
            "--memory", str(lim["mem"]), "--memory-swap", str(lim["mem"]), "--cpus", "1",
            "-e", "SOURCE_DATE_EPOCH=0", "-e", "FORCE_SOURCE_DATE=1",
            image_id, *command]


def run_container(tex: bytes, th) -> RunArtifacts:
    image_id = ensure_image()
    lim = _limits(th)
    dim_pt = lim["dim_pt"]
    cmd = ["/opt/hmb/run-job.sh", str(lim["cpu"]), str(lim["out"]), str(lim["page_cap"]),
           f"{dim_pt:.6f}", str(lim["pix"]), str(lim["rcpu"]), str(lim["dpi"]), str(lim["out"])]
    name = "hmb-" + uuid.uuid4().hex[:16]
    t0 = time.monotonic()
    exe = shutil.which("docker")
    if exe is None:
        raise SandboxUnavailable("docker executable not found")
    proc = subprocess.Popen([exe, *container_args(image_id, lim, name, cmd)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wall_killed = False
    try:
        out, err = proc.communicate(input=tex, timeout=lim["wall"])
    except subprocess.TimeoutExpired:
        wall_killed = True
        _docker("kill", name, timeout=60)
        out, err = proc.communicate(timeout=60)
    wall = time.monotonic() - t0
    ins = _docker("inspect", "--format", "{{.State.OOMKilled}} {{.State.ExitCode}}", name, timeout=60)
    _docker("rm", "-f", name, timeout=60)
    if ins.returncode != 0:
        raise SandboxHarnessError("container vanished before inspection: " + err.decode(errors="replace")[:500])
    oom_s, code_s = ins.stdout.decode().split()
    art = RunArtifacts(wall_killed, oom_s == "true", int(code_s), wall)
    if not wall_killed and out:
        try:
            with tarfile.open(fileobj=io.BytesIO(out), mode="r:") as tf:
                for m in tf.getmembers():
                    if m.isfile():
                        f = tf.extractfile(m)
                        art.files[m.name.lstrip("./")] = f.read() if f else b""
        except tarfile.TarError:
            if not art.oom_killed:
                raise SandboxHarnessError("runner produced no readable tar: " + err.decode(errors="replace")[:500])
    elif not wall_killed and not art.oom_killed and art.container_exit not in (0, None):
        raise SandboxHarnessError(f"runner exit {art.container_exit}: " + err.decode(errors="replace")[:500])
    return art


# ----------------------------------------------------------------------------- classification
@dataclass
class CompileResult:
    outcome: Outcome
    image_id: str | None = None
    exit_code: int | None = None
    signal: int | None = None
    cpu_seconds: float | None = None
    max_rss_kb: int | None = None
    wall_seconds: float | None = None
    wallclock_reruns: int = 0
    page_count: int | None = None
    page_size_cm: list[float] | None = None
    pdf_path: str | None = None
    pdf_sha256: str | None = None
    log_path: str | None = None
    log_sha256: str | None = None          # of the full log as produced
    log_bytes: int | None = None
    log_truncated: bool = False
    render_path: str | None = None
    render_sha256: str | None = None
    canon: dict | None = None
    ink_fraction: float | None = None
    image_xobjects: int | None = None
    path_op_count: int | None = None
    violations: list[str] = field(default_factory=list)
    missing_glyphs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


def parse_time(txt: str) -> tuple[int | None, int | None, float | None, int | None]:
    """GNU time output -> (exit, signal, cpu_seconds, max_rss_kb)."""
    sig = None
    m = re.search(r"Command terminated by signal (\d+)", txt)
    if m:
        sig = int(m.group(1))
    last = [ln for ln in txt.strip().splitlines() if re.match(r"^[\d.]+ [\d.]+ \d+ [\d.:]+ -?\d+$", ln.strip())]
    if not last:
        return None, sig, None, None
    u, s_, rss, _e, x = last[-1].split()
    return int(x), sig, round(float(u) + float(s_), 3), int(rss)


def find_violations(log: str) -> list[str]:
    out = []
    for key, rx in VIOLATION_PATTERNS.items():
        if key == "S4_read_outside":
            for m in rx.finditer(log):
                p = m.group(1)
                if not p.startswith(_ALLOWED_READ_ROOTS):
                    out.append(f"{key}:{p}")
        elif rx.search(log):
            out.append(key)
    return sorted(set(out))


def _pdfinfo(txt: str) -> tuple[int | None, tuple[float, float] | None]:
    """(pages, (w_pt, h_pt)) from `pdfinfo -box`; raw CropBox points, as run-job.sh reads them."""
    pages = re.search(r"^Pages:\s+(\d+)", txt, re.M)
    box = re.search(r"^CropBox:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", txt, re.M)
    size = None
    if box:
        x0, y0, x1, y1 = map(float, box.groups())
        size = (x1 - x0, y1 - y0)
    return (int(pages.group(1)) if pages else None), size


def pdf_raster_and_paths(pdf: bytes) -> tuple[int, int]:
    """(# image XObjects / inline images / PostScript XObjects, path-operation count). Host PyMuPDF."""
    import fitz
    n_img, n_ops = 0, 0
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        for x in range(1, doc.xref_length()):
            try:
                t, v = doc.xref_get_key(x, "Subtype")
            except Exception:
                continue
            if t == "name" and v in ("/Image", "/PS"):
                n_img += 1
        for page in doc:
            n_img += sum(1 for i in page.get_image_info() if i.get("xref", 0) == 0)   # inline images
            n_ops += sum(len(d["items"]) for d in page.get_drawings())
    return n_img, n_ops


def classify(art: RunArtifacts, th, workdir: Path) -> CompileResult:
    lim = _limits(th)
    res = CompileResult(Outcome.C_ERROR, wall_seconds=round(art.wall_seconds, 3))
    if art.wall_killed:
        res.outcome = Outcome.C_WALLCLOCK
        return res
    log_full = art.files.get("out/cand.log", b"")
    term = art.text("meta/stdout.txt") + "\n" + art.text("meta/stderr.txt")
    log_txt = log_full.decode("utf-8", errors="replace")
    workdir.mkdir(parents=True, exist_ok=True)
    if log_full:
        kept = log_full[:lim["log"]]
        res.log_truncated = len(log_full) > lim["log"]
        (workdir / "cand.log").write_bytes(kept)
        res.log_path, res.log_sha256, res.log_bytes = str(workdir / "cand.log"), sha256_bytes(log_full), len(log_full)
    res.exit_code, res.signal, res.cpu_seconds, res.max_rss_kb = parse_time(art.text("meta/time.txt"))
    res.violations = find_violations(log_txt + "\n" + term)
    glyph_txt = log_txt + "\n" + term          # the log AND the terminal output (red-team #8)
    res.missing_glyphs = sorted({f"{m.group(2)} {m.group(3).strip()}" for m in MISSING_GLYPH.finditer(glyph_txt)})
    missing = bool(MISSING_GLYPH_ANY.search(glyph_txt))
    too_big = any(len(v) > lim["out"] for k, v in art.files.items() if k.startswith("out/"))
    cpu_hit = res.signal == SIGXCPU or (res.signal == SIGKILL and (res.cpu_seconds or 0) >= lim["cpu"] - 0.5)
    if res.violations:
        res.outcome = Outcome.C_SANDBOX_VIOLATION
        return res
    if cpu_hit:
        res.outcome = Outcome.C_TIMEOUT
        return res
    if (art.oom_killed or res.signal in (SIGXFSZ, SIGKILL) or too_big or res.log_truncated
            or (res.exit_code not in (0, None) and RESOURCE_LOG.search(log_txt + term))):
        res.outcome = Outcome.C_RESOURCE
        res.notes.append("oom" if art.oom_killed else "fsize/log/capacity")
        return res
    if missing:
        res.outcome = Outcome.C_MISSING_GLYPH
        return res
    if res.exit_code != 0 or res.signal is not None:
        res.outcome = Outcome.C_ERROR
        return res
    pdf = art.files.get("out/cand.pdf")
    if not pdf:
        if "No pages of output" in log_txt + term:
            res.page_count = 0
            res.outcome = Outcome.V_MULTIPAGE
        else:
            res.outcome = Outcome.C_ERROR
        return res
    (workdir / "cand.pdf").write_bytes(pdf)
    res.pdf_path, res.pdf_sha256 = str(workdir / "cand.pdf"), sha256_bytes(pdf)
    res.page_count, size_pt = _pdfinfo(art.text("meta/pdfinfo.txt"))
    if res.page_count is None:                 # pdfinfo could not read the PDF (red-team #10)
        res.outcome = Outcome.C_ERROR
        res.notes.append("pdfinfo failed: " + art.text("meta/pdfinfo.txt")[:300])
        return res
    if size_pt is not None:
        res.page_size_cm = [round(size_pt[0] / PT_PER_CM, 4), round(size_pt[1] / PT_PER_CM, 4)]
    res.image_xobjects, res.path_op_count = pdf_raster_and_paths(pdf)
    if res.page_count != 1:
        res.outcome = Outcome.C_RESOURCE if (res.page_count or 0) > lim["page_cap"] else Outcome.V_MULTIPAGE
        return res
    if res.image_xobjects:
        res.outcome = Outcome.V_RASTER
        return res
    w, h = size_pt if size_pt is not None else (0.0, 0.0)
    if w <= 0 or h <= 0:                       # zero-size CropBox: nothing to render (red-team #6)
        res.ink_fraction = 0.0
        res.outcome = Outcome.V_EMPTY
        res.notes.append("zero-size CropBox")
        return res
    if not within_raster_caps(w, h, lim):
        res.outcome = Outcome.V_OVERSIZE
        return res
    raw_png = art.files.get("out/render.png")
    if art.text("meta/raster_exit").strip() != "0" or not raw_png:
        res.outcome = Outcome.C_RESOURCE
        res.notes.append("rasterizer failed: " + art.text("meta/raster.txt")[:300])
        return res
    png, meta = canonicalize_image(raw_png, th)
    (workdir / "render.png").write_bytes(png)
    res.render_path, res.render_sha256, res.canon = str(workdir / "render.png"), meta["render_sha256"], meta
    res.ink_fraction = round(ink_fraction(png), 8)
    if res.ink_fraction < float(th.value("eps_ink")):
        res.outcome = Outcome.V_EMPTY
        return res
    res.outcome = Outcome.OK
    return res


# C24: CPU time is load-dependent in the Docker Desktop VM, so scoring compiles run one at a time even
# when the hybrid scheduler (hmb.batch.scheduler) runs cells concurrently. A serial run is unaffected.
_COMPILE_LOCK = threading.Lock()


def compile_in_sandbox(tex: bytes, th, workdir: Path) -> CompileResult:
    """§9.6 + §9.5 post-compile checks. Raises SandboxUnavailable; never runs TeX on the host."""
    workdir = Path(workdir)
    if len(tex) > int(th.value("output_bytes")):
        return CompileResult(Outcome.C_RESOURCE, notes=["input larger than output_bytes"])
    with _COMPILE_LOCK:
        image_id = ensure_image()
        art = run_container(tex, th)
        reruns = 0
        if art.wall_killed:            # C_WALLCLOCK is a harness incident: re-run once (sequentially)
            reruns = 1
            art = run_container(tex, th)
        workdir.mkdir(parents=True, exist_ok=True)
        res = classify(art, th, workdir)
    res.image_id, res.wallclock_reruns = image_id, reruns
    (workdir / "compile_meta.json").write_text(json.dumps(res.to_json(), indent=1, sort_keys=True),
                                               encoding="utf-8")
    return res


# ----------------------------------------------------------------------------- determinism, gates
def determinism_sample(submission_ids: Iterable[str], th, seed: str = "hmb-determinism-v1") -> list[str]:
    """Fixed 5 percent (determinism_sample_frac) sample, chosen by hash rank (reproducible)."""
    ids = sorted(set(submission_ids))
    k = math.ceil(float(th.value("determinism_sample_frac")) * len(ids)) if ids else 0
    return sorted(sorted(ids, key=lambda i: sha256_bytes(f"{seed}:{i}".encode()))[:k])


def harness_determinism_rerun(tex: bytes, th, workdir: Path, runs: int = 2) -> dict:
    """HarnessDeterminism@rerun: recompile + re-render `runs` times; identical render_sha256 required."""
    rs = [compile_in_sandbox(tex, th, Path(workdir) / f"rerun{i + 1}") for i in range(runs)]
    renders = [r.render_sha256 for r in rs]
    pdfs = [r.pdf_sha256 for r in rs]
    return {"outcomes": [r.outcome.value for r in rs], "render_sha256": renders, "pdf_sha256": pdfs,
            "identical_render": len(set(renders)) == 1 and renders[0] is not None,
            "identical_pdf": len(set(pdfs)) == 1 and pdfs[0] is not None}


_TEX_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
            "_": r"\_", "%": r"\%", "^": r"\^{}", "~": r"\~{}"}


def tex_escape(s: str) -> str:
    return "".join(_TEX_ESC.get(c, c) for c in s)


def labels_from_cards(cards) -> list[str]:
    """Every label string of every eligible card fact + canonical entity label (§9.6 font gate)."""
    out: set[str] = set()
    for c in cards:
        for e in c.entities:
            if e.label:
                out.add(e.label)
        for f in c.facts:
            if f.predicate in ("label", "measure_label") and f.eligible:
                out.add(f.args[0])
    return sorted(out)


def font_coverage_gate(labels: Iterable[str], th, workdir: Path, chunk: int = 400) -> dict:
    """§9.6 one-time font-coverage gate: typeset every label with the wrapper_v1 font block
    (\\tracinglostchars=2 so every lost glyph is logged, not only the first) and require zero."""
    from .wrapper import harness_font_block
    labs = sorted(set(labels))
    missing: list[dict] = []
    runs = []
    for k in range(0, len(labs), chunk):
        part = labs[k:k + chunk]
        body = "\n".join(f"\\node[anchor=west] at (0,{-0.6 * i:.2f}) {{{tex_escape(t)}}};"
                         for i, t in enumerate(part))
        doc = ("\\documentclass[tikz,border=2mm]{standalone}\n\\usepackage{amsmath,amssymb}\n"
               + harness_font_block() + "\\tracinglostchars=2\n\\begin{document}\n"
               "\\begin{tikzpicture}\n" + body + "\n\\end{tikzpicture}\n\\end{document}\n")
        art = run_container(doc.encode("utf-8"), th)
        log = art.text("out/cand.log")
        ok_pdf = bool(art.files.get("out/cand.pdf"))
        runs.append({"n": len(part), "pdf": ok_pdf, "exit": art.text("meta/exit").strip()})
        for m in MISSING_GLYPH.finditer(log):
            cp = m.group(2)
            ch = chr(int(cp[2:], 16))
            missing.append({"codepoint": cp, "char": ch, "font": m.group(3).strip(),
                            "labels": [t for t in part if ch in t][:5]})
    Path(workdir).mkdir(parents=True, exist_ok=True)
    rep = {"n_labels": len(labs), "missing": missing, "runs": runs,
           "pass": not missing and all(r["pdf"] for r in runs),
           "labels_sha256": sha256_json(labs)}
    (Path(workdir) / "font_coverage_gate.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    return rep


def image_selftest(th) -> dict:
    """Run /opt/hmb/selftest.sh under the production container flags."""
    art_cmd = ["/bin/bash", "-c", "HMB_SELFTEST_RUNTIME=1 HMB_SELFTEST_DIR=/job/selftest /opt/hmb/selftest.sh"]
    image_id = ensure_image()
    lim = _limits(th)
    name = "hmb-selftest-" + uuid.uuid4().hex[:8]
    r = _docker(*container_args(image_id, lim, name, art_cmd), input=b"", timeout=300)
    _docker("rm", "-f", name, timeout=60)
    out = r.stdout.decode(errors="replace")
    return {"exit": r.returncode, "output": out, "pass": r.returncode == 0 and "SELFTEST ALL PASS" in out}
