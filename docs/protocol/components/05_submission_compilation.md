# Component 05 — Contestant protocol, TikZ extraction, sandboxed compilation, and render canonicalization

Part of *HistMathBench Zero-Annotation Project Plan v2*. Canonical entry point: [`../HistMathBench_Zero_Annotation_Project_Plan_v2.md`](../HistMathBench_Zero_Annotation_Project_Plan_v2.md). This file owns **Section 9**. Global notation is in the top-level plan, Section 3.4.

| Component card | |
|---|---|
| Purpose | Turn a model response into exactly one deterministic outcome code and, when valid, one canonical render and one compiled PDF that the adjudicator can analyse. |
| Inputs | Frozen prompt per track; canonical crop (and context for Track B); raw model response with stop reason. |
| Mechanism | Fixed extraction algorithm; three wrapper classes with a preamble allowlist grammar; pinned LuaLaTeX sandbox; post-compile artifact checks; pinned rasterization. |
| Outputs | Outcome code, normalized `.tex`, PDF, canonical PNG, hashes, compile log. |
| Gates | Outcome code `OK` is the only path to adjudication. |
| Retained artifacts | Raw response, extracted span, normalized tex, log, PDF, PNG, all hashed. |
| Cost tier | CPU only, seconds per item. |
| Lifecycle | Per item, per model, per track; harness built once. |
| Human effort | None per item. |
| Dependencies | Section 6.4 (crop canonicalization shares this pipeline), Section 7 (label inventory for the font-coverage gate), Section 10 (consumes PDF and PNG), Section 11 (consumes outcome codes), Section 15 (pins in the release manifest). |
| Downstream decisions | `Valid@1`, compile gate, everything judged. |

> **Checkpoint simplification 2026-09-26.** The Track C control suite (CP3) and evaluator validation (CP5) were
> deleted. No mechanism of Section 9 changes. The same harness still compiles the generated known-geometry fixtures
> of the offline tests, including the extractor round trip `hmb cgx-roundtrip` (component 08, §12); "Track C" survives
> only as the harness label of those fixture renders and is never a benchmark track.

---

## 9. Contestant protocol, extraction, compilation, rendering

### 9.1 Prompt specification

One frozen prompt per track, published with `prompt_id` and `prompt_sha256`. Each prompt states verbatim:

1. **Task.** Reconstruct the mathematical figure the original author intended; do not reproduce scan damage, paper texture, or print defects.
2. **Output contract.** Exactly one fenced block tagged ```` ```latex ````, containing exactly one `tikzpicture` (optionally inside a `standalone` document). Any reasoning goes outside the block. Nothing after the block.
3. **Allowlist.** The package and TikZ-library allowlist of Section 9.3, verbatim.
4. **Prohibitions.** Shell escape, `\write18`, `\directlua`, `\input`, `\include`, `\openin`, `\openout`, `filecontents`, `\includegraphics`, `\pdfximage`, `\special`, font-selection macros, external files, network access.
5. **Not judged.** Absolute coordinates, scale, line width, font, colour, small label offsets (the protocol constant `allowed_variation`, Section 7.6).
6. **Script.** Labels may be written literally in their original script (Chinese, Japanese, Vietnamese, Cyrillic, Greek, Latin); the compiler supplies covering fonts.

Track B appends the context text (Section 6.4) after the image and states that it is surrounding text from the same page and may contain the figure's caption.

Decoding: temperature 0 (or the provider's lowest deterministic setting), `top_p` 1, a `max_tokens` value identical across models where the API allows and otherwise the provider maximum, recorded per model. Retries are permitted only for transport errors, HTTP 5xx, and rate limits, at most three, every attempt logged. A retry is never permitted for a parse or compile failure. One output per item per track.

### 9.2 Extraction algorithm (deterministic)

```text
input : raw assistant text R; provider "thinking" segments are excluded and stored separately
output: candidate span S, or an outcome code

1. if R is empty or stop_reason == "refusal"                      -> E_NO_OUTPUT
2. B  := fenced blocks with info string in {latex, tex, tikz, ""}
   Bq := blocks in B containing "\begin{tikzpicture}" or "\documentclass"
3. if |Bq| >= 1: S := LAST element of Bq; path := fenced; n_blocks := |Bq|
   else if R contains "\documentclass":
        S := first "\documentclass" .. last "\end{document}"; path := bare_doc
   else if R contains "\begin{tikzpicture}":
        S := first "\begin{tikzpicture}" .. its BALANCED "\end{tikzpicture}"; path := bare_pic
   else -> E_TRUNCATED if stop_reason is a truncation (Converse "max_tokens", Mantle "length"; 2026-09-25) else E_NO_TIKZ
4. if top-level tikzpicture environments in S > 1                  -> E_MULTI_CANDIDATE
5. if S has an unbalanced tikzpicture environment
        -> E_TRUNCATED if stop_reason is a truncation else E_NO_TIKZ   (never auto-close)
6. return S
```

"Last block" is a choice, not a truth. The rate of each extraction path and `n_blocks` are published, and a one-off first-versus-last sensitivity re-extraction is run on the pilot (no new model or judge calls) and reported if any ranking changes.

### 9.3 Wrapper normalization: three classes and an allowlist grammar

- **Class D (full document).** `S` contains `\documentclass`. Compiled as submitted after the preamble grammar check. The prompt asks for `standalone`; other classes are allowed but will usually produce `V_OVERSIZE` or margins.
- **Class P (picture plus preamble lines).** `S` is one `tikzpicture` optionally preceded by preamble lines. Lines are copied verbatim, in order, into the fixed wrapper's preamble if and only if every line matches the grammar: `\usepackage[opts]{X}` with `X ∈ PKG_ALLOW`; `\usetikzlibrary{L,…}` with each `L ∈ LIB_ALLOW`; `\usepgfplotslibrary{…}` from the allowed set; `\pgfplotsset{…}`; `\tikzset{…}`; `\definecolor{…}`; `\newcommand`, `\renewcommand`, `\def`, `\let`; `\pgfdeclare*`. Anything else is `P_POLICY_REJECT` naming the token.
- **Class B (bare body).** A sequence of `\draw`/`\node`/… with no environment. Wrapped in `tikzpicture`. Allowed but counted; the prompt forbids it.

Fixed wrapper `wrapper_v1` (published, hashed):

```latex
\documentclass[tikz,border=2mm]{standalone}
\usepackage{amsmath,amssymb}
% fontspec with a bundled fallback chain: Latin+Vietnamese, Greek, Cyrillic, CJK (simplified and traditional), kana
\tracinglostchars=3
\usetikzlibrary{<LIB_BASE>}
<contestant preamble lines, verbatim, grammar-checked>
\begin{document}
<candidate span, byte-identical>
\end{document}
```

`PKG_ALLOW` (initial): `tikz, pgfplots, amsmath, amssymb, xcolor, tikz-3dplot`. `LIB_ALLOW` (initial): `calc, intersections, angles, quotes, positioning, arrows.meta, decorations.markings, decorations.pathmorphing, patterns, through, shapes.geometric, shapes.misc, 3d, perspective, math, fit, backgrounds, babel`. Explicitly not allowed: `external`, gnuplot-backed plotting, `datavisualization`, anything requiring shell escape, any font package. The allowlist must cover the idiomatic libraries of every diagram family in Section 2.6; `Valid@1` is reported by family as the check that no family is penalized by the allowlist.

The static checker is a **preamble allowlist grammar only**. It is not a security control: TeX is macro-expansion-complete and any denylist of tricks can be defeated. Security comes from the engine flags, the container, and post-compile artifact checks.

### 9.4 The no-hidden-repair invariant

> `sha256` of the byte span from `\begin{tikzpicture}` through its matching `\end{tikzpicture}` in `normalized.tex` **equals** the same hash computed on the extracted span in the raw response.

Normalization may add bytes before `\begin{document}` or outside the picture span and may delete fence delimiters and text outside the span. It may never rewrite a byte inside the span. The invariant is asserted at runtime; a violation is a harness bug that fails the run, never an item outcome. `body_sha256` is stored per submission.

### 9.5 Outcome codes (mutually exclusive, exhaustive)

| # | Code | Stage | Gate | Counted in |
|---|---|---|---|---|
| 1 | `E_NO_OUTPUT` | pre-compile | 0 | Extraction-failure@1 |
| 2 | `E_TRUNCATED` | pre-compile | 0 | Extraction-failure@1 (also published alone) |
| 3 | `E_NO_TIKZ` | pre-compile | 0 | Extraction-failure@1 |
| 4 | `E_MULTI_CANDIDATE` | pre-compile | 0 | Protocol-violation@1 |
| 5 | `P_POLICY_REJECT` | pre-compile | 0 | Unsafe/Invalid@1 |
| 6 | `C_ERROR` | compile | 0 | complement of Compile@1 |
| 7 | `C_TIMEOUT` (CPU limit) | compile | 0 | Timeout@1 |
| 8 | `C_WALLCLOCK` (liveness kill) | compile | 0 | harness incident; re-run once on an idle worker |
| 9 | `C_RESOURCE` (memory, log, file size, page cap) | compile | 0 | published alone |
| 10 | `C_MISSING_GLYPH` | compile | 0 | harness-quality indicator |
| 11 | `C_SANDBOX_VIOLATION` | compile | 0 | Unsafe/Invalid@1 |
| 12 | `V_MULTIPAGE` (pages ≠ 1) | post-compile | 0 | Protocol-violation@1 |
| 13 | `V_RASTER` (image XObject or embedded PostScript present) | post-compile | 0 | Unsafe/Invalid@1 |
| 14 | `V_EMPTY` (ink fraction < ε_ink) | post-compile | 0 | published alone |
| 15 | `V_OVERSIZE` (page dimension or pixel cap) | post-compile | 0 | published alone |
| 16 | `OK` | — | 1 | Valid@1 |

`OK` requires all of: exit code 0, a PDF exists, page count = 1, no image XObject, ink fraction ≥ ε_ink, dimensions within cap, no "Missing character" error. Exit code alone is not "compiles". Only code 16 reaches adjudication; codes 1–15 short-circuit it.

Derived rates over the same denominator n (eligible items):

```text
Valid@1              = #{OK} / n
Compile@1            = #{OK, V_*} / n
Unsafe/Invalid@1     = #{P_POLICY_REJECT, C_SANDBOX_VIOLATION, V_RASTER} / n
Timeout@1            = #{C_TIMEOUT} / n
Extraction-failure@1 = #{E_NO_OUTPUT, E_TRUNCATED, E_NO_TIKZ} / n
Protocol-violation@1 = #{E_MULTI_CANDIDATE, V_MULTIPAGE} / n
```

The full code distribution is published per model and track. ε_ink is preregistered from the engineering set (starting value 0.001 of the canonical canvas).

### 9.6 Sandbox and pins

Two verified facts shape this section. In TeX Live's shipped `texmf.cnf`, `shell_escape = p` (restricted; a list of external programs is permitted) and `openin_any` is documented as having no effect as of 2026, so the TeX layer provides **no read confinement** ([texmf.cnf, TeX Live source](https://raw.githubusercontent.com/TeX-Live/texlive-source/trunk/texk/kpathsea/texmf.cnf)). Read confinement is therefore a container property.

- **Image.** A pinned base image digest; TeX Live installed from a frozen year snapshot with the `texlive.tlpdb` hash recorded; the harness image referenced everywhere by `sandbox_image_digest`.
- **Engine, one for all items and tracks.** `lualatex --safer --nosocket --no-shell-escape --interaction=nonstopmode --halt-on-error --output-directory=/job/out --jobname=cand /job/cand.tex`. In the image's `texmf.cnf`: `shell_escape = f`, `openout_any = p`. Both asserted by an image self-test.
- **Fonts.** Bundled in the image with a fallback chain covering Latin with Vietnamese diacritics, Greek, Cyrillic, simplified and traditional Chinese, and kana. `\tracinglostchars=3` makes a missing glyph a coded failure (`C_MISSING_GLYPH`) instead of a silent blank. **Font-coverage gate (one time, before any official run):** render every label string in every eligible card (Section 7) and require zero missing glyphs; an uncovered script is a harness defect fixed by adding a font.
- **Determinism.** `SOURCE_DATE_EPOCH=0`, `FORCE_SOURCE_DATE=1`, prebuilt font cache baked into the image, single fixed CPU architecture. `HarnessDeterminism@rerun`: a fixed 5 percent random sample of submissions is recompiled and re-rendered twice in the pinned image and must give byte-identical `render_sha256`; the figure is published.
- **Container.** No network namespace, read-only root filesystem, a `tmpfs` job directory containing only `cand.tex`, non-root uid, all capabilities dropped, a process limit, default seccomp, and **no crop, card, or other item mounted anywhere in the namespace**.
- **Limits** (initial, calibrated on the engineering set, then frozen): CPU 20 s (`C_TIMEOUT`); wall-clock kill 60 s (`C_WALLCLOCK`); memory 1 GiB, output file 32 MiB, log 4 MiB (`C_RESOURCE`); PDF page cap 4 (`V_MULTIPAGE` above 1, `C_RESOURCE` above 4); page dimension cap 200 cm (`V_OVERSIZE`). The CPU-second limit is the scoring limit so that `Timeout@1` does not depend on host load.
- **Logs** retained and hashed.

A residual risk that cannot be closed is a "raster" drawn as tens of thousands of tiny filled rectangles; the path-operation count is logged as telemetry and the limitation is disclosed.

### 9.7 Render canonicalization (applied identically to candidate renders and to the historical crop the judge sees)

```text
pdftoppm -png -r 300 -aa yes -aaVector yes -cropbox -singlefile cand.pdf out
1. composite onto opaque white
2. s := min(1024 / max(W, H), 4.0)          # never upscale more than 4x
3. resample with a pinned filter (Lanczos, pinned library version)
4. letterbox centred on a 1024 x 1024 white canvas; record s, W, H
5. write PNG with fixed compression and no time or metadata chunks
6. record render_sha256
```

The historical crop supplied to the judge goes through steps 1–6 of the same code, so both judge inputs come from one pipeline. Anti-aliasing stays on; determinism comes from the pinned image, not from disabling it. No rotation, deskew, or geometric repair of either image.

### 9.8 Text extractability for the adjudicator

Section 10 extracts labels from the compiled PDF. LuaLaTeX with Unicode fonts produces text with ToUnicode maps, which is why the engine choice is not free: the Phase 0 smoke test (Section 17) must confirm that glyph strings and per-glyph boxes extract correctly for Latin, Greek, Cyrillic, and CJK labels.

### 9.9 Rejected alternatives

Auto-repair of near-miss TikZ (hidden repair); an engine cascade (render depends on which engine succeeded); per-item engine or font selection; compile-repair loops or best-of-k; disabling anti-aliasing for bit-exactness; a full TeX-construct denylist as a security control; one paraphrase prompt variant on a subsample (a sensitivity number that does not gate the pilot decision; deferred from V1).

### 9.10 Unverified items to settle in the Phase 0 smoke test

No TeX distribution was available in the planning environment, so nothing here was executed. To confirm: that `--safer` is compatible with the bundled font setup when the cache is read-only; that LuaTeX honours `SOURCE_DATE_EPOCH` the way pdfTeX documents; glyph coverage of the chosen CJK faces for 1920s traditional forms; and the numeric limits above.
