# Third-party notices

This repository's own code, schemas, prompts and documents are licensed under Apache-2.0 (`LICENSE`). It contains or
depends on the third-party material below. Nothing in this list is bundled unless marked **included**.

## Included in this repository

| Component | Where | Licence |
|---|---|---|
| Source Sans 3 (Adobe) | `research-paper/graphics/fonts/` | SIL Open Font License 1.1 (`research-paper/graphics/fonts/LICENSE.md`) |
| Figure crops and page text from the source works: two figures of one work (Ramus, *The Way to Geometry*, Project Gutenberg) for the example, and the fourteen sample crops of Figure 1 | `examples/sample_root/images/`, `research-paper/media/*.png` | reproductions of the source works; cite each work as listed in the dataset `works` table (see the dataset `LICENSE.md`) |
| Model outputs (answers, geometry, verdicts, scores) of the six contestants and three judges | `examples/sample_root/`, `results/` | generated through Amazon Bedrock under the providers' terms; see the dataset `LICENSE.md` |

## Python dependencies (installed from `requirements.lock`, not included)

Licences as declared in each package's metadata in the pinned environment. **PyMuPDF is licensed under the GNU AGPL
3.0 (or a commercial licence from Artifex).** The evaluation code imports it to read compiled PDFs (`hmb/cgx`) and to
render pages; distributing the source here does not include PyMuPDF, but anyone who redistributes a combined
environment or application must meet the AGPL's terms. The `anthropic` and `google-genai` packages are pinned in the
lock but not imported by the code (their connectors were removed before the run).

| Package | Licence (package metadata) |
|---|---|
| annotated-types 0.8.0 | MIT |
| anthropic 1.8.0 | MIT License |
| anyio 4.15.1 | MIT |
| attrs 26.1.0 | MIT |
| boto3 1.43.100 | Apache-2.0 |
| botocore 1.43.100 | Apache-2.0 |
| certifi 2026.7.22 | Mozilla Public License 2.0 (MPL 2.0) |
| cffi 2.1.1 | MIT-0 |
| charset-normalizer 3.5.1 | MIT |
| colorama 0.4.6 | BSD License |
| contourpy 1.4.0 | BSD-3-Clause |
| cryptography 50.0.1 | Apache-2.0 OR BSD-3-Clause |
| cycler 0.12.1 | BSD License |
| distro 1.9.0 | Apache Software License |
| docstring-parser 0.18.0 | MIT License |
| fonttools 4.66.1 | MIT |
| google-auth 2.58.0 | Apache Software License |
| google-genai 2.25.0 | Apache-2.0 |
| h11 0.16.0 | MIT License |
| httpcore 1.0.9 | BSD-3-Clause |
| httpcore2 2.13.0 | BSD-3-Clause |
| httpx 0.28.1 | BSD License |
| httpx2 2.13.0 | BSD-3-Clause |
| idna 3.20 | BSD-3-Clause |
| imageio 2.37.4 | BSD-2-Clause |
| iniconfig 2.3.0 | MIT |
| jiter 0.17.0 | MIT |
| jmespath 1.1.0 | MIT License |
| jsonschema 4.25.1 | MIT |
| jsonschema-specifications 2025.9.1 | MIT |
| kiwisolver 1.5.1 | BSD License |
| lazy-loader 0.6 | BSD-3-Clause |
| matplotlib 3.10.7 | Python Software Foundation License |
| networkx 3.4.2 | BSD License |
| numpy 2.2.6 | BSD License |
| openai 3.18.0 | Apache-2.0 |
| opencv-python-headless 4.11.0.86 | Apache Software License |
| packaging 26.3 | Apache-2.0 OR BSD-2-Clause |
| pillow 12.0.0 | MIT-CMU |
| pluggy 1.6.0 | MIT License |
| pyasn1 0.6.4 | BSD-2-Clause |
| pyasn1-modules 0.4.2 | BSD License |
| pycparser 3.0 | BSD-3-Clause |
| pydantic 2.12.5 | MIT |
| pydantic-core 2.41.5 | MIT |
| pygments 2.21.0 | BSD-2-Clause |
| pymupdf 1.26.5 | Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial Lic |
| pyparsing 3.3.3 | MIT |
| pytest 8.4.2 | MIT License |
| python-dateutil 2.9.0.post0 | BSD License; Apache Software License |
| pyyaml 6.0.3 | MIT License |
| referencing 0.37.0 | MIT |
| requests 2.34.2 | Apache Software License |
| rpds-py 2026.6.3 | MIT |
| s3transfer 0.19.2 | Apache Software License |
| scikit-image 0.25.2 | BSD License |
| scipy 1.15.3 | BSD License |
| six 1.17.0 | MIT License |
| sniffio 1.3.1 | MIT License; Apache Software License |
| tenacity 9.1.4 | Apache Software License |
| tifffile 2026.9.20 | BSD-3-Clause |
| truststore 0.10.4 | MIT |
| typing-extensions 4.16.0 | PSF-2.0 |
| typing-inspection 0.4.4 | MIT |
| urllib3 2.8.0 | MIT |
| websockets 16.1.1 | BSD-3-Clause |

## Sandbox image (built locally from `sandbox/Dockerfile`, not included)

The compile sandbox is built from `texlive/texlive:TL2025-historic` (pinned by digest in `sandbox/Dockerfile`) with
Debian packages from snapshot 20260824T000000Z: `poppler-utils`, `fonts-noto-cjk` and `time`. TeX Live, the Noto fonts
and the Debian packages carry their own licences (LPPL, GPL, SIL OFL and others); building the image fetches them
from their publishers. The image used for the paper run (`sha256:e902dd5a...`) is not redistributed.

## Models

The contestants (Gemma 4 26B A4B, Gemma 4 E2B, Llama 4 Maverick 17B, Ministral 3 14B, Nemotron Nano 12B v2 VL,
Claude Haiku 4.5), answer-key writers (GPT-6.1 Sol, GPT-6 Sol, Kimi K2.5), screener (GPT-6 Luna) and judges (Qwen3 VL
235B, Gemma 4 31B, Mistral Large 3) were used as hosted services through Amazon Bedrock. No model weights are
included. Their outputs in this release remain subject to the providers' terms.
