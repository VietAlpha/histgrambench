| Variant | H: Gemma 4 26B | H: Llama 4 Maverick | H: Ministral 3 14B | H: Gemma 4 E2B | H: Claude Haiku 4.5 | H: Nemotron Nano 12B v2 | Bands | Separated | Top pair p_adj | τ_b |
|---|---|---|---|---|---|---|---|---|---|---|
| Headline | 0.589 | 0.524 | 0.391 | 0.308 | 0.300 | 0.197 | 5 | 14 | 0.089 | 1.00 |
| Geometry-confirmed | 0.392 | 0.387 | 0.221 | 0.190 | 0.205 | 0.147 | 3 | 11 | 0.838 | 0.87 |
| Forced as absent | 0.571 | 0.524 | 0.355 | 0.298 | 0.299 | 0.197 | 3 | 11 | 0.220 | 0.87 |
| Forced as present | 0.607 | 0.524 | 0.406 | 0.360 | 0.300 | 0.197 | 6 | 13 | 0.035 | 1.00 |
| Fixed denominator | 0.553 | 0.487 | 0.355 | 0.263 | 0.276 | 0.165 | 5 | 14 | 0.081 | 0.87 |
| Without Google | 0.571 | 0.518 | 0.391 | 0.298 | 0.295 | 0.193 | 4 | 13 | 0.156 | 1.00 |
| Without Mistral | 0.589 | 0.500 | 0.355 | 0.308 | 0.282 | 0.185 | 6 | 13 | 0.022 | 1.00 |
| Without Qwen | 0.589 | 0.507 | 0.371 | 0.349 | 0.287 | 0.189 | 6 | 13 | 0.038 | 1.00 |
| Without largest group | 0.589 | 0.524 | 0.390 | 0.307 | 0.298 | 0.196 | 5 | 14 | 0.094 | 1.00 |
| Item-weighted | 0.571 | 0.489 | 0.460 | 0.379 | 0.361 | 0.295 | — | — | — | — |
| Without Numdam | 0.526 | 0.471 | 0.441 | 0.378 | 0.325 | 0.289 | — | — | — | — |
| Benjamini–Yekutieli | 0.589 | 0.524 | 0.391 | 0.308 | 0.300 | 0.197 | — | 13 | 0.295 | — |

Stability of the Track A comparison under scoring, weighting and multiplicity variants. H per model; bands: distinct rank bands; separated: pairs of 15 separated at BH q = 0.10; top pair: adjusted p for Gemma 4 26B against Llama 4 Maverick; τ_b: Kendall's τ_b against the headline order. Item-weighted and without-Numdam rows report point order only; the Benjamini–Yekutieli row adjusts the headline p values.
