| Stage | Code | Condition | Track A | Track B |
|---|---|---|---|---|
| Extraction | `E_NO_OUTPUT` | Empty answer or refusal | 0 | 0 |
|  | `E_TRUNCATED` | Answer cut off at the length limit | 1,332 | 1,373 |
|  | `E_NO_TIKZ` | No TikZ picture found | 14 | 12 |
|  | `E_MULTI_CANDIDATE` | More than one picture | 2 | 2 |
| Policy | `P_POLICY_REJECT` | Disallowed package, library or command | 325 | 329 |
| Compile | `C_ERROR` | Compile error | 3,891 | 3,715 |
|  | `C_TIMEOUT` | CPU time limit exceeded | 13 | 15 |
|  | `C_WALLCLOCK` | Wall-clock kill | 0 | 0 |
|  | `C_RESOURCE` | Memory, log, file-size or page cap | 5 | 5 |
|  | `C_MISSING_GLYPH` | Missing glyph | 24 | 23 |
|  | `C_SANDBOX_VIOLATION` | Sandbox violation | 1 | 0 |
| Output | `V_MULTIPAGE` | More than one page | 0 | 1 |
|  | `V_RASTER` | Raster image or embedded PostScript | 4 | 1 |
|  | `V_EMPTY` | Blank drawing | 3 | 2 |
|  | `V_OVERSIZE` | Page larger than the size cap | 15 | 17 |
| Valid | `OK` | Valid drawing | 9,413 | 9,547 |

Outcome codes, in the order of the plan's outcome table (§9.5), with counts summed over the six models (15,042 answers per track). Only OK yields a scored drawing.
