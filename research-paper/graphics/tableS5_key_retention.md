**(a)**

| Class | Union | Retained | Retention (%) | Both (%) |
|---|---|---|---|---|
| Entity | 54,496 | 45,627 | 83.7 | 42.0 |
| Relation | 91,803 | 50,817 | 55.4 | 13.5 |
| Label | 22,806 | 17,959 | 78.7 | 69.9 |

**(b)**

| Model | Entity: both | Entity: one | Relation: both | Relation: one | Label: both | Label: one |
|---|---|---|---|---|---|---|
| Gemma 4 26B | 0.88 | 0.80 | 0.88 | 0.73 | 0.92 | 0.80 |
| Llama 4 Maverick | 0.81 | 0.74 | 0.76 | 0.58 | 0.85 | 0.69 |
| Ministral 3 14B | 0.75 | 0.75 | 0.68 | 0.55 | 0.81 | 0.72 |
| Gemma 4 E2B | 0.67 | 0.65 | 0.71 | 0.52 | 0.81 | 0.62 |
| Claude Haiku 4.5 | 0.82 | 0.76 | 0.79 | 0.60 | 0.85 | 0.70 |
| Nemotron Nano 12B v2 | 0.63 | 0.61 | 0.62 | 0.45 | 0.79 | 0.57 |

**(c)**

| Predicate | Union | Kept (%) | Predicate | Union | Kept (%) |
|---|---|---|---|---|---|
| `endpoint_of` | 35,854 | 58 | `circle` | 1,194 | 72 |
| `label` | 21,949 | 79 | `center_of` | 1,160 | 56 |
| `segment` | 21,385 | 84 | `arc` | 1,110 | 84 |
| `on` | 18,822 | 59 | `measure_label` | 857 | 70 |
| `point` | 17,677 | 92 | `angle_marker` | 582 | 84 |
| `edge_of` | 9,200 | 64 | `region` | 436 | 88 |
| `intersect_at` | 5,358 | 43 | `ray` | 420 | 55 |
| `parallel` | 5,128 | 48 | `length_tick` | 344 | 72 |
| `line` | 3,844 | 80 | `arrow` | 312 | 71 |
| `polygon` | 2,912 | 68 | `asymptote_of` | 305 | 1 |
| `perpendicular` | 2,909 | 53 | `ellipse` | 277 | 78 |
| `between` | 2,716 | 62 | `solid` | 273 | 58 |
| `vertex_of` | 2,287 | 56 | `arc_of` | 225 | 38 |
| `collinear` | 2,159 | 42 | `face_of` | 209 | 4 |
| `tangent_to` | 2,068 | 40 | `equal_length` | 180 | 27 |
| `axis` | 1,942 | 87 | `right_angle_marked` | 122 | 2 |
| `curve` | 1,666 | 60 | `cross_section_of` | 82 | 10 |
| `hidden_edge` | 1,559 | 22 | `concyclic` | 47 | 49 |
| `contains` | 1,491 | 44 | `equal_angle` | 44 | 43 |

Answer-key retention under the unanimity rule (n = 114,403 retained facts). (a) Facts in the union of both writers and facts retained, by class; both: share of retained facts that both writers generated. (b) Share of checkable retained facts recovered on valid Track A drawings, by whether both writers or one generated the fact. (c) Retention by predicate.
