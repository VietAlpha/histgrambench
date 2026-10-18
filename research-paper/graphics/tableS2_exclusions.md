| Reason | Code | Regions |
|---|---|---|
| *Image gates (3,256)* |  |  |
| Split into panels | `split_into_panels` | 703 |
| Region too small | `box_too_small` | 563 |
| Photographic | `photo_grayscale` | 548 |
| Extreme aspect ratio | `aspect_extreme` | 287 |
| Region too large | `box_too_large` | 287 |
| Short strokes only | `no_long_strokes` | 217 |
| Halftone | `halftone` | 204 |
| Too dense | `ink_too_high` | 202 |
| Text block | `text_block` | 75 |
| Physical size unknown | `physical_size_unknown` | 71 |
| Prose in crop | `prose_in_crop` | 45 |
| Solid block | `solid_block` | 35 |
| Too faint | `ink_too_low` | 19 |
| *Screening (8,609)* |  |  |
| Rejected by the screener | `verifier_reject` | 8,609 |
| *Selection (2,043)* |  |  |
| Near-duplicate or second scan | `precard_duplicate` | 1,530 |
| Over the group's quota | `selection_budget` | 513 |
| *Answer-key gates (556)* |  |  |
| Ambiguous entities | `card_ambiguous_entities` | 313 |
| Too few key facts | `card_insufficient_facts` | 224 |
| Writers contradict each other | `card_contradiction_rate` | 19 |

Exclusion reasons with counts, over all candidate regions before the development/test split. The layout model proposes 17,113 regions on 34,540 pages; 5,444 pass screening. A region split into panels is replaced by its panels.
