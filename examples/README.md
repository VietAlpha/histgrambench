# Minimal example

`sample_root/` is a two-figure excerpt of the run's data root: two scored test figures of one work (Ramus, The Way to Geometry, 1636),
with their records, answer keys, all twelve stored contestant answers (6 models × 2 tracks), compile records, compiled
PDFs, geometry, alignments, verdicts and scores. The files are byte-identical to the corresponding records of the
Hugging Face dataset.

```bash
python examples/minimal_example.py                 # uses examples/sample_root; Docker optional
python examples/minimal_example.py --no-docker     # skip recompilation, use the stored compiled PDFs
HMB_ROOT=... python examples/minimal_example.py --root ../hgb_root --item hmb_...   # any materialised root
```

For each stored answer the script recompiles the raw model output in the sandbox (when Docker and the pinned image
are available), reads the PDF back as vector geometry, recomputes the alignment and every geometry-decided verdict,
takes judge-decided facts from the stored verdict file, and recomputes the item score with the frozen scorer. It
prints the recomputed and stored scores; they must be equal. No model is called.
