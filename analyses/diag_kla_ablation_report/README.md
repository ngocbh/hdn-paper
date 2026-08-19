# DiagKLA ablation analysis report

`report.tex` is a standalone analysis of all 11 registered DiagKLA ablation checkpoints.
The `figures/` tree contains byte-identical copies of the 66 sealed per-run figures and the
five nominal/crossed aggregate figures produced under:

`/checkpoints/ngocbh/hdn/ckpts/outputs/diag_kla_ablation_probes`

Compile from this directory with the cluster TinyTeX installation:

```bash
export PATH="/home/ngocbh/.TinyTeX/bin/x86_64-linux:$PATH"
latexmk -pdf -interaction=nonstopmode report.tex
```

The capture and plotting code remains in `scripts/analyses/` in the parent repository; this
directory contains only the report and its generated visual artifacts.
