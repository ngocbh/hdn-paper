# DiagKLA ablation analysis report

`report.tex` is a standalone analysis of all 11 registered DiagKLA ablation checkpoints.
The `figures/` tree contains byte-identical copies of the 66 sealed per-run figures and the
five nominal/crossed aggregate figures produced under:

`/checkpoints/ngocbh/hdn/ckpts/outputs/diag_kla_ablation_probes`

Compile from this directory with the cluster TinyTeX installation:

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
latexmk -pdf -interaction=nonstopmode report.tex
```

The checked-in PDF uses 300-dpi plot downsampling with lossless Flate encoding to stay below
ordinary Git host file-size limits while the original PNGs remain unchanged:

```bash
gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dCompatibilityLevel=1.7 \
  -dDetectDuplicateImages=true -dDownsampleColorImages=true \
  -dColorImageDownsampleType=/Bicubic -dColorImageResolution=300 \
  -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode \
  -dDownsampleGrayImages=true -dGrayImageDownsampleType=/Bicubic \
  -dGrayImageResolution=300 -dAutoFilterGrayImages=false \
  -dGrayImageFilter=/FlateEncode -sOutputFile=report.compact.pdf report.pdf
mv report.compact.pdf report.pdf
```

The capture and plotting code remains in `scripts/analyses/` in the parent repository; this
directory contains only the report and its generated visual artifacts.
