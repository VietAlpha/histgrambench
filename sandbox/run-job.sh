#!/bin/bash
# HistMathBench v2 job runner (plan §9.6/§9.7). Runs INSIDE the sandbox container only.
# stdin: cand.tex bytes.  stdout: tar of /job/out (minus the font cache) and /job/meta.
# args: CPU_S FSIZE_BYTES PAGE_CAP DIM_CAP_PT PIXEL_CAP RASTER_CPU_S DPI MAX_TEX_BYTES
#
# ENGINE LINE (must equal hmb.submission.sandbox.ENGINE_LINE; asserted by tests). It is the §9.6 line
# minus --safer: luaotfload 3.29 aborts under --safer ("luaotfload can not run with option --safer"),
# so fontspec / wrapper_v1 cannot load with it. Lead decision 2026-09-22, contradiction C16.
set -u
CPU=$1; FSIZE=$2; PAGECAP=$3; DIMPT=$4; PIXCAP=$5; RCPU=$6; DPI=$7; MAXTEX=$8
umask 022
cd /job || exit 90
mkdir -p out meta out/.hmbcache/luatex-cache/generic/names || exit 90
# seed the per-job writable luaotfload cache with the prebuilt names database (harness data only)
cp /opt/hmb/texmf-var/luatex-cache/generic/names/* out/.hmbcache/luatex-cache/generic/names/ || exit 90
head -c "$MAXTEX" > cand.tex
FSZK=$(( (FSIZE + 1023) / 1024 ))
(
  ulimit -t "$CPU" || exit 91
  ulimit -f "$FSZK" || exit 91
  exec /usr/bin/time -f '%U %S %M %e %x' -o /job/meta/time.txt \
    lualatex --nosocket --no-shell-escape --interaction=nonstopmode --halt-on-error --output-directory=/job/out --jobname=cand /job/cand.tex
) > meta/stdout.txt 2> meta/stderr.txt < /dev/null
echo $? > meta/exit
if [ -s out/cand.pdf ]; then
  pdfinfo -box out/cand.pdf > meta/pdfinfo.txt 2>&1
  pdfimages -list out/cand.pdf > meta/pdfimages.txt 2>&1
  GO=$(awk -v cap="$DIMPT" -v pix="$PIXCAP" -v dpi="$DPI" '
        /^Pages:/ {p=$2}
        /^CropBox:/ {w=$4-$2; h=$5-$3}
        END { if (p==1 && w>0 && h>0 && w<=cap && h<=cap && (w*dpi/72)*(h*dpi/72)<=pix) print 1; else print 0 }' meta/pdfinfo.txt)
  echo "$GO" > meta/raster_go
  if [ "$GO" = "1" ]; then
    (
      ulimit -t "$RCPU" || exit 91
      ulimit -f "$FSZK" || exit 91
      exec pdftoppm -png -r "$DPI" -aa yes -aaVector yes -cropbox -singlefile out/cand.pdf out/render
    ) > meta/raster.txt 2>&1 < /dev/null
    echo $? > meta/raster_exit
  fi
fi
pdftoppm -v > meta/pdftoppm_version.txt 2>&1
tar -C /job --exclude=out/.hmbcache -cf - out meta
