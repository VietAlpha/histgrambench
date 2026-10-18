#!/bin/bash
# Records the verifiable identity of the image contents (sandbox pins, plan §9.6 / §15.1).
set -eu
BASE="$1"; SNAP="$2"
TLPDB=$(ls /usr/local/texlive/*/tlpkg/texlive.tlpdb)
printf '{\n'
printf ' "base_image": "%s",\n' "$BASE"
printf ' "debian_snapshot": "%s",\n' "$SNAP"
printf ' "texlive_tlpdb_path": "%s",\n' "$TLPDB"
printf ' "texlive_tlpdb_sha256": "%s",\n' "$(sha256sum "$TLPDB" | cut -d" " -f1)"
printf ' "texlive_release": "%s",\n' "$(grep -m1 '^depend release/' "$TLPDB" | cut -d/ -f2)"
printf ' "engine": "%s",\n' "$(lualatex --version | head -1)"
printf ' "rasterizer": "%s",\n' "$(pdftoppm -v 2>&1 | head -1)"
printf ' "dpkg_manifest_sha256": "%s",\n' "$(dpkg-query -W -f='${Package}=${Version}\n' | LC_ALL=C sort | sha256sum | cut -d" " -f1)"
printf ' "fonts_manifest_sha256": "%s",\n' "$(fc-list --format '%{file}\n' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d" " -f1)"
printf ' "font_cache_sha256": "%s",\n' "$(cd /opt/hmb/texmf-var && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d" " -f1)"
printf ' "shell_escape": "%s",\n' "$(kpsewhich -var-value=shell_escape)"
printf ' "openout_any": "%s",\n' "$(kpsewhich -var-value=openout_any)"
printf ' "arch": "%s"\n' "$(uname -m)"
printf '}\n'
