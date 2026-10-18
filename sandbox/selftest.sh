#!/bin/bash
# Image self-test (plan §9.6). Runs at image build (as uid 10001) and, with HMB_SELFTEST_RUNTIME=1,
# under the production container flags (read-only root, tmpfs /job). Exit 0 only if all assertions hold.
set -u
D=${HMB_SELFTEST_DIR:-/job/selftest}
RT=${HMB_SELFTEST_RUNTIME:-0}
mkdir -p "$D/out/.hmbcache/luatex-cache/generic/names"; cd "$D"
cp /opt/hmb/texmf-var/luatex-cache/generic/names/* "$D/out/.hmbcache/luatex-cache/generic/names/"
export TEXMFOUTPUT="$D/out" TEXMFCACHE="$D/out/.hmbcache:/opt/hmb/texmf-var"
fail=0
ok()   { echo "SELFTEST PASS $1"; }
bad()  { echo "SELFTEST FAIL $1"; fail=1; }
ENG="lualatex --nosocket --no-shell-escape --interaction=nonstopmode --halt-on-error --output-directory=$D/out"

[ "$(kpsewhich -var-value=shell_escape)" = "f" ] && ok shell_escape=f || bad "shell_escape=$(kpsewhich -var-value=shell_escape)"
[ "$(kpsewhich -var-value=openout_any)" = "p" ] && ok openout_any=p || bad "openout_any=$(kpsewhich -var-value=openout_any)"
grep -q '^TEXMFOUTPUT = /job/out' /usr/local/texlive/2025/texmf.cnf && ok texmfoutput_pinned || bad texmfoutput_not_pinned
[ "$(id -u)" != "0" ] && ok "non_root_uid=$(id -u)" || bad "running as root"
[ -w /opt/hmb/texmf-var ] && bad "font cache writable" || ok font_cache_read_only
( : > /hmb_root_write ) 2>/dev/null && bad write_to_root_succeeded || ok write_to_root_refused
if [ "$RT" = "1" ]; then ( : > /tmp/hmb_tmp_write ) 2>/dev/null && bad write_to_tmp_succeeded || ok write_to_tmp_refused_readonly_root; fi

# \write18 must not run
printf '%s\n' '\documentclass{article}\begin{document}\immediate\write18{touch '"$D"'/pwned18}x\end{document}' > w18.tex
$ENG --jobname=w18 w18.tex >/dev/null 2>&1
[ -e "$D/pwned18" ] && bad write18_executed || ok write18_blocked

# Lua probes (no --safer): command execution, popen, io.open writes, lfs/os.remove/os.rename
TMPT=""
[ "$RT" = "1" ] && TMPT='"/tmp/hmbp_w.txt",'
cat > lua.tex <<LUA
\documentclass{article}
\begin{document}
\directlua{
local function w(k,v) texio.write_nl("log","HMBPROBE "..k.."="..tostring(v)) end
local r1, r2 = os.execute("touch $D/pwned_exec"); w("os_execute", r1)
w("io_popen", (io.popen("touch $D/pwned_popen")))
for _, p in ipairs({ $TMPT "/etc/hmbp_w.txt", "/opt/hmb/hmbp_w.txt", "/hmbp_w.txt", "$D/hmbp_w.txt" }) do
  local f = io.open(p, "w"); w("io_open_w "..p, f); if f then f:write("x"); f:close() end
end
w("os_remove", (os.remove("/opt/hmb/warmup.tex")))
w("os_rename", (os.rename("/opt/hmb/write-manifest.sh", "/opt/hmb/x")))
w("lfs_mkdir_opt", (lfs.mkdir("/opt/hmb/hmbp_dir")))
w("lfs_mkdir_etc", (lfs.mkdir("/etc/hmbp_dir")))
w("lfs_mkdir_root", (lfs.mkdir("/hmbp_dir")))
}
x
\end{document}
LUA
$ENG --jobname=lua lua.tex >/dev/null 2>&1
L=out/lua.log
grep -q 'HMBPROBE os_execute=nil' $L && [ ! -e "$D/pwned_exec" ] && ok os_execute_refused || bad os_execute
grep -q 'HMBPROBE io_popen=nil' $L && [ ! -e "$D/pwned_popen" ] && ok io_popen_refused || bad io_popen
if grep 'HMBPROBE io_open_w' $L | grep -vq '=nil'; then bad "io_open_write_outside_out: $(grep 'HMBPROBE io_open_w' $L | grep -v '=nil' | tr '\n' ' ')"; else ok io_open_write_outside_out_refused; fi
[ "$(grep -c 'HMBPROBE io_open_w' $L)" -ge 4 ] && ok io_open_probes_ran || bad io_open_probes_missing
for k in os_remove os_rename lfs_mkdir_opt lfs_mkdir_etc lfs_mkdir_root; do
  grep -q "HMBPROBE $k=nil" $L && ok "${k}_refused" || bad "$k: $(grep "HMBPROBE $k" $L)"
done
if [ -e /opt/hmb/warmup.tex ] && [ -e /opt/hmb/write-manifest.sh ] && [ ! -e /opt/hmb/hmbp_dir ] && [ ! -e /etc/hmbp_dir ] && [ ! -e /etc/hmbp_w.txt ]; then ok nothing_modified_outside_job; else bad modified_outside_job; fi

# \openout outside the output directory must be refused (openout_any = p)
printf '%s\n' '\documentclass{article}\begin{document}\newwrite\f\immediate\openout\f=/etc/pwnout.txt\immediate\write\f{x}\immediate\closeout\f x\end{document}' > oo.tex
$ENG --jobname=oo oo.tex >/dev/null 2>&1
[ -e /etc/pwnout.txt ] && bad openout_absolute_written || ok openout_absolute_refused

# wrapper_v1 font block compiles with the seeded cache (no names-db rebuild), no lost glyphs
/usr/bin/time -f '%U %S' -o warm.time $ENG --jobname=warm /opt/hmb/warmup.tex > warm.stdout 2>&1
if [ -s out/warm.pdf ] && ! grep -q 'Missing character' out/warm.log; then ok "wrapper_fonts_compile cpu=$(awk '{print $1+$2}' warm.time)s"; else bad wrapper_fonts_compile; fi
if grep -q 'Font names database not found\|Updating the font names database' out/warm.log; then bad names_db_rebuilt_at_runtime; else ok names_db_not_rebuilt; fi

# §9.10 record: --safer is incompatible with luaotfload (reason for the engine-line deviation, C16)
lualatex --safer --interaction=nonstopmode --output-directory="$D/out" --jobname=safer /opt/hmb/warmup.tex >/dev/null 2>&1
if grep -q "run with option --safer" out/safer.log; then echo "SELFTEST INFO safer_incompatible_with_luaotfload (C16)"; else echo "SELFTEST INFO safer_now_compatible_recheck_C16"; fi

if [ $fail = 0 ]; then echo "SELFTEST ALL PASS"; else echo "SELFTEST FAILED"; fi
exit $fail
