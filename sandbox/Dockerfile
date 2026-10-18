# HistMathBench v2 compile sandbox (plan §9.6). Build via `python -m hmb sandbox-build`
# (hmb.submission.sandbox.build_image), which records the resulting image id in sandbox/image.lock.json.
#
# Base: the TeX Live 2025 *historic* image, i.e. the frozen year-final TL2025 snapshot (full scheme),
# pinned by digest. Its texlive.tlpdb hash is recorded in /opt/hmb/image-manifest.json.
ARG BASE_IMAGE=texlive/texlive:TL2025-historic@sha256:f25ee2dcd00f58198f918064f4a1c8562410b33e84155bd55b02b419d73d9391
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
# Debian packages come from snapshot.debian.org at the base image's own snapshot date (frozen apt state).
ARG DEBIAN_SNAPSHOT=20260824T000000Z

RUN printf 'Types: deb\nURIs: http://snapshot.debian.org/archive/debian/%s\nSuites: testing\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.pgp\n' "$DEBIAN_SNAPSHOT" \
      > /etc/apt/sources.list.d/debian.sources \
 && rm -f /etc/apt/sources.list \
 && apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Retries=5 update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      poppler-utils fonts-noto-cjk time \
 && rm -rf /var/lib/apt/lists/*

# Non-root runtime user; the job dir is a tmpfs mounted at run time.
RUN useradd -u 10001 -M -d /job -s /usr/sbin/nologin hmb \
 && mkdir -p /job /opt/hmb/texmf-var /opt/hmb/texmf-home/tex/luatex

# texmf.cnf overrides (the TL local texmf.cnf has priority over texmf-dist/web2c/texmf.cnf).
COPY texmf.cnf.hmb /opt/hmb/texmf.cnf.hmb
RUN cat /opt/hmb/texmf.cnf.hmb >> /usr/local/texlive/2025/texmf.cnf

ENV SOURCE_DATE_EPOCH=0 \
    FORCE_SOURCE_DATE=1 \
    HOME=/job \
    LANG=C.UTF-8

COPY run-job.sh selftest.sh warmup.tex write-manifest.sh /opt/hmb/
COPY luaotfload.conf /opt/hmb/texmf-home/tex/luatex/luaotfload.conf
# Prebuilt font cache: luaotfload names database + processed-font caches for every face of the
# wrapper_v1 font block, built once as root and then left read-only (the root fs is read-only at run).
RUN chmod 0755 /opt/hmb/*.sh \
 && fc-cache -f >/dev/null \
 && mkdir -p /tmp/warm && cd /tmp/warm \
 && export TEXMFCACHE=/opt/hmb/texmf-var TEXMFOUTPUT=/tmp/warm \
 && luaotfload-tool --update --force >/dev/null \
 && lualatex --interaction=nonstopmode --halt-on-error /opt/hmb/warmup.tex >/dev/null \
 && cd / && rm -rf /tmp/warm \
 && chmod -R a+rX,go-w /opt/hmb \
 && /opt/hmb/write-manifest.sh "$BASE_IMAGE" "$DEBIAN_SNAPSHOT" > /opt/hmb/image-manifest.json

USER 10001:10001
WORKDIR /job
# Build-time self-test as the runtime user (texmf.cnf pins, engine confinement, read-only cache).
RUN mkdir -p /tmp/st && HMB_SELFTEST_DIR=/tmp/st /opt/hmb/selftest.sh
CMD ["/opt/hmb/selftest.sh"]
