"""64-bit DCT perceptual hash and union-find clustering (§6.3 G7d, §6.7). Cluster only, never drop.

pHash: resize to 32x32 (area interpolation), 2-D DCT-II (orthonormal), keep the top-left 8x8
block rounded to 1e-3, bit = coefficient > median of that block; bits row-major, MSB first; 16 hex chars.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn

from ..hashing import sha256_bytes

PHASH_VERSION = "phash-dct64-1"
SINGLETON = "singleton"


def phash64(gray: np.ndarray) -> str:
    g = np.asarray(gray)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("phash64 needs a non-empty 2-D grayscale array")
    small = cv2.resize(g.astype(np.float64), (32, 32), interpolation=cv2.INTER_AREA)
    # rounding removes floating-point residue so flat regions hash deterministically
    d = np.round(dctn(small, type=2, norm="ortho")[:8, :8], 3)
    bits = (d > np.median(d)).ravel()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def cluster_id(members) -> str:
    return "dup_" + sha256_bytes(min(members).encode())[:12]


def cluster(hashes: dict[str, str], max_hamming: int) -> dict[str, str]:
    """Union-find over all pairs with Hamming <= max_hamming. Returns id -> cluster id.

    Cluster ids derive from the lexicographically smallest member id only
    ("dup_" + sha256(min member)[:12]), so adding members never renames a cluster unless a new
    member sorts first or two clusters merge; singletons get "singleton". Callers that persist ids
    append updated rows when an id changes (append-only).
    """
    ids = sorted(hashes)
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    vals = [int(hashes[i], 16) for i in ids]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if bin(vals[a] ^ vals[b]).count("1") <= max_hamming:
                ra, rb = find(ids[a]), find(ids[b])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    members: dict[str, list[str]] = {}
    for i in ids:
        members.setdefault(find(i), []).append(i)
    out = {}
    for mem in members.values():
        cid = SINGLETON if len(mem) == 1 else cluster_id(mem)
        for i in mem:
            out[i] = cid
    return out


def exact_and_near(hashes: dict[str, str], near: int) -> dict[str, dict]:
    """Per id: exact-duplicate cluster (Hamming 0) and near-duplicate cluster (<= near), sizes."""
    ex, nr = cluster(hashes, 0), cluster(hashes, near)
    size_nr: dict[str, int] = {}
    for c in nr.values():
        size_nr[c] = size_nr.get(c, 0) + 1
    return {i: {"exact_cluster_id": ex[i], "phash_cluster_id": nr[i],
                "cluster_size": 1 if nr[i] == SINGLETON else size_nr[nr[i]]} for i in hashes}
