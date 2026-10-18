"""G7d pHash: known hashes, Hamming, union-find clustering (cluster only)."""
import numpy as np
import pytest

from hmb.acquisition import phash as P
from w1_helpers import blank, draw_diagram


def test_constant_image_hash_is_dc_bit_only():
    # DCT of a constant image: only the DC coefficient is non-zero; median of 64 coefs is 0,
    # so exactly the first (MSB) bit is set.
    assert P.phash64(blank(50, 70, 200)) == "8000000000000000"


def test_hamming_known():
    assert P.hamming("0000000000000000", "000000000000000f") == 4
    assert P.hamming("ffffffffffffffff", "0000000000000000") == 64


def test_near_duplicate_small_distance_distinct_image_large():
    a = draw_diagram(blank(300, 300), 150, 150, 100)
    rng = np.random.default_rng(0)
    b = np.clip(a.astype(int) + rng.integers(-20, 21, a.shape), 0, 255).astype(np.uint8)
    c = blank(300, 300)
    c[:, :150] = 0
    assert P.hamming(P.phash64(a), P.phash64(b)) <= 6
    assert P.hamming(P.phash64(a), P.phash64(c)) > 10


def test_union_find_is_transitive_and_ids_content_derived():
    h = {"a": "0000000000000000", "b": "000000000000003f", "c": "0000000000000fff", "d": "ffffffffffffffff"}
    cl = P.cluster(h, 6)                 # a-b = 6, b-c = 6, a-c = 12: chained into one cluster
    assert cl["a"] == cl["b"] == cl["c"] != "singleton"
    assert cl["d"] == "singleton"
    import hashlib
    assert cl["a"] == "dup_" + hashlib.sha256(b"a").hexdigest()[:12]      # id from the smallest member only
    info = P.exact_and_near(h, 6)
    assert info["a"]["cluster_size"] == 3 and info["a"]["exact_cluster_id"] == "singleton"


def test_bad_input():
    with pytest.raises(ValueError):
        P.phash64(np.zeros((0, 5), np.uint8))
