"""Shared fixtures for the W3 (submission / sandbox / render / CGX) tests.

Docker-dependent tests carry `@pytest.mark.docker` and are skipped with the reason when the pinned
sandbox image (sandbox/image.lock.json) is not available. Nothing here runs TeX on the host.
"""
from __future__ import annotations

import pytest

from hmb.config import fixture_thresholds
from hmb.store import Layout


def _sandbox_reason() -> str | None:
    try:
        from hmb.submission.sandbox import SandboxUnavailable, ensure_image
        ensure_image()
        return None
    except SandboxUnavailable as e:
        return f"sandbox unavailable: {e}"
    except Exception as e:  # pragma: no cover - defensive
        return f"sandbox check failed: {type(e).__name__}: {e}"


_REASON = _sandbox_reason()


def pytest_collection_modifyitems(config, items):
    if _REASON is None:
        return
    skip = pytest.mark.skip(reason=_REASON)
    for it in items:
        if "docker" in it.keywords and "tests/w3" in str(it.fspath).replace("\\", "/"):
            it.add_marker(skip)


@pytest.fixture
def th():
    return fixture_thresholds()


@pytest.fixture
def layout(tmp_path):
    return Layout(tmp_path / "root")

