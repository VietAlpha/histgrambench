"""Shared fixtures for the B5 (known-geometry fixtures, ledger, gates) tests. Fixture-profile only."""
import sys

import pytest

from hmb.config import fixture_thresholds

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def set_param(th, name, value):
    th.params[name] = dict(th.params[name], value=value)
    return th


@pytest.fixture
def th():
    """Fixture thresholds (engineering values only)."""
    return fixture_thresholds()


@pytest.fixture
def setp():
    return set_param
