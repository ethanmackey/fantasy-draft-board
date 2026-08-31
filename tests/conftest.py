"""Shared fixtures.

The real pool is loaded once for the whole session. It costs a second and reads two
CSVs, and every test that needs a pool needs the same one -- but more importantly,
several tests are only meaningful against the real data: the ADP plateau, the bye
weeks, whether there are enough kickers.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draftsim.config import DEFAULT_LEAGUE, League, SimConfig, Budget  # noqa: E402
from draftsim.draft import DraftEngine  # noqa: E402
from draftsim.pool import load_pool  # noqa: E402
from draftsim.season import SeasonSimulator  # noqa: E402


@pytest.fixture(scope="session")
def pool():
    return load_pool()


@pytest.fixture(scope="session")
def league():
    return DEFAULT_LEAGUE


@pytest.fixture
def engine(pool, league):
    return DraftEngine(pool, league)


@pytest.fixture
def season(pool, league):
    return SeasonSimulator(pool, league)


@pytest.fixture
def rng():
    return np.random.default_rng(4242)
