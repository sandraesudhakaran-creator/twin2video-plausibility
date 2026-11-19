import copy
from pathlib import Path

import pytest

from twinplausi.twin import AssemblyTwin, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "station.yaml")


@pytest.fixture(scope="session")
def rec(cfg):
    """A shorter run of the real station (half the durations, 12 fps) to keep CI fast."""
    c = copy.deepcopy(cfg)
    c["scene"]["fps"] = 12
    for step in c["process_plan"]:
        for a in step["actions"]:
            if "duration" in a:
                a["duration"] *= 0.5
    return AssemblyTwin(c).run()
