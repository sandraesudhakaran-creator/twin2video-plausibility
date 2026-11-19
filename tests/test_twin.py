import numpy as np

from twinplausi.twin import min_jerk


def test_min_jerk_profile():
    assert min_jerk(0) == 0 and min_jerk(1) == 1
    assert abs(min_jerk(0.5) - 0.5) < 1e-12
    s = np.linspace(0, 1, 101)
    assert np.all(np.diff([min_jerk(x) for x in s]) >= 0)  # never moves backwards


def test_steps_follow_the_plan(rec, cfg):
    order = [s["id"] for s in cfg["process_plan"]]
    assert list(rec.step_spans) == order
    ends = [rec.step_spans[s] for s in order]
    for (a0, b0), (a1, b1) in zip(ends, ends[1:]):
        assert a1 == b0 + 1  # steps are contiguous, no gaps or overlaps


def test_cover_falls_and_rests_on_housing(rec, cfg):
    o = cfg["objects"]
    housing_top = o["housing"]["pos"][2] + o["housing"]["size"][2] / 2
    expected_z = housing_top + o["cover"]["size"][2] / 2
    t_rel = rec.events["release_cover"]
    t_fast = rec.events["fasten_cover"]
    z = rec.pos3d["cover"][:, 2]
    assert z[t_rel] > expected_z + 0.01            # released above the housing ...
    assert abs(z[t_fast] - expected_z) < 0.002     # ... and settled on it by gravity
    xy = rec.pos3d["cover"][t_fast, :2] - rec.pos3d["housing"][t_fast, :2]
    assert np.linalg.norm(xy) < 0.005              # fell straight down, did not slide off


def test_fastened_cover_moves_with_housing(rec):
    t0 = rec.events["fasten_cover"]
    d = rec.pos3d["cover"][t0:] - rec.pos3d["housing"][t0:]
    assert np.abs(d - d[0]).max() < 1e-6
    assert rec.pos3d["housing"][-1, 0] - rec.pos3d["housing"][t0, 0] > 0.4


def test_ground_truth_boxes_match_masks(rec):
    b = rec.boxes("housing")
    px = rec.visible_px("housing")
    vis = px >= 30
    assert (~np.isnan(b[vis, 0])).all()
    area = (b[vis, 2] - b[vis, 0]) * (b[vis, 3] - b[vis, 1])
    assert (area >= 0.9 * px[vis]).all()  # the box holds the (cleaned) mask
