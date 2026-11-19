import numpy as np
import pytest

from twinplausi.checks import _runs, run_checks
from twinplausi.evaluate import EVAL_OBJECTS, make_detector
from twinplausi.faults import FAULTS, Injected, inject, score
from twinplausi.track import KalmanBox, clear_mot, iou


def test_runs():
    f = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert _runs(f, 1) == [(1, 2), (4, 6), (8, 8)]
    assert _runs(f, 3) == [(4, 6)]


def test_iou_and_kalman():
    a = np.array([0, 0, 10, 10.0])
    assert iou(a, a) == 1 and iou(a, a + 20) == 0
    kf = KalmanBox(a)
    for k in range(1, 15):
        kf.predict()
        kf.update(a + [2 * k, 0, 2 * k, 0])
    assert abs(kf.x[4] - 2) < 0.2  # learns the 2 px/frame velocity


def test_clear_mot_counts_id_switch():
    box = np.array([[0, 0, 10, 10.0]] * 4)
    gt = {"a": box}
    perfect = [[(0, "a", box[t])] for t in range(4)]
    assert clear_mot(gt, perfect)["MOTA"] == 1
    switched = [[(0, "a", box[0])], [(0, "a", box[1])], [(7, "a", box[2])], [(7, "a", box[3])]]
    m = clear_mot(gt, switched)
    assert m["IDSW"] == 1 and m["MOTA"] == 0.75


def test_detector_finds_parts_on_twin_render(rec, cfg):
    det = make_detector(cfg)
    for name in EVAL_OBJECTS:
        gt = rec.boxes(name)
        vis = np.nonzero(rec.visible_px(name) >= 300)[0]
        ious = []
        for t in vis[::3]:
            d = [x for x in det.detect(rec.rgb[t]) if x.label == name]
            ious.append(iou(d[0].box, gt[t]) if d else 0)
        assert np.mean(ious) > 0.85, name


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clean_restyled_video_raises_no_flags(rec, cfg, seed):
    frames, info = inject(rec, None, np.random.default_rng(seed), EVAL_OBJECTS)
    res = run_checks(rec, frames, EVAL_OBJECTS, make_detector(cfg))
    assert info is None
    assert res["events"] == []
    assert res["mot"]["MOTA"] > 0.95


@pytest.mark.parametrize("kind,mag", [("jump", 30.0), ("drift", 28.0), ("scale", 1.5),
                                      ("vanish", 6), ("flicker", 0.15)])
def test_each_fault_is_found_by_its_check(rec, cfg, kind, mag):
    assert kind in FAULTS
    frames, info = inject(rec, kind, np.random.default_rng(42), EVAL_OBJECTS, magnitude=mag)
    res = run_checks(rec, frames, EVAL_OBJECTS, make_detector(cfg))
    s = score(res["events"], info)
    assert s["detected"], (info, res["events"])
    assert s["false_alarms"] == []


def test_score_counts_flags_outside_the_fault_as_false_alarms():
    inj = Injected("jump", "cover", 10, 20, 30.0)
    ev = [dict(check="jump", obj="cover", start=10, end=10, peak=30.0),
          dict(check="drift", obj="cover", start=12, end=20, peak=30.0),
          dict(check="flicker", obj="background", start=50, end=50, peak=9.0)]
    s = score(ev, inj)
    assert s["detected"] and s["latency_frames"] == 0
    assert s["false_alarms"] == ["flicker"]
