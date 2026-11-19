import json

import numpy as np

from twinplausi.controls import VARIANTS, depth_control, export_scene, scene_prompt
from twinplausi.twin import load_recording
from twinplausi.videoio import align, read_mp4


def test_depth_control_near_is_bright():
    d = np.full((2, 10, 10), 2.0, np.float32)
    d[:, :, :5] = 0.5
    c = depth_control(d)
    assert c[0, 0, 0] > c[0, 0, 9]
    assert np.array_equal(c[0], c[1])  # one scale for the whole clip


def test_prompt_contains_every_step(rec):
    p = scene_prompt(rec, "training_studio")
    for instruction in rec.instructions.values():
        assert instruction.rstrip(".")[1:].lower() in p.lower()


def test_export_round_trip(rec, tmp_path):
    out = export_scene(rec, tmp_path / "scene")
    for f in ["rgb", "depth", "seg", "edge", "assistance"]:
        frames, fps = read_mp4(out / f"{f}.mp4")
        assert frames.shape[:3] == (rec.n_frames, rec.height, rec.width)
        assert round(fps) == rec.fps
    for v in VARIANTS:
        spec = json.loads((out / "specs" / f"{v}.json").read_text())
        assert spec["video_path"] == "rgb.mp4" and spec["prompt"]
        assert {"depth", "seg", "edge"} <= spec.keys()
    back = load_recording(out)
    assert back.step_spans == rec.step_spans
    assert np.array_equal(back.seg, rec.seg)


def test_align_never_duplicates_frames():
    gi, ti = align(n_gen=60, gen_fps=12, n_twin=120, twin_fps=24)
    assert len(set(gi)) == len(gi) and len(set(ti)) == len(ti)
    assert np.array_equal(ti, np.arange(0, 120, 2))
    gi, ti = align(n_gen=100, gen_fps=24, n_twin=120, twin_fps=24)  # shorter output
    assert len(gi) == 100 and np.array_equal(gi, ti)
