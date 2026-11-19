"""Physics-plausibility checks for a generated video against the digital twin.

The twin knows where every part is in every frame. A video generated from
the twin's control maps should keep that geometry and motion, whatever it
does to textures and lighting. Each check looks for one way a generator
breaks physics and reports events (check, object, first frame, last frame):

  permanence  a part that the twin shows is missing in the video
  jump        a part moves in one frame much further than the twin moved it
  drift       a part slowly leaves the position the twin gives it
  scale       a rigid part grows or shrinks
  flicker     the static background changes brightness from frame to frame
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .detect import ColourDetector, Detection
from .track import clear_mot, track_video
from .twin import Recording


@dataclass
class Thresholds:
    min_visible_px: int = 150     # twin must show at least this much of a part
    permanence_frames: int = 3    # missing for this many frames in a row
    jump_px: float = 10.0         # per-frame motion error vs twin
    drift_px: float = 9.0         # position error vs twin ...
    drift_frames: int = 6         # ... held for this many frames
    scale_tol: float = 0.25       # relative change in linear size
    scale_frames: int = 4
    flicker_level: float = 6.0    # grey levels of background change not in the twin
    # Thresholds sit well above what the clean twin render and the style
    # changes produce (see tests/test_checks.py) and below the smallest
    # injected faults in the benchmark.


@dataclass
class Event:
    check: str
    obj: str
    start: int
    end: int
    peak: float

    def overlaps(self, start: int, end: int, slack: int = 2) -> bool:
        return self.start <= end + slack and self.end >= start - slack


def _runs(flag: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Start/end (inclusive) of runs of True at least min_len long."""
    out, t, n = [], 0, len(flag)
    while t < n:
        if flag[t]:
            s = t
            while t + 1 < n and flag[t + 1]:
                t += 1
            if t - s + 1 >= min_len:
                out.append((s, t))
        t += 1
    return out


def per_object_series(dets: list[list[Detection]], name: str) -> tuple[np.ndarray, np.ndarray]:
    T = len(dets)
    c = np.full((T, 2), np.nan)
    size = np.full(T, np.nan)
    for t, ds in enumerate(dets):
        for d in ds:
            if d.label == name:
                c[t] = d.centre
                size[t] = np.sqrt((d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
    return c, size


def run_checks(rec: Recording, frames: np.ndarray, objects: list[str],
               detector: ColourDetector, th: Thresholds = Thresholds()) -> dict:
    T = rec.n_frames
    dets = [detector.detect(f) for f in frames]
    events: list[Event] = []
    per_obj = {}
    for name in objects:
        gt_box = rec.boxes(name)
        gt_c = np.stack([(gt_box[:, 0] + gt_box[:, 2]) / 2, (gt_box[:, 1] + gt_box[:, 3]) / 2], 1)
        gt_size = np.sqrt((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1]))
        shown = rec.visible_px(name) >= th.min_visible_px
        c, size = per_object_series(dets, name)
        found = ~np.isnan(c[:, 0])

        missing = shown & ~found
        for s, e in _runs(missing, th.permanence_frames):
            events.append(Event("permanence", name, s, e, float(e - s + 1)))

        both = shown & found
        err = np.linalg.norm(c - gt_c, axis=1)
        step_err = np.full(T, np.nan)
        step_err[1:] = np.linalg.norm(np.diff(c, axis=0) - np.diff(gt_c, axis=0), axis=1)
        ok_pair = np.zeros(T, bool)
        ok_pair[1:] = both[1:] & both[:-1]
        jump = ok_pair & (np.nan_to_num(step_err) > th.jump_px)
        for s, e in _runs(jump, 1):
            events.append(Event("jump", name, s, e, float(np.nanmax(step_err[s:e + 1]))))
        drift = both & (np.nan_to_num(err) > th.drift_px)
        for s, e in _runs(drift, th.drift_frames):
            events.append(Event("drift", name, s, e, float(np.nanmax(err[s:e + 1]))))
        ratio = size / gt_size
        bad_scale = both & (np.abs(np.nan_to_num(ratio, nan=1.0) - 1) > th.scale_tol)
        for s, e in _runs(bad_scale, th.scale_frames):
            events.append(Event("scale", name, s, e, float(np.nanmax(np.abs(ratio[s:e + 1] - 1)))))

        per_step = {}
        for sid, (a, b) in rec.step_spans.items():
            v = err[a:b + 1][both[a:b + 1]]
            per_step[sid] = float(np.mean(v)) if v.size else float("nan")
        per_obj[name] = dict(
            mean_pos_err_px=float(np.nanmean(err[both])) if both.any() else float("nan"),
            detected_frac=float(found[shown].mean()) if shown.any() else float("nan"),
            pos_err_by_step=per_step,
        )

    # flicker: brightness change of static background that the twin does not have
    bg = np.isin(rec.seg, [rec.class_id("background"), rec.class_id("floor")])
    def bg_mean(vid):
        g = vid.mean(-1)
        return np.array([g[t][bg[t]].mean() for t in range(T)])
    d_gen, d_gt = np.diff(bg_mean(frames)), np.diff(bg_mean(rec.rgb))
    fl = np.zeros(T, bool)
    fl[1:] = np.abs(d_gen - d_gt) > th.flicker_level
    for s, e in _runs(fl, 1):
        events.append(Event("flicker", "background", s, e, float(np.abs(d_gen - d_gt)[s - 1:e].max())))

    gt_boxes = {n: rec.boxes(n) for n in objects}
    mot = clear_mot(gt_boxes, track_video(dets))
    return dict(events=[asdict(e) for e in sorted(events, key=lambda e: e.start)],
                objects=per_obj, mot=mot)
