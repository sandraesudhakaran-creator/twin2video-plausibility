"""Evaluate one generated video against the twin it was generated from."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .checks import Thresholds, run_checks
from .detect import ColourDetector
from .twin import Recording
from .videoio import align, read_mp4, resize_all, write_mp4

EVAL_OBJECTS = ["housing", "cover"]   # the gripper is grey like the background; see README


def make_detector(cfg: dict, objects=EVAL_OBJECTS) -> ColourDetector:
    return ColourDetector({o: cfg["objects"][o]["rgb"] for o in objects})


def annotate(rec: Recording, frames: np.ndarray, result: dict, objects=EVAL_OBJECTS) -> np.ndarray:
    """Twin boxes (white) over the generated video, flagged frames marked in red."""
    out = frames.copy()
    flagged: dict[int, list[str]] = {}
    for e in result["events"]:
        for t in range(e["start"], e["end"] + 1):
            flagged.setdefault(t, []).append(f"{e['check']}:{e['obj']}")
    boxes = {n: rec.boxes(n) for n in objects}
    for t in range(rec.n_frames):
        f = out[t]
        for n in objects:
            b = boxes[n][t]
            if not np.isnan(b).any():
                x1, y1, x2, y2 = b.astype(int)
                cv2.rectangle(f, (x1, y1), (x2, y2), (255, 255, 255), 1)
        if t in flagged:
            cv2.rectangle(f, (0, 0), (rec.width - 1, rec.height - 1), (230, 40, 40), 4)
            cv2.putText(f, ", ".join(sorted(set(flagged[t]))), (10, rec.height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 80), 1, cv2.LINE_AA)
    return out


def evaluate_video(rec: Recording, cfg: dict, video_path: str | Path, out_dir: str | Path,
                   th: Thresholds = Thresholds()) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, fps = read_mp4(video_path)
    gi, ti = align(len(frames), fps, rec.n_frames, rec.fps)
    if len(ti) != rec.n_frames or not np.array_equal(ti, np.arange(rec.n_frames)):
        rec = rec.subsample(ti, min(fps, rec.fps))
    frames = resize_all(frames[gi], (rec.width, rec.height))
    result = run_checks(rec, frames, EVAL_OBJECTS, make_detector(cfg), th)
    result["video"] = str(video_path)
    result["eval_fps"] = rec.fps
    result["twin_frames_used"] = ti.tolist()
    (out / "plausibility.json").write_text(json.dumps(result, indent=2))
    write_mp4(out / "plausibility_annotated.mp4", annotate(rec, frames, result), rec.fps)
    return result
