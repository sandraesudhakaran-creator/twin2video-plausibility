"""Colour-based part detector.

The parts in the twin have known, distinct colours, and the prompt asks the
generator to keep them. A colour detector is therefore enough for this setup
and keeps the evaluation free of a learned model whose own errors would mix
with the generator's errors. For photoreal outputs where colours change, swap
in a learned detector with the same `detect(frame) -> list[Detection]` call.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    box: np.ndarray        # x1 y1 x2 y2
    area_px: int

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


class ColourDetector:
    def __init__(self, colours: dict[str, list[float]], hue_tol: int = 12,
                 min_sat: int = 90, min_val: int = 50, min_area: int = 60, merge_gap: int = 10):
        self.min_area = min_area
        self.merge_gap = merge_gap
        self.min_sat, self.min_val, self.hue_tol = min_sat, min_val, hue_tol
        self.hues = {}
        for name, rgb in colours.items():
            px = np.uint8([[np.round(np.array(rgb) * 255)]])
            self.hues[name] = int(cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0, 0])

    def masks(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
        out = {}
        for name, hue in self.hues.items():
            dh = np.abs(h - hue)
            dh = np.minimum(dh, 180 - dh)  # hue is circular
            m = ((dh <= self.hue_tol) & (s >= self.min_sat) & (v >= self.min_val)).astype(np.uint8)
            out[name] = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return out

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets = []
        for name, m in self.masks(frame).items():
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            comps = [stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= self.min_area // 3]
            if not comps:
                continue
            comps.sort(key=lambda s: -s[cv2.CC_STAT_AREA])
            # one part can be split by an occluder (the gripper over the cover):
            # merge fragments lying next to the largest piece
            x, y, w, h, a = comps[0]
            box = np.array([x, y, x + w, y + h], float)
            area = a
            for x, y, w, h, a in comps[1:]:
                b = np.array([x, y, x + w, y + h], float)
                gap = max(box[0] - b[2], b[0] - box[2], box[1] - b[3], b[1] - box[3])
                if gap <= self.merge_gap:
                    box = np.array([min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])])
                    area += a
            if area >= self.min_area:
                dets.append(Detection(name, box, int(area)))
        return dets
