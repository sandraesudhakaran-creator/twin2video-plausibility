"""2D multi-object tracking and CLEAR-MOT scoring.

Tracking-by-detection: a constant-velocity Kalman filter per track and
Hungarian matching on IoU, kept per class. CLEAR-MOT (MOTA, ID switches,
fragmentations) is computed against the twin's ground-truth boxes, the same
way detection-and-tracking results are scored on KITTI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Detection


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class KalmanBox:
    """State [cx, cy, w, h, vx, vy]; constant velocity, size as a random walk."""

    def __init__(self, box: np.ndarray, q: float = 1.0, r: float = 4.0):
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0, 0], float)
        self.P = np.diag([r, r, r, r, 100, 100]).astype(float)
        self.F = np.eye(6)
        self.F[0, 4] = self.F[1, 5] = 1
        self.H = np.eye(4, 6)
        self.Q = np.eye(6) * q
        self.R = np.eye(4) * r

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box

    def update(self, box: np.ndarray) -> None:
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    tid: int
    label: str
    kf: KalmanBox
    hits: int = 1
    misses: int = 0
    last_box: np.ndarray | None = None   # the detection matched in this frame


def match_cost(a: np.ndarray, b: np.ndarray) -> float:
    """1 - IoU when boxes overlap; otherwise 1 + centre distance / box diagonal.

    Pure IoU fails when a part starts moving faster than its own size per
    frame (the cover being lifted at low frame rates): prediction and
    detection no longer overlap. The distance term still links them.
    """
    o = iou(a, b)
    if o > 0:
        return 1 - o
    ca, cb = (a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2
    diag = np.hypot(a[2] - a[0], a[3] - a[1]) + 1e-6
    return 1 + np.linalg.norm(ca - cb) / diag


@dataclass
class Tracker:
    max_cost: float = 2.0      # IoU > 0, or centre within one box diagonal
    max_misses: int = 4
    min_hits: int = 2
    tracks: list[Track] = field(default_factory=list)
    _next: int = 0
    _frame: int = 0

    def step(self, dets: list[Detection]) -> list[tuple[int, str, np.ndarray]]:
        for t in self.tracks:
            t.kf.predict()
        unmatched = set(range(len(dets)))
        for label in {d.label for d in dets} | {t.label for t in self.tracks}:
            tr = [t for t in self.tracks if t.label == label]
            di = [i for i in unmatched if dets[i].label == label]
            if not tr or not di:
                continue
            cost = np.array([[match_cost(t.kf.box, dets[i].box) for i in di] for t in tr])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < self.max_cost:
                    tr[r].kf.update(dets[di[c]].box)
                    tr[r].last_box = dets[di[c]].box
                    tr[r].hits += 1
                    tr[r].misses = -1  # incremented to 0 below
                    unmatched.discard(di[c])
        for t in self.tracks:
            t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for i in unmatched:
            self.tracks.append(Track(self._next, dets[i].label, KalmanBox(dets[i].box), last_box=dets[i].box))
            self._next += 1
        self._frame += 1
        # like SORT: report new tracks straight away during the first frames.
        # The reported box is the matched detection; the Kalman state is only
        # used for association, so its lag under acceleration does not count
        # against the generator.
        return [(t.tid, t.label, t.last_box) for t in self.tracks
                if t.misses == 0 and (t.hits >= self.min_hits or self._frame <= self.min_hits)]


def track_video(det_per_frame: list[list[Detection]]) -> list[list[tuple[int, str, np.ndarray]]]:
    tr = Tracker()
    return [tr.step(d) for d in det_per_frame]


def clear_mot(gt: dict[str, np.ndarray], tracks: list[list[tuple[int, str, np.ndarray]]],
              iou_thr: float = 0.5) -> dict[str, float]:
    """CLEAR-MOT against GT boxes (one GT identity per object name, NaN = not visible)."""
    n_gt = fp = fn = idsw = 0
    ious = []
    last_match: dict[str, int] = {}
    matched_prev: dict[str, bool] = {}
    frag = 0
    T = len(tracks)
    for t in range(T):
        g_names = [n for n, b in gt.items() if not np.isnan(b[t]).any()]
        n_gt += len(g_names)
        hyp = tracks[t]
        cost = np.ones((len(g_names), len(hyp)))
        for i, n in enumerate(g_names):
            for j, (tid, label, box) in enumerate(hyp):
                if label == n:
                    cost[i, j] = 1 - iou(gt[n][t], box)
        matched_g, matched_h = set(), set()
        if g_names and hyp:
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if 1 - cost[r, c] >= iou_thr:
                    n, tid = g_names[r], hyp[c][0]
                    if n in last_match and last_match[n] != tid:
                        idsw += 1
                    if n in matched_prev and not matched_prev[n]:
                        frag += 1
                    last_match[n] = tid
                    ious.append(1 - cost[r, c])
                    matched_g.add(n)
                    matched_h.add(c)
        fn += len(g_names) - len(matched_g)
        fp += len(hyp) - len(matched_h)
        for n in g_names:
            if n in last_match or n in matched_g:
                matched_prev[n] = n in matched_g
    mota = 1 - (fn + fp + idsw) / max(n_gt, 1)
    return dict(MOTA=mota, MOTP_IoU=float(np.mean(ious)) if ious else 0.0,
                FN=fn, FP=fp, IDSW=idsw, FRAG=frag, GT=n_gt)
