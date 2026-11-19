"""Synthetic 'generator' with known failure modes.

Running a real video model needs a large GPU. To develop and validate the
checks before that, this module turns the twin render into a fake generated
video: a random restyle (colour temperature, gamma, blur, noise, compression)
that a good generator is allowed to do, plus optionally one injected physics
fault at known frames. Because we know exactly what was injected, we can
measure how many faults each check finds and how many false alarms it raises.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .twin import Recording

FAULTS = ["vanish", "jump", "drift", "scale", "flicker"]


@dataclass
class Injected:
    kind: str
    obj: str
    start: int
    end: int
    magnitude: float


def restyle(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Appearance change that keeps all geometry: allowed, must not raise flags."""
    gains = rng.uniform(0.9, 1.1, 3)
    gamma = rng.uniform(0.85, 1.15)
    sigma = rng.uniform(0.0, 0.8)
    noise = rng.uniform(1.0, 3.0)
    q = int(rng.integers(70, 95))
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        x = (f.astype(np.float32) / 255.0) ** gamma * gains
        x = np.clip(x * 255 + rng.normal(0, noise, x.shape), 0, 255).astype(np.uint8)
        if sigma > 0.05:
            x = cv2.GaussianBlur(x, (0, 0), sigma)
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
        out[t] = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return out


def _move_object(frame: np.ndarray, mask: np.ndarray, dx: int, dy: int, scale: float = 1.0) -> np.ndarray:
    """Cut the part out (inpaint the hole) and paste it shifted and/or scaled."""
    m8 = mask.astype(np.uint8)
    hole = cv2.dilate(m8, np.ones((5, 5), np.uint8))
    out = cv2.inpaint(frame, hole, 5, cv2.INPAINT_TELEA)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return frame
    cx, cy = xs.mean(), ys.mean()
    # affine map: scale about the part centre, then translate
    A = np.array([[scale, 0, (1 - scale) * cx + dx], [0, scale, (1 - scale) * cy + dy]], np.float32)
    h, w = mask.shape
    obj = cv2.warpAffine(frame, A, (w, h), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(m8 * 255, A, (w, h), flags=cv2.INTER_LINEAR) > 127
    out[m] = obj[m]
    return out


def _vanish(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hole = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    return cv2.inpaint(frame, hole, 5, cv2.INPAINT_TELEA)


def inject(rec: Recording, kind: str | None, rng: np.random.Generator,
           objects: list[str], min_visible_px: int = 400,
           magnitude: float | None = None) -> tuple[np.ndarray, Injected | None]:
    """Restyle the twin render and inject one fault (or none if kind is None).

    magnitude overrides the random severity: pixels for jump/drift, scale
    factor for scale, relative gain for flicker, frames for vanish.
    """
    frames = rec.rgb.copy()
    info = None
    if kind is not None:
        T = rec.n_frames
        L = int(rng.integers(8, 25)) if kind != "drift" else int(rng.integers(18, 30))
        if kind == "vanish" and magnitude is not None:
            L = int(magnitude)
        # pick an object and a window in which it is clearly visible throughout
        for _ in range(200):
            obj = objects[int(rng.integers(len(objects)))]
            s = int(rng.integers(5, T - L - 5))
            if (rec.visible_px(obj)[s:s + L] >= min_visible_px).all():
                break
        else:
            raise RuntimeError("no window where the object is visible")
        e = s + L - 1
        cid = rec.class_id(obj)
        ang = rng.uniform(0, 2 * np.pi)
        if kind == "jump":
            mag = magnitude or rng.uniform(25, 50)
            for t in range(s, e + 1):
                frames[t] = _move_object(frames[t], rec.seg[t] == cid,
                                         int(mag * np.cos(ang)), int(mag * np.sin(ang)))
        elif kind == "drift":
            mag = magnitude or rng.uniform(18, 32)       # reached at the end of the window, ~1 px per frame
            for t in range(s, e + 1):
                k = (t - s + 1) / L
                frames[t] = _move_object(frames[t], rec.seg[t] == cid,
                                         int(round(mag * k * np.cos(ang))), int(round(mag * k * np.sin(ang))))
        elif kind == "scale":
            mag = magnitude or rng.choice([rng.uniform(1.35, 1.6), rng.uniform(0.6, 0.72)])
            for t in range(s, e + 1):
                frames[t] = _move_object(frames[t], rec.seg[t] == cid, 0, 0, scale=mag)
        elif kind == "vanish":
            mag = float(L)
            for t in range(s, e + 1):
                frames[t] = _vanish(frames[t], rec.seg[t] == cid)
        elif kind == "flicker":
            mag = magnitude or rng.uniform(0.08, 0.2)
            obj = "background"
            for i, t in enumerate(range(s, e + 1)):
                g = 1 + mag * (1 if i % 2 == 0 else -1)
                frames[t] = np.clip(frames[t].astype(np.float32) * g, 0, 255).astype(np.uint8)
        else:
            raise ValueError(kind)
        info = Injected(kind, obj, s, e, float(mag))
    return restyle(frames, rng), info


def score(events: list[dict], injected: Injected | None, slack: int = 2) -> dict:
    """Did the matching check find the fault, and which flags are false alarms?"""
    from .checks import Event
    evs = [Event(**e) for e in events]
    hit = False
    latency = None
    false_alarms = []
    for ev in evs:
        inside = injected is not None and ev.overlaps(injected.start, injected.end, slack)
        if inside and ev.check == {"vanish": "permanence"}.get(injected.kind, injected.kind):
            hit = True
            lat = max(0, ev.start - injected.start)
            latency = lat if latency is None else min(latency, lat)
        if not inside:
            false_alarms.append(ev.check)
    return dict(injected=asdict(injected) if injected else None, detected=hit, latency_frames=latency,
                false_alarms=false_alarms, checks_fired=sorted({e.check for e in evs}))
