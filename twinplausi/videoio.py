"""Small helpers to read and write mp4 files (H.264 through imageio-ffmpeg)."""
from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def write_mp4(path: str | Path, frames: np.ndarray, fps: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim == 3:  # grayscale -> 3 channels, most players and Cosmos expect RGB
        frames = np.repeat(frames[..., None], 3, axis=-1)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=9,
                            pixelformat="yuv420p", macro_block_size=8,
                            ffmpeg_params=["-preset", "veryfast"]) as w:
        for f in frames:
            w.append_data(np.ascontiguousarray(f))
    return path


def read_mp4(path: str | Path) -> tuple[np.ndarray, float]:
    r = imageio.get_reader(path)
    fps = r.get_meta_data().get("fps", 24.0)
    frames = np.stack([f[..., :3] for f in r])
    r.close()
    return frames, fps


def resize_all(frames: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    if frames.shape[1:3] == (h, w):
        return frames
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])


def align(n_gen: int, gen_fps: float, n_twin: int, twin_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Pair generated and twin frames on a common clock.

    Generators rarely return the input frame count or rate. We evaluate on
    the coarser of the two time bases, so no frame is ever duplicated:
    duplicated frames would look like stop-and-go motion and raise false
    'jump' flags. Returns (generated frame indices, twin frame indices).
    """
    duration = min(n_gen / gen_fps, n_twin / twin_fps)
    fps = min(gen_fps, twin_fps)
    t = np.arange(int(np.floor(duration * fps + 1e-6))) / fps
    gi = np.clip(np.round(t * gen_fps).astype(int), 0, n_gen - 1)
    ti = np.clip(np.round(t * twin_fps).astype(int), 0, n_twin - 1)
    return gi, ti
