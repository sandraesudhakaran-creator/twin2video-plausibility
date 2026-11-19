"""Turn a twin recording into inputs for a video generator.

Outputs, per scene:
  rgb.mp4         the plain twin render (the Cosmos `video_path` input)
  depth.mp4       normalised inverse depth, near = bright (depth control)
  seg.mp4         flat colour per object class (segmentation control)
  edge.mp4        Canny edges of the render (edge control)
  prompts/*.txt   one text prompt per visual variant, written from the process plan
  specs/*.json    Cosmos-Transfer2.5 parameter files, one per variant
  assistance.mp4  the twin render annotated with the current process step
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .twin import Recording
from .videoio import write_mp4

# visual variants: what the factory should look like; the geometry and
# timing always come from the twin through the control videos
VARIANTS = {
    "factory_daylight": "a bright modern car-component factory hall with daylight from large windows, "
                        "clean grey epoxy floor, aluminium machine frames",
    "factory_night_shift": "the same factory hall at night under cool LED ceiling lights, "
                           "slight reflections on metal surfaces",
    "training_studio": "a neutral, evenly lit training workshop with a matte light-grey workbench",
}

SEG_PALETTE = np.array([
    [0, 0, 0], [70, 70, 70], [120, 120, 140], [180, 160, 120],
    [30, 90, 230], [250, 140, 20], [220, 220, 230], [120, 220, 120],
], dtype=np.uint8)


def depth_control(depth: np.ndarray) -> np.ndarray:
    """Inverse depth scaled to 0..255 over the whole clip (near = bright).

    One scale for the whole clip, not per frame, so brightness does not pump
    as objects move: the control itself must be temporally stable.
    """
    inv = 1.0 / np.clip(depth, 1e-3, None)
    lo, hi = np.percentile(inv, 1), np.percentile(inv, 99.5)
    return (np.clip((inv - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def seg_control(seg: np.ndarray) -> np.ndarray:
    return SEG_PALETTE[np.clip(seg, 0, len(SEG_PALETTE) - 1)]


def edge_control(rgb: np.ndarray, lo: int = 40, hi: int = 120) -> np.ndarray:
    return np.stack([cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), lo, hi) for f in rgb])


def scene_prompt(rec: Recording, variant: str) -> str:
    steps = ". Then the ".join(s.rstrip(".")[0].lower() + s.rstrip(".")[1:] for s in rec.instructions.values())
    return (
        f"A static camera films an automated assembly station in {VARIANTS[variant]}. "
        f"A dark conveyor belt runs left to right. A blue metal housing, an orange flat cover "
        f"and a small silver gripper are the only moving parts. First the {steps}. "
        f"All motion is smooth and physically realistic: parts keep their size and shape, "
        f"never pass through each other, rest still when not moved and fall straight down when released. "
        f"Photorealistic industrial video, sharp focus, no text, no people."
    )


def cosmos_spec(scene_dir: Path, variant: str, prompt: str, weights: dict[str, float],
                seed: int = 1) -> dict:
    # keys follow the Cosmos-Transfer2.5 quickstart parameter file
    spec = {
        "name": f"{scene_dir.name}_{variant}",
        "prompt": prompt,
        "video_path": "rgb.mp4",
        "output_dir": f"generated/{variant}",
        "guidance": 3,
        "seed": seed,
    }
    for ctrl, wgt in weights.items():
        spec[ctrl] = {"control_path": f"{ctrl}.mp4", "control_weight": wgt}
    return spec


def step_overlay(rec: Recording) -> np.ndarray:
    """Assistance video: current step, instruction and the parts involved."""
    order = list(rec.instructions)
    out = rec.rgb.copy()
    boxes = {n: rec.boxes(n) for n in rec.tracked}
    moving = {n: np.linalg.norm(rec.vel3d[n], axis=1) > 0.01 for n in rec.tracked}
    for t in range(rec.n_frames):
        f = out[t]
        sid = rec.steps[t]
        k = order.index(sid)
        for n in rec.tracked:  # highlight what is moving right now
            b = boxes[n][t]
            if moving[n][t] and not np.isnan(b).any():
                x1, y1, x2, y2 = b.astype(int)
                cv2.rectangle(f, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 230, 0), 2)
        cv2.rectangle(f, (0, 0), (rec.width, 34), (20, 20, 20), -1)
        cv2.putText(f, f"Step {k + 1}/{len(order)}: {rec.instructions[sid]}", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        seg_w = rec.width / len(order)  # progress bar, one segment per step
        for i in range(len(order)):
            col = (60, 200, 90) if i < k else (255, 230, 0) if i == k else (90, 90, 90)
            cv2.rectangle(f, (int(i * seg_w) + 2, rec.height - 8), (int((i + 1) * seg_w) - 2, rec.height - 3), col, -1)
    return out


def export_scene(rec: Recording, out_dir: str | Path,
                 weights: dict[str, float] | None = None) -> Path:
    out = Path(out_dir)
    weights = weights or {"depth": 0.6, "seg": 0.6, "edge": 0.8}
    write_mp4(out / "rgb.mp4", rec.rgb, rec.fps)
    write_mp4(out / "depth.mp4", depth_control(rec.depth), rec.fps)
    write_mp4(out / "seg.mp4", seg_control(rec.seg), rec.fps)
    write_mp4(out / "edge.mp4", edge_control(rec.rgb), rec.fps)
    write_mp4(out / "assistance.mp4", step_overlay(rec), rec.fps)
    (out / "prompts").mkdir(exist_ok=True)
    (out / "specs").mkdir(exist_ok=True)
    for v in VARIANTS:
        prompt = scene_prompt(rec, v)
        (out / "prompts" / f"{v}.txt").write_text(prompt + "\n")
        (out / "specs" / f"{v}.json").write_text(json.dumps(cosmos_spec(out, v, prompt, weights), indent=2) + "\n")
    rec.save(out)
    return out
