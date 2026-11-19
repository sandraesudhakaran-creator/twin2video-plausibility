"""Figures for the README and paper, and LaTeX tables from benchmark results."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .checks import Thresholds, per_object_series, run_checks
from .controls import depth_control, edge_control, seg_control, step_overlay
from .evaluate import EVAL_OBJECTS, annotate, make_detector
from .faults import inject
from .twin import load_recording

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4df"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlecolor": INK,
    "axes.titlesize": 9.5, "savefig.dpi": 200, "savefig.bbox": "tight",
})

UNITS = {"jump": "offset (px)", "drift": "final offset (px)", "scale": "scale factor",
         "vanish": "frames missing", "flicker": "brightness change (±)"}
THRESH = {"jump": Thresholds().jump_px, "drift": Thresholds().drift_px,
          "scale": 1 + Thresholds().scale_tol, "vanish": Thresholds().permanence_frames}


def fig_controls(rec, path: Path, t: int):
    rng = np.random.default_rng(7)
    fake, _ = inject(rec, None, rng, EVAL_OBJECTS)
    panels = [
        ("Twin render (RGB)", rec.rgb[t]),
        ("Depth control", np.repeat(depth_control(rec.depth[t:t + 1])[0][..., None], 3, -1)),
        ("Segmentation control", seg_control(rec.seg[t])),
        ("Edge control", np.repeat(edge_control(rec.rgb[t:t + 1])[0][..., None], 3, -1)),
        ("Assistance video", step_overlay(rec)[t]),
        ("Restyled stand-in output", fake[t]),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(9, 3.4))
    for ax, (title, img) in zip(axs.flat, panels):
        ax.imshow(img)
        ax.set_title(title, loc="left")
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_sweep(swp: pd.DataFrame, path: Path):
    kinds = list(UNITS)
    fig, axs = plt.subplots(1, len(kinds), figsize=(10, 2.3), sharey=True)
    for ax, k in zip(axs, kinds):
        d = swp[swp.kind == k].groupby("magnitude").detected.mean().reset_index()
        ax.plot(d.magnitude, d.detected * 100, color=BLUE, lw=2, marker="o", ms=4,
                markeredgecolor="white", markeredgewidth=0.8)
        if k in THRESH:
            ax.axvline(THRESH[k], color=INK2, lw=1, ls="--")
            ax.text(THRESH[k], 50, " threshold", color=INK2, fontsize=7.5, va="center")
        ax.set_title(k, loc="left")
        ax.set_xlabel(UNITS[k])
        ax.set_ylim(-5, 105)
    axs[0].set_ylabel("faults detected (%)")
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def example_fault(rec, cfg, kind="drift", obj="cover", seed=11, magnitude=24.0):
    for s in range(seed, seed + 200):
        frames, info = inject(rec, kind, np.random.default_rng(s), EVAL_OBJECTS, magnitude=magnitude)
        if info.obj == obj:
            return frames, info
    raise RuntimeError("no example found")


def fig_timeline(rec, cfg, path: Path):
    frames, info = example_fault(rec, cfg)
    res = run_checks(rec, frames, EVAL_OBJECTS, make_detector(cfg))
    det = make_detector(cfg)
    dets = [det.detect(f) for f in frames]
    c, _ = per_object_series(dets, info.obj)
    b = rec.boxes(info.obj)
    gt = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    err = np.linalg.norm(c - gt, axis=1)
    t = np.arange(rec.n_frames) / rec.fps
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.axvspan(info.start / rec.fps, info.end / rec.fps, color=ORANGE, alpha=0.18, lw=0)
    ax.text((info.start + info.end) / 2 / rec.fps, Thresholds().drift_px * 3.1, "injected drift",
            color=INK2, ha="center", fontsize=8)
    ax.plot(t, err, color=BLUE, lw=1.6, label=f"{info.obj}: distance to twin position")
    ax.axhline(Thresholds().drift_px, color=INK2, lw=1, ls="--")
    ax.text(t[-1], Thresholds().drift_px, "drift threshold ", color=INK2, fontsize=7.5, ha="right", va="bottom")
    for e in res["events"]:
        ax.plot([e["start"] / rec.fps, e["end"] / rec.fps], [-1.2, -1.2], color=RED, lw=4,
                solid_capstyle="butt")
    ax.text(0, -1.2, "flags ", color=RED, fontsize=7.5, ha="right", va="center")
    for sid, (a, _) in rec.step_spans.items():
        ax.axvline(a / rec.fps, color=GRID, lw=1, zorder=0)
        ax.text(a / rec.fps + 0.05, Thresholds().drift_px * 3.6, sid.replace("_", " "), fontsize=7, color=INK2)
    ax.set_ylim(-3, Thresholds().drift_px * 4)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error (px)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.85), frameon=False, fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)
    return frames, info, res


def gif(frames: np.ndarray, path: Path, fps: int, width: int = 400, step: int = 2):
    h = int(frames.shape[1] * width / frames.shape[2])
    small = [cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA) for f in frames[::step]]
    imageio.mimsave(path, small, duration=step / fps, loop=0)


def latex_tables(summary: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Fault & Videos & Detected & Median latency & False alarms & Mean MOTA \\",
             r" & & & (frames) & & \\", r"\midrule"]
    for _, r in summary.iterrows():
        det = "--" if pd.isna(r.detected) else f"{int(r.detected)}/{int(r.videos)}"
        lat = "--" if pd.isna(r.median_latency) else f"{r.median_latency:.0f}"
        lines.append(f"{r.fault} & {int(r.videos)} & {det} & {lat} & {int(r.false_alarms)} & {r.mean_MOTA:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "detection_table.tex").write_text("\n".join(lines) + "\n")


def make_report(scene, cfg, results, figures, tables):
    figures, tables, results = Path(figures), Path(tables), Path(results)
    figures.mkdir(parents=True, exist_ok=True)
    rec = load_recording(scene)
    t_mid = rec.step_spans["transfer"][0] + 8
    fig_controls(rec, figures / "controls.png", t_mid)
    frames, info, res = fig_timeline(rec, cfg, figures / "timeline.png")
    swp = pd.read_csv(results / "severity_sweep.csv")
    fig_sweep(swp, figures / "sweep.png")
    summary = pd.read_csv(results / "detection_summary.csv")
    latex_tables(summary, tables)
    # GIFs for the README
    gif(step_overlay(rec), figures / "assistance.gif", rec.fps)
    jf, ji = example_fault(rec, cfg, kind="jump", obj="cover", magnitude=35.0)
    jr = run_checks(rec, jf, EVAL_OBJECTS, make_detector(cfg))
    a, b = max(0, ji.start - 24), min(rec.n_frames, ji.end + 24)
    gif(annotate(rec, jf, jr)[a:b], figures / "flagged_jump.gif", rec.fps, width=360, step=2)
    print(summary.to_string(index=False))
