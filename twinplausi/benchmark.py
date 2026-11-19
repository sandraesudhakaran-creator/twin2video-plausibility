"""Benchmark the checks on videos with known injected faults.

Two experiments:
  1. detection: N random videos per fault type plus N clean (restyle only)
     videos -> detection rate, latency, false alarms, MOTA per fault type
  2. severity sweep: fixed-size faults from barely visible to obvious
     -> at what size does each check start to see the fault?
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .checks import Thresholds, run_checks
from .evaluate import EVAL_OBJECTS, make_detector
from .faults import FAULTS, inject, score

SWEEP = {
    "jump":    [4, 6, 8, 10, 12, 16, 24, 32],          # px offset
    "drift":   [4, 6, 8, 10, 12, 16, 24, 32],          # px offset reached at end of window
    "scale":   [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50],
    "vanish":  [1, 2, 3, 4, 6, 8, 12, 16],             # frames missing
    "flicker": [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16],  # relative gain
}

_REC = None
_CFG = None


def _init(rec, cfg):
    global _REC, _CFG
    _REC, _CFG = rec, cfg


def _one(job):
    kind, seed, magnitude = job
    rng = np.random.default_rng(seed)
    frames, info = inject(_REC, kind, rng, EVAL_OBJECTS, magnitude=magnitude)
    res = run_checks(_REC, frames, EVAL_OBJECTS, make_detector(_CFG), Thresholds())
    s = score(res["events"], info)
    return dict(kind=kind or "clean", seed=seed, magnitude=info.magnitude if info else None,
                obj=info.obj if info else None, start=info.start if info else None,
                end=info.end if info else None, detected=s["detected"],
                latency_frames=s["latency_frames"], n_false_alarms=len(s["false_alarms"]),
                false_alarm_checks=",".join(s["false_alarms"]), checks_fired=",".join(s["checks_fired"]),
                MOTA=res["mot"]["MOTA"], IDSW=res["mot"]["IDSW"])


def run_benchmark(rec, cfg, out_dir: str | Path, n_per_kind: int = 12, sweep_seeds: int = 4,
                  workers: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(None, 1000 + i, None) for i in range(n_per_kind)]
    jobs += [(k, 2000 + 100 * j + i, None) for j, k in enumerate(FAULTS) for i in range(n_per_kind)]
    sweep = [(k, 5000 + 10 * i + s, m) for k, mags in SWEEP.items()
             for i, m in enumerate(mags) for s in range(sweep_seeds)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(rec, cfg)) as ex:
        det = pd.DataFrame(list(ex.map(_one, jobs)))
        swp = pd.DataFrame(list(ex.map(_one, sweep)))
    det.to_csv(out / "detection_runs.csv", index=False)
    swp.to_csv(out / "severity_sweep.csv", index=False)
    summary = summarise(det)
    summary.to_csv(out / "detection_summary.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(Thresholds().__dict__, indent=2))
    return det, swp


def summarise(det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind in ["clean", *FAULTS]:
        d = det[det.kind == kind]
        rows.append(dict(
            fault=kind, videos=len(d),
            detected=int(d.detected.sum()) if kind != "clean" else None,
            median_latency=float(d.latency_frames.dropna().median()) if kind != "clean" else None,
            false_alarms=int(d.n_false_alarms.sum()),
            mean_MOTA=float(d.MOTA.mean()), min_MOTA=float(d.MOTA.min()),
        ))
    return pd.DataFrame(rows)
