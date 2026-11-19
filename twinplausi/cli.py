"""Command line: twinplausi {twin, evaluate, benchmark, report}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="twinplausi", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("twin", help="run the digital twin and export generator inputs")
    a.add_argument("--config", default="configs/station.yaml")
    a.add_argument("--out", default="out/scene")
    a.add_argument("--cosmos-root", default=None,
                   help="absolute path the scene folder will have on the GPU machine; "
                        "written into the Cosmos spec files")

    e = sub.add_parser("evaluate", help="check a generated video against the twin")
    e.add_argument("--scene", default="out/scene")
    e.add_argument("--config", default="configs/station.yaml")
    e.add_argument("--video", required=True)
    e.add_argument("--out", default=None)

    b = sub.add_parser("benchmark", help="score the checks on videos with injected faults")
    b.add_argument("--scene", default="out/scene")
    b.add_argument("--config", default="configs/station.yaml")
    b.add_argument("--out", default="results")
    b.add_argument("--n", type=int, default=12, help="videos per fault type")
    b.add_argument("--sweep-seeds", type=int, default=3)
    b.add_argument("--workers", type=int, default=2)

    r = sub.add_parser("report", help="figures and LaTeX tables from benchmark results")
    r.add_argument("--scene", default="out/scene")
    r.add_argument("--config", default="configs/station.yaml")
    r.add_argument("--results", default="results")
    r.add_argument("--figures", default="docs/img")
    r.add_argument("--tables", default="paper/generated")

    args = ap.parse_args(argv)
    from .twin import load_config, load_recording
    cfg = load_config(args.config)

    if args.cmd == "twin":
        from .controls import export_scene
        from .twin import AssemblyTwin
        rec = AssemblyTwin(cfg).run()
        out = export_scene(rec, args.out)
        if args.cosmos_root:
            for spec in (out / "specs").glob("*.json"):
                d = json.loads(spec.read_text())
                for k in ("video_path", "output_dir"):
                    d[k] = f"{args.cosmos_root.rstrip('/')}/{d[k]}"
                for c in ("depth", "seg", "edge", "vis"):
                    if c in d and "control_path" in d[c]:
                        d[c]["control_path"] = f"{args.cosmos_root.rstrip('/')}/{d[c]['control_path']}"
                spec.write_text(json.dumps(d, indent=2) + "\n")
        print(f"twin: {rec.n_frames} frames, {len(rec.instructions)} steps -> {out}")

    elif args.cmd == "evaluate":
        from .evaluate import evaluate_video
        rec = load_recording(args.scene)
        out = args.out or str(Path(args.video).with_suffix("")) + "_eval"
        res = evaluate_video(rec, cfg, args.video, out)
        print(json.dumps({"mot": res["mot"], "events": res["events"]}, indent=2))
        print(f"written to {out}")

    elif args.cmd == "benchmark":
        from .benchmark import run_benchmark, summarise
        rec = load_recording(args.scene)
        det, _ = run_benchmark(rec, cfg, args.out, args.n, args.sweep_seeds, args.workers)
        print(summarise(det).to_string(index=False))

    elif args.cmd == "report":
        from .report import make_report
        make_report(args.scene, cfg, args.results, args.figures, args.tables)


if __name__ == "__main__":
    main()
