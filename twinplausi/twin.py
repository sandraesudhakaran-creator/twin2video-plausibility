"""Digital twin of the assembly station.

The twin runs the process plan in PyBullet and records, for every video
frame, the RGB image, metric depth, a class segmentation mask and the ground
truth state of every tracked object (3D pose, velocity, visible 2D box).
That ground truth is what generated videos are later checked against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pybullet as p
import yaml


def min_jerk(s: float) -> float:
    """Minimum-jerk time scaling: smooth start and stop, like a real axis."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


@dataclass
class Recording:
    fps: int
    width: int
    height: int
    class_names: list[str]               # index = class id in seg; 0 = background
    tracked: list[str]                   # names of tracked objects
    rgb: np.ndarray                      # (T, H, W, 3) uint8
    depth: np.ndarray                    # (T, H, W) float32, metres
    seg: np.ndarray                      # (T, H, W) uint8 class ids
    steps: list[str]                     # step id per frame
    step_spans: dict[str, tuple[int, int]]  # step id -> [first, last] frame
    instructions: dict[str, str]
    pos3d: dict[str, np.ndarray]         # name -> (T, 3)
    vel3d: dict[str, np.ndarray]         # name -> (T, 3)
    events: dict[str, int] = field(default_factory=dict)  # e.g. release -> frame

    @property
    def n_frames(self) -> int:
        return self.rgb.shape[0]

    def class_id(self, name: str) -> int:
        return self.class_names.index(name)

    def boxes(self, name: str, min_px: int = 30) -> np.ndarray:
        """Visible 2D box per frame from the seg mask, (T, 4) x1 y1 x2 y2, NaN if hidden.

        The mask is opened with a 3x3 kernel first: PyBullet's TinyRenderer
        sometimes draws 1-2 px slivers of a face through the edge of the part
        in front of it, which would stretch the box.
        """
        import cv2
        cid = self.class_id(name)
        kernel = np.ones((3, 3), np.uint8)
        out = np.full((self.n_frames, 4), np.nan)
        for t in range(self.n_frames):
            m = cv2.morphologyEx((self.seg[t] == cid).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            ys, xs = np.nonzero(m)
            if xs.size >= min_px:
                out[t] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        return out

    def visible_px(self, name: str) -> np.ndarray:
        cid = self.class_id(name)
        return (self.seg == cid).reshape(self.n_frames, -1).sum(1)

    def subsample(self, idx: np.ndarray, fps: float) -> "Recording":
        """The twin seen at the frames idx only (to match a generator's time base)."""
        idx = np.asarray(idx)
        steps = [self.steps[i] for i in idx]
        spans: dict[str, tuple[int, int]] = {}
        for t, sid in enumerate(steps):
            spans[sid] = (spans.get(sid, (t, t))[0], t)
        return Recording(
            fps=fps, width=self.width, height=self.height, class_names=self.class_names,
            tracked=self.tracked, rgb=self.rgb[idx], depth=self.depth[idx], seg=self.seg[idx],
            steps=steps, step_spans=spans, instructions=self.instructions,
            pos3d={k: v[idx] for k, v in self.pos3d.items()},
            vel3d={k: v[idx] for k, v in self.vel3d.items()},
            events={k: int(np.argmin(np.abs(idx - v))) for k, v in self.events.items()},
        )

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "twin_gt.npz",
            seg=self.seg,
            depth_mm=np.clip(self.depth * 1000, 0, 65535).astype(np.uint16),
            **{f"pos3d_{k}": v for k, v in self.pos3d.items()},
            **{f"vel3d_{k}": v for k, v in self.vel3d.items()},
        )
        meta = dict(
            fps=self.fps, width=self.width, height=self.height,
            class_names=self.class_names, tracked=self.tracked, steps=self.steps,
            step_spans={k: list(v) for k, v in self.step_spans.items()},
            instructions=self.instructions, events=self.events,
        )
        (out_dir / "twin_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def load_recording(scene_dir: str | Path) -> Recording:
    """Rebuild a Recording from an exported scene folder (RGB comes from rgb.mp4)."""
    from .videoio import read_mp4
    d = Path(scene_dir)
    meta = yaml.safe_load((d / "twin_meta.yaml").read_text())
    z = np.load(d / "twin_gt.npz")
    rgb, _ = read_mp4(d / "rgb.mp4")
    names = meta["tracked"]
    return Recording(
        fps=meta["fps"], width=meta["width"], height=meta["height"],
        class_names=meta["class_names"], tracked=names, rgb=rgb,
        depth=z["depth_mm"].astype(np.float32) / 1000, seg=z["seg"], steps=meta["steps"],
        step_spans={k: tuple(v) for k, v in meta["step_spans"].items()},
        instructions=meta["instructions"],
        pos3d={n: z[f"pos3d_{n}"] for n in names}, vel3d={n: z[f"vel3d_{n}"] for n in names},
        events=meta["events"],
    )


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


class AssemblyTwin:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        sc = cfg["scene"]
        self.fps = sc["fps"]
        self.sim_hz = sc["sim_hz"]
        if self.sim_hz % self.fps:
            raise ValueError("sim_hz must be a multiple of fps")
        self.substeps = self.sim_hz // self.fps
        self.w, self.h = sc["resolution"]
        cam = sc["camera"]
        self.near, self.far = cam["near"], cam["far"]
        self.view = p.computeViewMatrix(cam["eye"], cam["target"], cam["up"])
        self.proj = p.computeProjectionMatrixFOV(cam["fov_deg"], self.w / self.h, self.near, self.far)

    # ------------------------------------------------------------------ setup
    def _build(self):
        self.cid = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1.0 / self.sim_hz, physicsClientId=self.cid)
        self.class_names = ["background", "floor"]
        floor_col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.cid)
        floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[2, 2, 0.001],
                                        rgbaColor=[0.78, 0.76, 0.72, 1], physicsClientId=self.cid)
        floor = p.createMultiBody(0, floor_col, floor_vis, [0, 0, 0], physicsClientId=self.cid)
        self.body_to_class = {floor: 1}
        self.bodies: dict[str, int] = {}
        self.tracked: list[str] = []
        for name, o in self.cfg["objects"].items():
            half = [s / 2 for s in o["size"]]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[*o["rgb"], 1],
                                      physicsClientId=self.cid)
            mass = 0.0 if (o.get("static") or o.get("kinematic")) else o.get("mass", 0.1)
            b = p.createMultiBody(mass, col, vis, o["pos"], physicsClientId=self.cid)
            p.changeDynamics(b, -1, lateralFriction=0.8, restitution=0.05, physicsClientId=self.cid)
            self.bodies[name] = b
            self.class_names.append(name)
            self.body_to_class[b] = len(self.class_names) - 1
            if o.get("tracked"):
                self.tracked.append(name)
        self.held: tuple[str, np.ndarray] | None = None       # (object, offset from tool)
        self.fastened: dict[str, tuple[str, np.ndarray]] = {}  # child -> (parent, offset)

    def _pos(self, name: str) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(self.bodies[name], physicsClientId=self.cid)[0])

    def _set_pos(self, name: str, pos) -> None:
        p.resetBasePositionAndOrientation(self.bodies[name], list(pos), [0, 0, 0, 1], physicsClientId=self.cid)
        p.resetBaseVelocity(self.bodies[name], [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

    # --------------------------------------------------------------- rendering
    def _render(self):
        _, _, rgba, zbuf, seg = p.getCameraImage(
            self.w, self.h, self.view, self.proj, renderer=p.ER_TINY_RENDERER,
            shadow=1, lightDirection=[0.4, -0.6, 1.0], physicsClientId=self.cid)
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(self.h, self.w, 4)[..., :3]
        z = np.asarray(zbuf, dtype=np.float32).reshape(self.h, self.w)
        depth = self.far * self.near / (self.far - (self.far - self.near) * z)
        seg = np.asarray(seg).reshape(self.h, self.w)
        cls = np.zeros_like(seg, dtype=np.uint8)
        for b, c in self.body_to_class.items():
            cls[seg == b] = c
        return rgb, depth, cls

    # -------------------------------------------------------------- execution
    def _step(self, rec: dict, step_id: str):
        # attached objects follow their carrier exactly (kinematic coupling)
        if self.held:
            obj, off = self.held
            self._set_pos(obj, self._pos(self.held_tool) + off)
        for child, (parent, off) in self.fastened.items():
            self._set_pos(child, self._pos(parent) + off)
        p.stepSimulation(physicsClientId=self.cid)
        self._tick += 1
        if self._tick % self.substeps == 0:
            rgb, depth, seg = self._render()
            rec["rgb"].append(rgb)
            rec["depth"].append(depth)
            rec["seg"].append(seg)
            rec["steps"].append(step_id)
            for n in self.tracked:
                rec["pos"][n].append(self._pos(n))

    def _frame_index(self, rec) -> int:
        return len(rec["rgb"])

    def run(self) -> Recording:
        self._build()
        self._tick = 0
        rec = dict(rgb=[], depth=[], seg=[], steps=[], pos={n: [] for n in self.tracked})
        events: dict[str, int] = {}
        instructions = {}
        for step in self.cfg["process_plan"]:
            sid = step["id"]
            instructions[sid] = step["instruction"]
            for a in step["actions"]:
                kind = a["type"]
                if kind == "move":
                    start = self._pos(a["object"])
                    goal = np.array(a["to"], dtype=float)
                    n = int(round(a["duration"] * self.sim_hz))
                    for i in range(1, n + 1):
                        self._set_pos(a["object"], start + (goal - start) * min_jerk(i / n))
                        self._step(rec, sid)
                elif kind == "wait":
                    for _ in range(int(round(a["duration"] * self.sim_hz))):
                        self._step(rec, sid)
                elif kind == "grasp":
                    self.held_tool = a["tool"]
                    self.held = (a["object"], self._pos(a["object"]) - self._pos(a["tool"]))
                    events[f"grasp_{a['object']}"] = self._frame_index(rec)
                elif kind == "release":
                    obj = self.held[0]
                    self.held = None   # from here on gravity and contact decide
                    events[f"release_{obj}"] = self._frame_index(rec)
                elif kind == "fasten":
                    self.fastened[a["object"]] = (a["to"], self._pos(a["object"]) - self._pos(a["to"]))
                    events[f"fasten_{a['object']}"] = self._frame_index(rec)
                else:
                    raise ValueError(f"unknown action {kind}")
        p.disconnect(self.cid)

        steps = rec["steps"]
        spans: dict[str, tuple[int, int]] = {}
        for t, s in enumerate(steps):
            spans[s] = (spans.get(s, (t, t))[0], t)
        pos3d = {n: np.array(v) for n, v in rec["pos"].items()}
        vel3d = {n: np.gradient(v, 1.0 / self.fps, axis=0) for n, v in pos3d.items()}
        return Recording(
            fps=self.fps, width=self.w, height=self.h, class_names=self.class_names,
            tracked=self.tracked, rgb=np.stack(rec["rgb"]), depth=np.stack(rec["depth"]),
            seg=np.stack(rec["seg"]), steps=steps, step_spans=spans,
            instructions=instructions, pos3d=pos3d, vel3d=vel3d, events=events,
        )


def run_twin(config_path: str | Path) -> Recording:
    return AssemblyTwin(load_config(config_path)).run()
