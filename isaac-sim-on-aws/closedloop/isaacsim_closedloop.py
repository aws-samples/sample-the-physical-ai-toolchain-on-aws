"""Closed-loop test of our GR00T N1.7 g1_dex3 policy on a Unitree G1 in Isaac Sim 6.0.

Unlike an open-loop replay, which plays back joint angles computed offline from the real
video, this script closes the loop. Every action chunk it:
  1. renders the robot's own head camera (cam_high pose, 640x480 RGB),
  2. reads the 28 simulated arm and hand joint angles,
  3. sends both, with the task text, to policy_bridge.py on 127.0.0.1:5568,
  4. drives the arm and hand joints with the returned 16-step chunk, then asks again.

So the robot reacts to what its camera sees in the sim. Its pelvis is pinned, exactly as
in the replay, because the policy only drives arms and hands.

Modes:
  camcheck  build the scene, put the robot in the episode-0 start pose, save one head-
            camera frame, and exit. Used to match the sim camera to the real cam_high.
  run       run --episodes closed-loop episodes and write a JSON report; with --record,
            save spectator frames with the head-camera view inset in the corner.

    cd /opt/IsaacSim && ./python.sh /home/ubuntu/groot-eval/closedloop/isaacsim_closedloop.py \
        --mode run --episodes 10 --record
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("camcheck", "probe", "calib", "graspfit", "run"), default="run")
parser.add_argument("--trace", default="/home/ubuntu/groot-eval/out/trace_ep0.json",
                    help="trace_ep0.json from export_trace.py; supplies joint names and start pose")
parser.add_argument("--out", default="/home/ubuntu/groot-eval/closedloop/out")
parser.add_argument("--tag", default="", help="suffix for camcheck frame names")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--bridge-port", type=int, default=5568)
parser.add_argument("--task", default="pick the blue box")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--episode-frames", type=int, default=300, help="control frames per episode at 30 Hz")
parser.add_argument("--replan-every", type=int, default=8, help="execute N of the 16 chunk steps, then re-query")
parser.add_argument("--smooth-alpha", type=float, default=0.4, help="causal EMA on targets; 1.0 = off")
parser.add_argument("--substeps", type=int, default=2, help="sim updates per 30 Hz control frame (60 Hz physics)")
parser.add_argument("--box-jitter", type=float, default=0.02, help="+/- metres of box xy jitter after episode 0")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--record", action="store_true", help="save spectator+inset frames every 2nd control frame")
# Head camera. Defaults: Unitree G1 URDF d435_joint on torso_link, RealSense D435 colour at 640x480.
parser.add_argument("--cam-offset", type=float, nargs=3, default=(0.0576235, 0.01753, 0.48987))
parser.add_argument("--cam-pitch-deg", type=float, default=46.0, help="downward pitch about torso y")
parser.add_argument("--cam-yaw-deg", type=float, default=4.0)
parser.add_argument("--fx", type=float, default=560.0, help="focal length in pixels at 640 wide")
# Scene, matching the real episode-0 frame: white table, grey wall, cyan-blue box.
parser.add_argument("--box-center", type=float, nargs=3, default=(0.38, -0.045, 0.715))
parser.add_argument("--box-size", type=float, nargs=3, default=(0.08, 0.14, 0.09))
parser.add_argument("--box-color", type=float, nargs=3, default=(0.10, 0.60, 0.82))
parser.add_argument("--box-mass", type=float, default=0.15)
parser.add_argument("--box-friction", type=float, default=1.5)
parser.add_argument("--table-near-x", type=float, default=0.18, help="table near edge ahead of the pelvis")
parser.add_argument("--wall-x", type=float, default=1.25, help="grey wall distance ahead of the pelvis; 0 = none")
parser.add_argument("--hold-damping", type=float, default=300.0)
# Appearance matching. Visual only: none of these change physics or touch the box.
parser.add_argument("--hand-color", type=float, nargs=3, default=(0.05, 0.05, 0.05),
                    help="override Dex3 hand link colour (real hands are black, e.g. 0.05 0.05 0.05)")
parser.add_argument("--dome-intensity", type=float, default=600.0)
parser.add_argument("--table-color", type=float, nargs=3, default=(0.93, 0.93, 0.92))
parser.add_argument("--box-band", type=float, default=0.0,
                    help="height in metres of a white band around the box base, like the real box; 0 = none")
parser.add_argument("--second-box", type=float, nargs=3, default=None,
                    help="xyz of a static second blue box, like the one at the real frame's top right edge")
# Probe mode: policy answers on sim vs real images at recorded states (no stepping).
parser.add_argument("--probe-frames", type=int, nargs="+", default=(0, 8, 16, 24, 32, 40))
parser.add_argument("--real-dir", default="/home/ubuntu/groot-eval/closedloop/real",
                    help="real cam_high frames of episode 0 as f00000.png ...")
parser.add_argument("--samples", type=int, default=4, help="diffusion samples per query; 0 = bridge default")
# Calib mode: fit camera pose to the real frames by overlap of the black hands, then the box.
parser.add_argument("--calib-frames", type=int, nargs="+", default=(0, 24, 40, 56))
parser.add_argument("--calib-pitch", type=float, nargs=3, default=(34.0, 54.0, 2.0), help="start stop step")
parser.add_argument("--calib-fx", type=float, nargs=3, default=(380.0, 640.0, 30.0))
parser.add_argument("--calib-yaw", type=float, nargs=3, default=(-8.0, 8.0, 2.0))
parser.add_argument("--calib-dz", type=float, nargs=3, default=(-0.06, 0.06, 0.03), help="camera height offset, m")
args, _ = parser.parse_known_args()

with open(args.trace) as fh:
    trace = json.load(fh)
SIM_JOINT_NAMES = trace["sim_joint_names"]
START_STATE = trace["state"][0]
os.makedirs(args.out, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": bool(args.headless), "multi_gpu": False, "sync_loads": True,
                                "width": 1600, "height": 900})

import numpy as np  # noqa: E402

BASE_HEIGHT = 0.7923
ROBOT = "/World/G1"
CAM_W, CAM_H = 640, 480
TABLE_DEPTH, TABLE_WIDTH, TABLE_TOP_THICKNESS = 0.6, 1.2, 0.04
TABLE_NEAR_EDGE_X = args.table_near_x

report = {"mode": args.mode, "args": vars(args), "ok": False, "error": None}


def log(msg):
    print(msg, flush=True)


def _cube(stage, path, center, size, color, collider=True):
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    if collider:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def _phys_mat(stage, path, friction):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(0.0)
    try:
        from pxr import PhysxSchema

        PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr("max")
    except Exception:  # noqa: BLE001
        pass
    return mat


def _bind_phys(prim, mat):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def pin_base(stage):
    from isaacsim.core.experimental.prims import Articulation
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(ROBOT)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, BASE_HEIGHT))
    roots = Articulation.fetch_articulation_root_api_prim_paths(ROBOT)
    joint = UsdPhysics.FixedJoint.Define(stage, f"{ROBOT}/root_joint")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(roots[0])])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, BASE_HEIGHT))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    return roots[0]


def build_scene(stage):
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    root = "/World/Scene"
    UsdGeom.Xform.Define(stage, root)
    bx, by, bz = args.box_center
    dx, dy, dz = args.box_size
    top_z = bz - dz / 2.0
    white = tuple(args.table_color)
    tcx = TABLE_NEAR_EDGE_X + TABLE_DEPTH / 2.0
    top = _cube(stage, f"{root}/TableTop", (tcx, 0.0, top_z - TABLE_TOP_THICKNESS / 2.0),
                (TABLE_DEPTH, TABLE_WIDTH, TABLE_TOP_THICKNESS), white)
    leg_h = top_z - TABLE_TOP_THICKNESS
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        _cube(stage, f"{root}/TableLeg{i}",
              (tcx + sx * (TABLE_DEPTH / 2 - 0.04), sy * (TABLE_WIDTH / 2 - 0.04), leg_h / 2),
              (0.04, 0.04, leg_h), white)
    _bind_phys(top, _phys_mat(stage, f"{root}/TableMaterial", 0.6))
    if args.wall_x > 0:
        _cube(stage, f"{root}/Wall", (args.wall_x, 0.0, 1.5), (0.05, 4.0, 3.0), (0.78, 0.79, 0.80), collider=False)
    box_start = (bx, by, bz)
    box = _cube(stage, f"{root}/Box", box_start, (dx, dy, dz), tuple(args.box_color))
    UsdPhysics.RigidBodyAPI.Apply(box)
    UsdPhysics.MassAPI.Apply(box).CreateMassAttr(float(args.box_mass))
    _bind_phys(box, _phys_mat(stage, f"{root}/BoxMaterial", args.box_friction))
    if args.box_band > 0:
        # Thin visual-only shell slightly larger than the box, parented to it so it moves with it.
        from pxr import UsdGeom as _G

        band = _G.Cube.Define(stage, f"{root}/Box/Band")
        band.CreateSizeAttr(1.0)
        band.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.92, 0.92)])
        bxf = _G.Xformable(band.GetPrim())
        frac = min(1.0, args.box_band / dz)
        bxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.5 + frac / 2.0))
        bxf.AddScaleOp().Set(Gf.Vec3f(1.01, 1.01, frac))
    if args.second_box:
        _cube(stage, f"{root}/SecondBox", tuple(args.second_box), (dx, dy, dz), tuple(args.box_color), collider=False)
    # Soft fill light so the table reads white, like the bright real lab.
    dome = UsdLux.DomeLight.Define(stage, "/World/FillDome")
    dome.CreateIntensityAttr(float(args.dome_intensity))
    return f"{root}/Box", box_start, top_z


def recolor_hands(stage, color):
    """Bind a plain preview-surface material to every Dex3 hand link (visual only)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    mat = UsdShade.Material.Define(stage, "/World/Looks/HandBlack")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/HandBlack/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        name = prim.GetName()
        if ("_hand_" in name or name.endswith("_hand_palm_link") or "_palm" in name) and prim.IsA(UsdGeom.Xformable):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
            n += 1
    return n


def find_link(stage, names):
    from pxr import Usd

    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        if prim.GetName() in names:
            return prim
    return None


def cam_matrix(pitch_deg, yaw_deg, offset):
    from pxr import Gf

    th, ps = math.radians(pitch_deg), math.radians(yaw_deg)
    ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    rz = np.array([[math.cos(ps), -math.sin(ps), 0], [math.sin(ps), math.cos(ps), 0], [0, 0, 1]])
    cam_to_link = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    r = rz @ ry @ cam_to_link
    t = offset
    return Gf.Matrix4d(
        r[0, 0], r[1, 0], r[2, 0], 0.0,
        r[0, 1], r[1, 1], r[2, 1], 0.0,
        r[0, 2], r[1, 2], r[2, 2], 0.0,
        float(t[0]), float(t[1]), float(t[2]), 1.0,
    )


def add_head_camera(stage):
    """Camera rigidly attached to the torso at the RealSense mount, looking forward-down."""
    from pxr import Gf, UsdGeom

    torso = find_link(stage, ("torso_link",))
    if torso is None:
        raise RuntimeError("torso_link not found in the G1 asset")
    th, ps = math.radians(args.cam_pitch_deg), math.radians(args.cam_yaw_deg)
    ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    rz = np.array([[math.cos(ps), -math.sin(ps), 0], [math.sin(ps), math.cos(ps), 0], [0, 0, 1]])
    # USD camera looks down -Z with +Y up; the link frame is x forward, z up.
    cam_to_link = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    r = rz @ ry @ cam_to_link
    t = args.cam_offset
    m = Gf.Matrix4d(
        r[0, 0], r[1, 0], r[2, 0], 0.0,
        r[0, 1], r[1, 1], r[2, 1], 0.0,
        r[0, 2], r[1, 2], r[2, 2], 0.0,
        float(t[0]), float(t[1]), float(t[2]), 1.0,
    )
    path = f"{torso.GetPath()}/cam_high"
    cam = UsdGeom.Camera.Define(stage, path)
    aperture = 20.955
    cam.CreateHorizontalApertureAttr(aperture)
    cam.CreateVerticalApertureAttr(aperture * CAM_H / CAM_W)
    cam.CreateFocalLengthAttr(float(args.fx) * aperture / CAM_W)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)
    return path


def add_spectator_camera(stage):
    from pxr import Gf, UsdGeom

    path = "/World/SpectatorCam"
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    eye = Gf.Vec3d(1.15, -1.25, 1.55)
    target = Gf.Vec3d(0.38, 0.0, 0.92)
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(view.GetInverse())
    return path


def save_png(arr, path):
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
    except Exception:  # noqa: BLE001
        np.save(path + ".npy", arr)


class WindowClosed(Exception):
    pass


try:
    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.replicator.core as rep
    import omni.timeline
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.storage.native import get_assets_root_path

    stage_utils.create_new_stage(template="sunlight")
    from isaacsim.core.experimental.objects import GroundPlane

    GroundPlane("/World/GroundPlane", sizes=[[20.0]], positions=[[0.0, 0.0, 0.0]])
    stage_utils.add_reference_to_stage(usd_path=get_assets_root_path() + "/Isaac/Robots/Unitree/G1/g1.usd", path=ROBOT)
    simulation_app.update()
    stage = stage_utils.get_current_stage()
    report["root_link"] = pin_base(stage)
    box_path, box_start, top_z = build_scene(stage)
    cam_path = add_head_camera(stage)
    if args.hand_color:
        report["hand_prims_recolored"] = recolor_hands(stage, args.hand_color)
    spec_path = add_spectator_camera(stage)
    report.update({"head_camera": cam_path, "table_top_z": round(top_z, 4), "box_start": list(box_start)})
    d435 = find_link(stage, ("d435_link", "head_link", "logo_link"))
    report["head_like_link"] = str(d435.GetPath()) if d435 else None
    simulation_app.update()

    head_rp = rep.create.render_product(cam_path, (CAM_W, CAM_H))
    head_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    head_rgb.attach([head_rp])
    spec_rgb = None
    if args.record:
        spec_rp = rep.create.render_product(spec_path, (CAM_W, CAM_H))
        spec_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        spec_rgb.attach([spec_rp])

    if not args.headless:
        try:
            from omni.kit.viewport.utility import create_viewport_window, get_active_viewport

            get_active_viewport().camera_path = spec_path
            create_viewport_window("Robot head camera (cam_high)", width=480, height=360,
                                   position_x=1100, position_y=60, camera_path=cam_path)
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: viewport setup failed: {exc}")

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    robot = Articulation(ROBOT)
    box = RigidPrim(box_path)
    simulation_app.update()

    dof_names = list(robot.dof_names)
    missing = [j for j in SIM_JOINT_NAMES if j not in dof_names]
    if missing:
        raise RuntimeError(f"G1 asset is missing joints {missing}")
    idx = robot.get_dof_indices(SIM_JOINT_NAMES)
    held = [n for n in dof_names if n not in SIM_JOINT_NAMES]
    held_idx = robot.get_dof_indices(held)
    if args.hold_damping > 0:
        robot.set_dof_gains(dampings=np.full((1, len(held)), args.hold_damping, np.float32), dof_indices=held_idx)

    start = np.asarray(START_STATE, np.float32).reshape(1, -1)

    first_frame = {"pending": True}

    def grab(annot):
        # The first frame can take minutes while RTX compiles shaders on a new
        # workstation, so wait by wall-clock time instead of a few updates.
        import time

        limit = 300.0 if first_frame["pending"] else 10.0
        deadline = time.monotonic() + limit
        warned = False
        while True:
            data = annot.get_data()
            if data is not None and getattr(data, "size", 0):
                first_frame["pending"] = False
                arr = np.asarray(data)
                return np.ascontiguousarray(arr[..., :3].astype(np.uint8))
            if time.monotonic() >= deadline:
                raise RuntimeError(f"camera returned no data after {limit:.0f} s")
            if first_frame["pending"] and not warned:
                log("waiting for the first camera frame (RTX shader compile, up to 5 minutes)")
                warned = True
            simulation_app.update()

    def joints():
        return robot.get_dof_positions(dof_indices=idx).numpy().reshape(-1).astype(np.float32)

    def box_pos():
        return box.get_world_poses()[0].numpy().reshape(-1).astype(float)

    def reset(box_xy_offset=(0.0, 0.0)):
        robot.set_dof_positions(start, dof_indices=idx)
        robot.set_dof_position_targets(start, dof_indices=idx)
        p = np.asarray([[box_start[0] + box_xy_offset[0], box_start[1] + box_xy_offset[1], box_start[2]]], np.float32)
        box.set_world_poses(positions=p, orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32))
        box.set_velocities(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
        for _ in range(40):
            simulation_app.update()
        held_now = robot.get_dof_positions(dof_indices=held_idx).numpy().reshape(1, -1)
        robot.set_dof_position_targets(held_now, dof_indices=held_idx)

    reset()

    if args.mode == "graspfit":
        # Where was the box? Drive the arm to the recorded grasp frames and read where the
        # right fingertips and thumb tip are. The box centre lies between them.
        states = np.asarray(trace["state"], np.float32)
        names = {"palm": ("right_hand_palm_link",), "thumb": ("right_hand_thumb_2_link",),
                 "index": ("right_hand_index_1_link",), "middle": ("right_hand_middle_1_link",)}
        prims = {}
        for key, cands in names.items():
            pr = find_link(stage, cands)
            if pr is None:
                raise RuntimeError(f"link {cands} not found")
            prims[key] = RigidPrim(str(pr.GetPath()))
        far = np.asarray([[5.0, 3.0, 0.05]], np.float32)
        box.set_world_poses(positions=far, orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32))
        rows = []
        for k in range(56, 100, 4):
            s_k = states[k].reshape(1, -1)
            robot.set_dof_positions(s_k, dof_indices=idx)
            robot.set_dof_position_targets(s_k, dof_indices=idx)
            for _ in range(12):
                simulation_app.update()
            pos = {key: pr.get_world_poses()[0].numpy().reshape(-1).astype(float) for key, pr in prims.items()}
            fingers = (pos["index"] + pos["middle"]) / 2.0
            mid = (fingers + pos["thumb"]) / 2.0
            row = {"frame": k, "grip_mid": [round(v, 3) for v in mid], "palm": [round(v, 3) for v in pos["palm"]],
                   "thumb_finger_gap_m": round(float(np.linalg.norm(fingers - pos["thumb"])), 3)}
            rows.append(row)
            log("GRASPFIT " + json.dumps(row))
        report["graspfit"] = rows
        report["ok"] = True
        raise SystemExit(0)

    if args.mode == "calib":
        from PIL import Image
        from pxr import UsdGeom

        cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
        cam_xf = UsdGeom.Xformable(cam.GetPrim()).GetOrderedXformOps()[0]
        states = np.asarray(trace["state"], np.float32)
        aperture = 20.955

        def small(img):
            return np.asarray(Image.fromarray(img).resize((160, 120), Image.BILINEAR), np.int16)

        def dark(a):
            return a.sum(-1) < 3 * 55

        def blue(a):
            return (a[..., 2] > a[..., 0] + 45) & (a[..., 2] > 90)

        def iou(m1, m2):
            u = (m1 | m2).sum()
            return float((m1 & m2).sum()) / float(u) if u else 0.0

        def rng_(t):
            return list(np.arange(t[0], t[1] + 1e-6, t[2]))

        real = {k: small(np.asarray(Image.open(os.path.join(args.real_dir, f"f{k:05d}.png")).convert("RGB")))
                for k in args.calib_frames}
        combos = [(p, f, y, dz) for p in rng_(args.calib_pitch) for f in rng_(args.calib_fx)
                  for y in rng_(args.calib_yaw) for dz in rng_(args.calib_dz)]
        score = {c: 0.0 for c in combos}
        far = np.asarray([[5.0, 3.0, 0.05]], np.float32)  # move the box out of view for the hand fit
        box.set_world_poses(positions=far, orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32))
        for k in args.calib_frames:
            s_k = states[k].reshape(1, -1)
            robot.set_dof_positions(s_k, dof_indices=idx)
            robot.set_dof_position_targets(s_k, dof_indices=idx)
            for _ in range(10):
                simulation_app.update()
            rd = dark(real[k])
            for c in combos:
                p, f, y, dz = c
                off = (args.cam_offset[0], args.cam_offset[1], args.cam_offset[2] + dz)
                cam_xf.Set(cam_matrix(p, y, off))
                cam.GetFocalLengthAttr().Set(float(f) * aperture / CAM_W)
                simulation_app.update()
                simulation_app.update()
                score[c] += iou(dark(small(grab(head_rgb))), rd) / len(args.calib_frames)
        ranked = sorted(score.items(), key=lambda kv: -kv[1])
        report["calib_hands_top"] = [{"pitch": round(c[0], 1), "fx": round(c[1], 1), "yaw": round(c[2], 1),
                                      "dz": round(c[3], 3), "iou": round(v, 4)} for c, v in ranked[:10]]
        cur = min(combos, key=lambda c: abs(c[0] - args.cam_pitch_deg) + abs(c[1] - args.fx) / 30
                  + abs(c[2] - args.cam_yaw_deg) + abs(c[3]) * 30)
        report["calib_hands_current"] = {"combo": [round(v, 3) for v in cur], "iou": round(score[cur], 4)}
        log("CALIB_HANDS " + json.dumps(report["calib_hands_top"][:5]) + " current " + json.dumps(report["calib_hands_current"]))
        # Box placement with the best camera, at frame 0 where the hands are clear of it.
        p, f, y, dz = ranked[0][0]
        cam_xf.Set(cam_matrix(p, y, (args.cam_offset[0], args.cam_offset[1], args.cam_offset[2] + dz)))
        cam.GetFocalLengthAttr().Set(float(f) * aperture / CAM_W)
        s0 = states[0].reshape(1, -1)
        robot.set_dof_positions(s0, dof_indices=idx)
        robot.set_dof_position_targets(s0, dof_indices=idx)
        for _ in range(10):
            simulation_app.update()
        rb = blue(real[0])
        bscore = []
        for bx in np.arange(0.28, 0.52, 0.02):
            for by in np.arange(-0.20, 0.081, 0.02):
                box.set_world_poses(positions=np.asarray([[bx, by, box_start[2]]], np.float32),
                                    orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32))
                box.set_velocities(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
                simulation_app.update()
                simulation_app.update()
                bscore.append((iou(blue(small(grab(head_rgb))), rb), round(float(bx), 3), round(float(by), 3)))
        bscore.sort(reverse=True)
        report["calib_box_top"] = [{"x": b[1], "y": b[2], "iou": round(b[0], 4)} for b in bscore[:8]]
        log("CALIB_BOX " + json.dumps(report["calib_box_top"][:5]))
        best = bscore[0]
        box.set_world_poses(positions=np.asarray([[best[1], best[2], box_start[2]]], np.float32),
                            orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32))
        for _ in range(3):
            simulation_app.update()
        save_png(grab(head_rgb), os.path.join(args.out, f"calib_best_f000{args.tag}.png"))
        report["calib_flags"] = (f"--cam-pitch-deg {p:.1f} --fx {f:.0f} --cam-yaw-deg {y:.1f} "
                                 f"--cam-offset {args.cam_offset[0]} {args.cam_offset[1]} {args.cam_offset[2] + dz:.4f} "
                                 f"--box-center {best[1]} {best[2]} {box_start[2]}")
        log("CALIB_FLAGS " + report["calib_flags"])
        report["ok"] = True
        raise SystemExit(0)

    if args.mode == "camcheck":
        img = grab(head_rgb)
        save_png(img, os.path.join(args.out, f"sim_cam_frame0{args.tag}.png"))
        report["frame_mean_rgb"] = [round(float(v), 1) for v in img.reshape(-1, 3).mean(0)]
        report["ok"] = True
        raise SystemExit(0)

    from multiprocessing.connection import Client

    conn = Client(("127.0.0.1", args.bridge_port), authkey=b"groot-closedloop")
    conn.send({"cmd": "config"})
    report["bridge"] = {k: v for k, v in conn.recv().items() if k != "modality_config_repr"}
    log(f"bridge: {report['bridge']}")

    def query(img, state):
        req = {"cmd": "act", "image": img.tobytes(), "shape": list(img.shape), "state": list(state), "task": args.task}
        if args.samples:
            req["samples"] = args.samples
        conn.send(req)
        resp = conn.recv()
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return np.asarray(resp["action"], np.float32)

    if args.mode == "probe":
        from PIL import Image

        rec = np.asarray(trace["recorded"], np.float32)
        states = np.asarray(trace["state"], np.float32)
        rows = []
        for k in args.probe_frames:
            s_k = states[k].reshape(1, -1)
            robot.set_dof_positions(s_k, dof_indices=idx)
            robot.set_dof_position_targets(s_k, dof_indices=idx)
            for _ in range(10):
                simulation_app.update()
            # The jump to a recorded pose can sweep the arm through the box; put it back at rest
            # so every probe frame sees the box where the real episode had it before contact.
            box.set_world_poses(positions=np.asarray([box_start], np.float32),
                                orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32))
            box.set_velocities(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
            for _ in range(3):
                simulation_app.update()
            report.setdefault("probe_box_drift_m", []).append(
                round(float(np.linalg.norm(box_pos() - np.asarray(box_start))), 4))
            sim_img = grab(head_rgb)
            save_png(sim_img, os.path.join(args.out, f"probe_sim_f{k:03d}{args.tag}.png"))
            real_img = np.asarray(Image.open(os.path.join(args.real_dir, f"f{k:05d}.png")).convert("RGB"), np.uint8)
            n = min(16, len(rec) - k)
            ref = rec[k:k + n]
            p_sim = query(sim_img, states[k])[:n]
            p_real = query(real_img, states[k])[:n]
            ra = slice(7, 14)
            rh = slice(21, 28)
            row = {
                "frame": k,
                "mae_sim": round(float(np.abs(p_sim - ref).mean()), 4),
                "mae_real": round(float(np.abs(p_real - ref).mean()), 4),
                "mae_sim_right_arm": round(float(np.abs(p_sim[:, ra] - ref[:, ra]).mean()), 4),
                "mae_real_right_arm": round(float(np.abs(p_real[:, ra] - ref[:, ra]).mean()), 4),
                "mae_sim_right_hand": round(float(np.abs(p_sim[:, rh] - ref[:, rh]).mean()), 4),
                "mae_real_right_hand": round(float(np.abs(p_real[:, rh] - ref[:, rh]).mean()), 4),
                "sim_vs_real": round(float(np.abs(p_sim - p_real).mean()), 4),
                "img_mean_sim": [round(float(v), 1) for v in sim_img.reshape(-1, 3).mean(0)],
                "img_mean_real": [round(float(v), 1) for v in real_img.reshape(-1, 3).mean(0)],
            }
            rows.append(row)
            log("PROBE " + json.dumps(row))
        conn.send({"cmd": "close"})
        keys = [k for k in rows[0] if k.startswith(("mae", "sim_vs"))]
        report["probe"] = rows
        report["probe_mean"] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
        report["ok"] = True
        raise SystemExit(0)

    rng = np.random.default_rng(args.seed)
    episodes = []
    frame_dir = os.path.join(args.out, "frames")
    if args.record:
        os.makedirs(frame_dir, exist_ok=True)
    frame_no = 0
    for ep in range(args.episodes):
        off = (0.0, 0.0) if ep == 0 else tuple(rng.uniform(-args.box_jitter, args.box_jitter, 2))
        reset(off)
        rest = box_pos()
        z, xy = [], []
        lat = []
        chunk = None
        target = start.reshape(-1).copy()
        t_ep = time.time()
        for fi in range(args.episode_frames):
            if not args.headless and not simulation_app.is_running():
                raise WindowClosed
            k = fi % args.replan_every
            if k == 0:
                img = grab(head_rgb)
                if ep == 0 and fi == 0:
                    save_png(img, os.path.join(args.out, "sim_cam_ep0_f0.png"))
                t_q = time.time()
                chunk = query(img, joints().tolist())
                lat.append(time.time() - t_q)
            raw = chunk[k]
            target = args.smooth_alpha * raw + (1.0 - args.smooth_alpha) * target
            robot.set_dof_position_targets(target.reshape(1, -1), dof_indices=idx)
            for _ in range(args.substeps):
                simulation_app.update()
            p = box_pos()
            z.append(p[2] - rest[2])
            xy.append(float(np.linalg.norm(p[:2] - rest[:2])))
            if args.record and fi % 2 == 0:
                spec = grab(spec_rgb).copy()
                inset = grab(head_rgb)[::2, ::2]
                h, w = inset.shape[:2]
                spec[8:8 + h + 4, CAM_W - w - 12:CAM_W - 8] = 255
                spec[10:10 + h, CAM_W - w - 10:CAM_W - 10] = inset
                save_png(spec, os.path.join(frame_dir, f"ep{ep:02d}_{frame_no:05d}.png"))
                frame_no += 1
        z = np.asarray(z)
        lifted = z > 0.05
        best_run, run = 0, 0
        for v in lifted:
            run = run + 1 if v else 0
            best_run = max(best_run, run)
        res = {
            "episode": ep,
            "box_offset_xy": [round(float(v), 3) for v in off],
            "lift_max_m": round(float(z.max()), 4),
            "lift_max_frame": int(z.argmax()),
            "frames_above_5cm_longest": int(best_run),
            "xy_travel_max_m": round(float(max(xy)), 4),
            "touched": bool(max(xy) > 0.01 or z.max() > 0.01),
            "success": bool(best_run >= 15),
            "policy_queries": len(lat),
            "latency_mean_s": round(float(np.mean(lat)), 3),
            "wall_s": round(time.time() - t_ep, 1),
        }
        episodes.append(res)
        log(f"EPISODE {json.dumps(res)}")
    conn.send({"cmd": "close"})
    report["episodes"] = episodes
    report["success_rate"] = round(sum(e["success"] for e in episodes) / max(1, len(episodes)), 3)
    report["touched_rate"] = round(sum(e["touched"] for e in episodes) / max(1, len(episodes)), 3)
    report["lift_max_m"] = round(max(e["lift_max_m"] for e in episodes), 4) if episodes else None
    report["ok"] = True

except SystemExit:
    pass
except WindowClosed:
    report["note"] = "window closed early"
except Exception as exc:  # noqa: BLE001
    import traceback

    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc()
    print(report["traceback"], file=sys.stderr, flush=True)

with open(os.path.join(args.out, f"report_{args.mode}.json"), "w") as fh:
    json.dump(report, fh, indent=2)
print("CLOSEDLOOP_REPORT " + json.dumps({k: v for k, v in report.items() if k not in ("args", "traceback", "episodes")}), flush=True)
simulation_app.close()
