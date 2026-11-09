"""Turn a Blender-exported OpenUSD scene into a physics scene in Isaac Sim.

Run with Isaac Sim's Python:

    /opt/IsaacSim/python.sh astra_scene_to_isaac.py --scene ~/astra-scene/out/scene.usdc \
        --out ~/astra-scene/out [--headless] [--hold-seconds 60]

It references the scene under /World/AstraScene, makes the floor, table top and legs
static colliders, makes the prim named "Box" a 0.15 kg rigid body with a convex-hull
collider, adds the Unitree G1 pinned at the origin facing +x (the same placement the
closed-loop test uses), and steps physics for --settle-seconds. The box must come to
rest on the table, so the check fails if it falls, sinks, or slides. It then saves the
composed stage as scene_physics.usda plus a spectator image, and prints one
ASTRA_ISAAC_REPORT JSON line.
"""
from __future__ import annotations

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True, help="scene.usdc exported by Blender")
parser.add_argument("--out", required=True, help="folder for scene_physics.usda and isaac_view.png")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--settle-seconds", type=float, default=3.0)
parser.add_argument("--hold-seconds", type=float, default=0.0,
                    help="keep the GUI open this long after the check (ignored when headless)")
parser.add_argument("--box-name", default="Box")
parser.add_argument("--box-mass", type=float, default=0.15)
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--no-robot", action="store_true", help="skip adding the G1")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": bool(args.headless), "sync_loads": True,
                                "width": 1600, "height": 900})

import numpy as np  # noqa: E402

SCENE = "/World/AstraScene"
ROBOT = "/World/G1"
BASE_HEIGHT = 0.7923
STATIC_PREFIXES = ("Floor", "TableTop", "TableLeg")
report = {"ok": False, "error": None, "scene": os.path.abspath(args.scene)}


def top_level_name(prim):
    parts = str(prim.GetPath())[len(SCENE) + 1:].split("/")
    return parts[0] if parts else ""


def world_range(prim):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def add_physics(stage):
    from pxr import UsdGeom, UsdPhysics, UsdShade

    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    mat = UsdShade.Material.Define(stage, "/World/PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(args.friction)
    api.CreateDynamicFrictionAttr(args.friction)
    api.CreateRestitutionAttr(0.0)

    static, box = [], None
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(SCENE + "/"):
            continue
        name = top_level_name(prim)
        if name == args.box_name and box is None and path == f"{SCENE}/{name}":
            box = prim
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if name == args.box_name:
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        elif name.startswith(STATIC_PREFIXES):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
            static.append(path)
        else:
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    if box is None:
        raise RuntimeError(f"no prim named {args.box_name!r} under {SCENE}")
    UsdPhysics.RigidBodyAPI.Apply(box)
    UsdPhysics.MassAPI.Apply(box).CreateMassAttr(args.box_mass)
    return box, static


def add_robot(stage):
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.storage.native import get_assets_root_path
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage_utils.add_reference_to_stage(
        usd_path=get_assets_root_path() + "/Isaac/Robots/Unitree/G1/g1.usd", path=ROBOT)
    simulation_app.update()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, BASE_HEIGHT))
    root = Articulation.fetch_articulation_root_api_prim_paths(ROBOT)[0]
    joint = UsdPhysics.FixedJoint.Define(stage, f"{ROBOT}/root_joint")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(root)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, BASE_HEIGHT))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def add_spectator(stage):
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, "/World/Spectator")
    cam.CreateFocalLengthAttr(18.0)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(1.9, -1.6, 1.5))
    # Look back toward the table and robot.
    target, eye = Gf.Vec3d(0.4, 0.0, 0.7), Gf.Vec3d(1.9, -1.6, 1.5)
    m = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)
    return str(cam.GetPath())


try:
    import isaacsim.core.experimental.utils.stage as stage_utils
    import omni.replicator.core as rep
    import omni.timeline
    from isaacsim.core.experimental.prims import RigidPrim

    stage_utils.create_new_stage(template="sunlight")
    stage_utils.add_reference_to_stage(usd_path=os.path.abspath(args.scene), path=SCENE)
    simulation_app.update()
    stage = stage_utils.get_current_stage()

    box_prim, static = add_physics(stage)
    report["static_colliders"] = len(static)
    report["box_prim"] = str(box_prim.GetPath())
    rng = world_range(box_prim)
    box_min, box_max = rng.GetMin(), rng.GetMax()
    table_top = max((world_range(stage.GetPrimAtPath(p)).GetMax()[2] for p in static
                     if top_level_name(stage.GetPrimAtPath(p)).startswith("TableTop")), default=None)
    if table_top is None:
        raise RuntimeError("no TableTop mesh found")
    report["table_top_z"] = round(float(table_top), 4)
    report["box_size_m"] = [round(float(box_max[i] - box_min[i]), 4) for i in range(3)]
    if not args.no_robot:
        add_robot(stage)
    spec = add_spectator(stage)
    simulation_app.update()

    rp = rep.create.render_product(spec, (960, 540))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])
    if not args.headless:
        try:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().camera_path = spec
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: viewport setup failed: {exc}", flush=True)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    box = RigidPrim(str(box_prim.GetPath()))
    simulation_app.update()
    start = box.get_world_poses()[0].numpy().reshape(-1)[:3].astype(float)

    steps = int(args.settle_seconds * 60)
    for _ in range(steps):
        simulation_app.update()
    end = box.get_world_poses()[0].numpy().reshape(-1)[:3].astype(float)

    half_h = report["box_size_m"][2] / 2.0
    rest_gap = float(end[2] - half_h - table_top)
    slide = float(np.linalg.norm(end[:2] - start[:2]))
    report.update({
        "box_start": [round(v, 4) for v in start.tolist()],
        "box_end": [round(v, 4) for v in end.tolist()],
        "box_rest_gap_m": round(rest_gap, 4),
        "box_slide_m": round(slide, 4),
        "settle_steps": steps,
    })
    report["ok"] = abs(rest_gap) < 0.01 and slide < 0.01

    os.makedirs(args.out, exist_ok=True)
    img = rgb.get_data()
    if img is not None and getattr(img, "size", 0):
        from PIL import Image

        Image.fromarray(np.asarray(img)[..., :3]).save(os.path.join(args.out, "isaac_view.png"))
        report["image"] = os.path.join(args.out, "isaac_view.png")
    out_usd = os.path.join(args.out, "scene_physics.usda")
    stage.Export(out_usd)
    report["saved_stage"] = out_usd

    if not args.headless and args.hold_seconds > 0:
        # Hold by wall-clock time. Counting 60 frames per second closed the window
        # early because the GUI renders faster than 60 fps on these GPUs.
        import time

        print(f"ASTRA_ISAAC_HOLD viewing for {args.hold_seconds:.0f} s; close the window to stop sooner",
              flush=True)
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline and simulation_app.is_running():
            simulation_app.update()
except Exception as exc:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    report["error"] = f"{type(exc).__name__}: {exc}"
finally:
    print("ASTRA_ISAAC_REPORT " + json.dumps(report), flush=True)
    simulation_app.close()
