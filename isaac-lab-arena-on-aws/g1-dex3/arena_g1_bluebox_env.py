"""Isaac Lab-Arena environment: Unitree G1 with Dex3 hands picks a blue box from a white table.

Recreates the scene of the workshop's "Validate GR00T policy" page (assets/isaacsim_closedloop.py)
inside Arena, so the fine-tuned GR00T N1.7 `g1_dex3` checkpoint can be scored with Arena's
policy runner and metrics:

* Robot: Arena's `g1_wbc_joint` G1 (29 DoF + Dex3 hands). The whole-body controller keeps the
  legs balanced; the policy drives the 14 arm and 14 hand joints.
* Camera: `cam_high` on the pelvis at the torso_link mount, placed and aimed as calibrated against the real head camera
  (pitch 46 deg down, yaw 4 deg, fx 560 px at 640x480).
* Table and box: the same sizes, colours, friction and mass as the Isaac Sim lab. Heights are
  measured from the pelvis, because the WBC stands the pelvis at 0.757 m (measured) instead of the
  pinned 0.7923 m used there.
* Success: the box is lifted 5 cm above its start height. Touch: the box moves faster than
  0.1 m/s at any point (Arena's object-moved metric).

Load it in Arena with:
  --external_environment_class_path arena_g1_bluebox_env:G1BlueBoxEnvironment g1_blue_box
"""

from __future__ import annotations

import argparse
import json
import math

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Isaac Sim lab geometry (assets/isaacsim_closedloop.py defaults).
SIM_LAB_PELVIS_Z = 0.7923
# torso_link origin in the pelvis frame, measured in Arena (gate 0) with the waist at 0.
TORSO_IN_PELVIS = (-0.002, 0.0, 0.044)
SIM_LAB_TABLE_TOP_Z = 0.67
TABLE_DEPTH, TABLE_WIDTH, TABLE_TOP_THICKNESS = 0.6, 1.2, 0.04
BOX_SIZE = (0.08, 0.14, 0.09)
BOX_COLOR = (0.10, 0.60, 0.82)
TABLE_COLOR = (0.93, 0.93, 0.92)
WALL_COLOR = (0.78, 0.79, 0.80)
CAM_W, CAM_H = 640, 480
HORIZONTAL_APERTURE_MM = 20.955

# The dataset's 28 joints, in observation.state / action order.
DATASET_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]


def _cam_quat_xyzw(pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Camera-to-torso rotation in the OpenGL camera convention (-Z forward, +Y up).

    Same matrix as add_head_camera() in isaacsim_closedloop.py: yaw @ pitch @ cam_to_link.
    """
    th, ps = math.radians(pitch_deg), math.radians(yaw_deg)
    ry = [[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]]
    rz = [[math.cos(ps), -math.sin(ps), 0], [math.sin(ps), math.cos(ps), 0], [0, 0, 1]]
    c2l = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]

    def mm(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    r = mm(mm(rz, ry), c2l)
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        w, x, y, z = (r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        w, x, y, z = (r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        w, x, y, z = (r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s
    return (x, y, z, w)


def _box_lifted(env, object_cfg, start_z: float, lift_m: float):
    import torch  # noqa: F401
    import warp as wp

    pos = wp.to_torch(env.scene[object_cfg.name].data.root_pos_w)
    z = pos[:, 2] - env.scene.env_origins[:, 2]
    return z > start_z + lift_m


def _box_fell(env, object_cfg, min_z: float):
    import warp as wp

    pos = wp.to_torch(env.scene[object_cfg.name].data.root_pos_w)
    return (pos[:, 2] - env.scene.env_origins[:, 2]) < min_z


def _recolor_hands(env, env_ids, color):
    """Bind a dark preview-surface material to every Dex3 hand link (visual only), as the
    Isaac Sim lab does: the real Dex3 hands are black, and the policy saw only real images."""
    del env_ids
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = env.sim.stage
    mat = UsdShade.Material.Define(stage, "/World/Looks/HandBlack")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/HandBlack/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    n = 0
    for env_path in env.scene.env_prim_paths:
        root = stage.GetPrimAtPath(f"{env_path}/Robot")
        if not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if ("_hand_" in name or "_palm" in name) and prim.IsA(UsdGeom.Xformable):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
                n += 1
    print(f"G1DEX3_ENV hand prims recoloured: {n}", flush=True)


class G1BlueBoxEnvironment(ExampleEnvironmentBase):
    """G1 + Dex3 blue-box pick on a white table, matched to the real head camera."""

    name: str = "g1_blue_box"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab import sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
        from isaaclab_arena.metrics.success_rate import SuccessRateMetric
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.task_base import TaskBase
        from isaaclab_arena.utils.pose import Pose, PoseRange
        import isaaclab.envs.mdp as mdp_isaac_lab

        # Heights relative to the pelvis, preserving the camera-to-table geometry of the sim lab.
        table_top_z = args_cli.pelvis_z - (SIM_LAB_PELVIS_Z - SIM_LAB_TABLE_TOP_Z)
        box_z = table_top_z + BOX_SIZE[2] / 2.0 + 0.002
        bx, by = args_cli.box_xy
        near_x = args_cli.table_near_x
        tcx = near_x + TABLE_DEPTH / 2.0

        def static_block(name, center, size, color, collide=True):
            class _Block(Object):
                def __init__(self):
                    self.spawner_cfg = sim_utils.CuboidCfg(
                        size=size,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
                        collision_props=sim_utils.CollisionPropertiesCfg() if collide else None,
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                    )
                    super().__init__(
                        name=name,
                        prim_path="{ENV_REGEX_NS}/" + name,
                        object_type=ObjectType.SPAWNER,
                        initial_pose=Pose(position_xyz=center, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
                        tags=["background", "procedural"],
                    )

            return _Block()

        class BlueBox(Object):
            def __init__(self, initial_pose):
                self.spawner_cfg = sim_utils.CuboidCfg(
                    size=BOX_SIZE,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOX_COLOR, roughness=0.5),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=1.0,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
                )
                super().__init__(
                    name="blue_box",
                    prim_path="{ENV_REGEX_NS}/blue_box",
                    object_type=ObjectType.SPAWNER,
                    initial_pose=initial_pose,
                    tags=["object"],
                )

            def _generate_spawner_cfg(self):
                cfg = RigidObjectCfg(prim_path=self.prim_path, spawn=self.spawner_cfg)
                return self._add_initial_pose_to_cfg(cfg)

            def _requires_reset_pose_event(self) -> bool:
                return self.get_initial_pose() is not None and self.reset_pose

        jitter = args_cli.box_jitter
        box = BlueBox(
            PoseRange(
                position_xyz_min=(bx - jitter, by - jitter, box_z),
                position_xyz_max=(bx + jitter, by + jitter, box_z),
                rpy_min=(0.0, 0.0, 0.0),
                rpy_max=(0.0, 0.0, 0.0),
            )
        )

        leg_h = table_top_z - TABLE_TOP_THICKNESS
        assets = [
            static_block("table_top", (tcx, 0.0, table_top_z - TABLE_TOP_THICKNESS / 2.0),
                         (TABLE_DEPTH, TABLE_WIDTH, TABLE_TOP_THICKNESS), TABLE_COLOR),
            box,
        ]
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            assets.append(static_block(
                f"table_leg_{i}",
                (tcx + sx * (TABLE_DEPTH / 2 - 0.04), sy * (TABLE_WIDTH / 2 - 0.04), leg_h / 2.0),
                (0.04, 0.04, leg_h), TABLE_COLOR))
        if args_cli.wall_x > 0:
            assets.append(static_block("wall", (args_cli.wall_x, 0.0, 1.5), (0.05, 4.0, 3.0), WALL_COLOR, collide=False))

        ground = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=args_cli.dome_intensity)
        )

        class SunLight(Object):
            """Matches the 'sunlight' stage template the Isaac Sim lab starts from."""

            def __init__(self):
                self.spawner_cfg = sim_utils.DistantLightCfg(
                    color=(1.0, 1.0, 1.0), intensity=args_cli.sun_intensity, angle=0.53
                )
                super().__init__(
                    name="sun_light",
                    prim_path="/World/SunLight",
                    object_type=ObjectType.SPAWNER,
                    # Tilted 45 deg from vertical, shining down and forward over the robot's shoulder.
                    initial_pose=Pose(position_xyz=(0.0, 0.0, 3.0), rotation_xyzw=(0.0, -0.3826834, 0.0, 0.9238795)),
                    tags=["light"],
                )

        assets += [ground, light, SunLight()]

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        # Spawn standing: at z = 0 the hands start under the tabletop and the arms hit it as the WBC stands up.
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, args_cli.pelvis_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        if args_cli.start_pose_json:
            with open(args_cli.start_pose_json) as f:
                start = json.load(f)
            if isinstance(start, list):  # 28 values in dataset order
                start = dict(zip(DATASET_JOINTS, start))
            embodiment.set_joint_initial_pos({k: float(v) for k, v in start.items() if k in DATASET_JOINTS})

        # Replace Arena's head camera with our calibrated cam_high. Arena's G1 asset has no prim at
        # Robot/torso_link (a camera there stays fixed in the world), so mount it on Robot/pelvis,
        # which Arena itself anchors to, and add the torso offset (the waist is held at 0).
        cam = embodiment.camera_config.robot_head_cam
        cam.prim_path = "{ENV_REGEX_NS}/Robot/pelvis/RobotHeadCam"
        cam.width, cam.height = CAM_W, CAM_H
        cam.spawn.focal_length = args_cli.fx * HORIZONTAL_APERTURE_MM / CAM_W
        cam.spawn.horizontal_aperture = HORIZONTAL_APERTURE_MM
        # Near clip past the G1 head shell (Arena's own head camera uses 0.1 m).
        cam.spawn.clipping_range = (args_cli.near_clip, 10.0)
        cam.offset.pos = tuple(a + b for a, b in zip(args_cli.cam_offset, TORSO_IN_PELVIS))
        cam.offset.rot = _cam_quat_xyzw(args_cli.cam_pitch_deg, args_cli.cam_yaw_deg)
        cam.offset.convention = "opengl"

        lift_m = args_cli.lift_m

        class BoxLiftTask(TaskBase):
            def __init__(self):
                super().__init__(episode_length_s=args_cli.episode_s, task_description=args_cli.task_description)

            def get_scene_cfg(self):
                return None

            def get_events_cfg(self):
                return None

            def get_mimic_env_cfg(self, arm_mode):
                return None

            def get_termination_cfg(self):
                from isaaclab.utils import configclass

                @configclass
                class _Terms:
                    time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
                    success = TerminationTermCfg(
                        func=_box_lifted,
                        params={"object_cfg": SceneEntityCfg("blue_box"), "start_z": box_z, "lift_m": lift_m},
                    )
                    object_dropped = TerminationTermCfg(
                        func=_box_fell,
                        params={"object_cfg": SceneEntityCfg("blue_box"), "min_z": table_top_z - 0.10},
                    )

                return _Terms()

            def get_metrics(self):
                return [SuccessRateMetric(), ObjectMovedRateMetric(box, object_velocity_threshold=args_cli.touch_speed)]

        def env_cfg_callback(env_cfg):
            from isaaclab.managers import EventTermCfg

            # Fresh camera frame after each reset, so the first query never sees the last episode.
            env_cfg.num_rerenders_on_reset = 1
            if args_cli.hand_color:
                env_cfg.events.recolor_dex3_hands = EventTermCfg(
                    func=_recolor_hands, mode="prestartup", params={"color": tuple(args_cli.hand_color)}
                )
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=assets),
            task=BoxLiftTask(),
            env_cfg_callback=env_cfg_callback,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="g1_wbc_joint")
        parser.add_argument("--task_description", type=str, default="pick the blue box")
        parser.add_argument("--episode_s", type=float, default=10.0, help="episode length; 10 s = 300 dataset frames")
        parser.add_argument("--pelvis_z", type=float, default=0.757, help="standing pelvis height under the WBC (measured)")
        parser.add_argument("--table_near_x", type=float, default=0.18)
        parser.add_argument("--box_xy", type=float, nargs=2, default=(0.38, -0.045))
        parser.add_argument("--box_jitter", type=float, default=0.02, help="+/- m of box xy randomisation per episode")
        parser.add_argument("--wall_x", type=float, default=1.25)
        parser.add_argument("--dome_intensity", type=float, default=600.0)
        parser.add_argument("--sun_intensity", type=float, default=1500.0)
        parser.add_argument("--near_clip", type=float, default=0.12, help="camera near clip (m); clears the head shell")
        parser.add_argument("--hand_color", type=float, nargs=3, default=(0.05, 0.05, 0.05))
        parser.add_argument("--cam_offset", type=float, nargs=3, default=(0.0576235, 0.01753, 0.48987))
        parser.add_argument("--cam_pitch_deg", type=float, default=46.0)
        parser.add_argument("--cam_yaw_deg", type=float, default=4.0)
        parser.add_argument("--fx", type=float, default=560.0)
        parser.add_argument("--lift_m", type=float, default=0.05)
        parser.add_argument("--touch_speed", type=float, default=0.1, help="box speed (m/s) that counts as touched")
        parser.add_argument("--start_pose_json", type=str, default="", help="joint name -> start angle (rad)")
