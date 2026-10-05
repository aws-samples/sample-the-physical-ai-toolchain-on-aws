"""Run Isaac Sim headless. Load scene, robot, camera, run physics loop, save outputs."""

import argparse
import math
import os

import numpy as np
import yaml
from isaacsim import SimulationApp


def parse_args():
    p = argparse.ArgumentParser(description="Isaac Sim Batch Simulation")
    p.add_argument("--config", required=True, help="Path to config YAML")
    p.add_argument("--output-dir", required=True, help="Output directory for results")
    return p.parse_args()


def compute_trajectory(step: int, config: dict, num_dof: int = 9) -> list:
    """Sinusoidal joint trajectory for demonstration. Clamped to safe range."""
    # Trajectory advances at the render-step rate (one render step per loop iteration)
    t = step / config["render_hz"]
    num_arm_joints = 7
    # Franka Panda joint limits are approximately +/- 2.8 rad.
    # Keep amplitude conservative (0.3 rad) to avoid PhysX instability.
    arm_targets = [
        0.3 * math.sin(2 * math.pi * 0.2 * t + i * math.pi / 3)
        for i in range(num_arm_joints)
    ]
    # Pad with gripper joints (0.04 = open position for Franka fingers)
    gripper_targets = [0.04] * (num_dof - num_arm_joints)
    return arm_targets + gripper_targets


def validate_config(config: dict) -> None:
    """Validate the simulation config before use (threat T5).

    Rejects malformed or malicious configs: wrong types, out-of-range values, or
    an environment_usd that escapes the Isaac asset tree (it is concatenated onto
    the asset root and loaded). Unknown keys are ignored (e.g. domain_randomization
    added by generate_configs.py) since they are never interpolated into paths.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")

    def _num(key, lo, hi, required=False, default=None):
        if key not in config:
            if required:
                raise ValueError(f"config missing required key: {key}")
            return default
        v = config[key]
        # bool is a subclass of int — reject it explicitly.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"config[{key!r}] must be a number")
        if not (lo <= v <= hi):
            raise ValueError(f"config[{key!r}]={v} out of range [{lo}, {hi}]")
        return v

    _num("physics_hz", 1, 10000, required=True)
    _num("render_hz", 1, 10000, required=True)
    _num("num_steps", 1, 1_000_000, required=True)
    _num("render_width", 16, 7680)
    _num("render_height", 16, 7680)
    _num("num_joints", 1, 64)
    _num("capture_interval", 1, 1_000_000)

    env = config.get("environment_usd", "/Isaac/Environments/Simple_Room/simple_room.usd")
    if not isinstance(env, str):
        raise ValueError("config['environment_usd'] must be a string")
    if not env.startswith("/Isaac/") or ".." in env or "://" in env:
        raise ValueError(
            "config['environment_usd'] must be an /Isaac/... asset path "
            "(no '..' traversal, no URL scheme)"
        )


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    validate_config(config)

    # Launch headless
    sim_app = SimulationApp({
        "headless": True,
        "width": config.get("render_width", 640),
        "height": config.get("render_height", 480),
    })

    # Import after SimulationApp init (Isaac Sim 6.0 namespace)
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from pxr import UsdGeom, Gf

    assets_root = get_assets_root_path()

    # Create world
    world = World(
        physics_dt=1.0 / config["physics_hz"],
        rendering_dt=1.0 / config["render_hz"],
        stage_units_in_meters=1.0,
    )

    # Load NVIDIA bundled environment (includes lighting, floor, materials)
    env_usd = config.get("environment_usd", "/Isaac/Environments/Simple_Room/simple_room.usd")
    env_path = assets_root + env_usd
    add_reference_to_stage(usd_path=env_path, prim_path="/World/Environment")
    print(f"Loaded environment: {env_path}")

    # Load Franka Panda robot from NVIDIA asset server
    franka_path = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=franka_path, prim_path="/World/Robot")
    print(f"Loaded robot: {franka_path}")

    # Setup camera with look-at targeting the robot
    stage = world.stage
    camera_path = "/World/ObservationCamera"
    camera_prim = stage.DefinePrim(camera_path, "Camera")
    UsdGeom.Camera(camera_prim).GetFocalLengthAttr().Set(24.0)
    UsdGeom.Camera(camera_prim).GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(
        eye=np.array([1.2, 1.2, 0.8]),
        target=np.array([0.0, 0.0, 0.4]),
        camera_prim_path=camera_path,
    )

    # Setup camera sensor
    camera = None
    try:
        from isaacsim.sensors.camera import Camera
        camera = Camera(
            prim_path=camera_path,
            resolution=(config.get("render_width", 640), config.get("render_height", 480)),
            frequency=30,
        )
        print("Camera sensor configured (fixed viewpoint)")
    except Exception as e:
        print(f"Warning: Camera setup failed: {e}")

    # Add robot to scene BEFORE world.reset() so physics initializes it
    robot_obj = None
    ArticulationAction = None
    try:
        from isaacsim.core.api.robots import Robot
        try:
            from isaacsim.core.api.controllers import ArticulationAction
        except ImportError:
            from isaacsim.core.api.controllers.articulation_controller import ArticulationAction
        robot_obj = world.scene.add(Robot(prim_path="/World/Robot", name="franka"))
    except Exception as e:
        print(f"Warning: Robot scene setup failed: {e}")

    # Reset world (initializes physics for all prims including robot)
    world.reset()

    if camera:
        camera.initialize()

    # Get articulation controller AFTER reset (now initialized)
    articulation = None
    num_dof = config.get("num_joints", 7)
    if robot_obj:
        try:
            robot_obj.initialize()
            articulation = robot_obj.get_articulation_controller()
            num_dof = robot_obj.num_dof
            print(f"Robot articulation ready: {num_dof} DOF")
        except Exception as e:
            print(f"Warning: Articulation setup failed: {e}")
            print("Running without joint control (physics only)")

    os.makedirs(args.output_dir, exist_ok=True)

    observations = []
    joint_states = []
    num_steps = config["num_steps"]
    capture_interval = config.get("capture_interval", 10)

    print(f"Running {num_steps} steps at {config['physics_hz']} Hz...")

    # Warm up renderer. In headless mode the camera annotator needs many
    # render frames before it produces valid data.
    print("Warming up renderer...")
    for _ in range(150):
        world.step(render=True)

    # Verify camera is producing data before main loop
    if camera:
        camera_ready = False
        for _ in range(50):
            world.step(render=True)
            rgb = camera.get_rgba()
            if rgb is not None and rgb.size > 0:
                camera_ready = True
                break
        if camera_ready:
            print(f"Camera ready after warm-up (producing {rgb.shape} frames)")
        else:
            print("Warning: Camera not producing data after warm-up. Disabling captures.")
            camera = None

    # Simulation loop
    for step in range(num_steps):
        # Compute and apply joint targets
        targets = compute_trajectory(step, config, num_dof)

        if articulation:
            try:
                action = ArticulationAction(
                    joint_positions=np.array(targets)
                )
                articulation.apply_action(action)
            except Exception as e:
                if step == 0:
                    print(f"Warning: Joint control failed: {e}")
                    print("Continuing without joint actuation")
                    articulation = None

        # Step physics + render
        world.step(render=(camera is not None))

        # Record actual joint state
        if articulation:
            try:
                joint_pos = articulation.get_joint_positions()
                joint_states.append(joint_pos.tolist() if joint_pos is not None else targets)
            except Exception:
                joint_states.append(targets)
        else:
            joint_states.append(targets)

        # Capture sensor data at intervals
        if camera and step % capture_interval == 0:
            try:
                rgb = camera.get_rgba()
                if rgb is not None and rgb.any():
                    obs = {
                        "step": step,
                        "camera_rgb": rgb[:, :, :3].copy(),
                    }
                    observations.append(obs)
            except Exception as e:
                if step == 0:
                    print(f"Warning: Sensor capture failed: {e}")

        if step % 100 == 0:
            print(f"  Step {step}/{num_steps}")

    print(f"  Step {num_steps}/{num_steps} - Complete")

    # Save outputs.
    # Store plain numeric arrays (not Python objects) so the reader can load
    # with allow_pickle=False. Saving a list of dicts here would force numpy to
    # pickle them, and np.load(allow_pickle=True) on a tampered file is a remote
    # code-execution vector (see threat T20).
    if observations:
        steps = np.array([o["step"] for o in observations], dtype=np.int64)
        camera_rgb = np.stack([o["camera_rgb"] for o in observations]).astype(np.uint8)
        np.savez_compressed(
            os.path.join(args.output_dir, "sensor_data.npz"),
            camera_rgb=camera_rgb,
            steps=steps,
        )
        print(f"Saved {len(observations)} sensor observations")

    np.savetxt(
        os.path.join(args.output_dir, "joint_trajectory.csv"),
        np.array(joint_states),
        delimiter=",",
        header=",".join(f"joint_{i}" for i in range(len(joint_states[0]))),
    )

    with open(os.path.join(args.output_dir, "metadata.yaml"), "w") as f:
        yaml.dump({
            "config": config,
            "num_steps": num_steps,
            "observations_captured": len(observations),
            "robot_loaded": True,
            "camera_enabled": camera is not None,
            "articulation_enabled": articulation is not None,
        }, f)

    print(f"Results saved to {args.output_dir}")
    sim_app.close()
    print("Done.")


if __name__ == "__main__":
    main()
