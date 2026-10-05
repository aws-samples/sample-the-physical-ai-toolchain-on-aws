"""Download and visualize simulation results."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def visualize_sensors(data_path: str, output_dir: str):
    """Create sensor data visualizations."""
    # allow_pickle=False: results are plain numeric arrays, never Python objects.
    # This refuses any tampered .npz that smuggles a pickle payload (threat T20).
    data = np.load(data_path, allow_pickle=False)

    steps = data["steps"]
    camera_rgb = data["camera_rgb"]
    if camera_rgb.ndim != 4 or len(steps) != len(camera_rgb):
        raise ValueError("Unexpected sensor_data.npz shape; refusing to render")

    mid = len(steps) // 2
    step = int(steps[mid])
    rgb = camera_rgb[mid]

    has_lidar = "lidar_points" in data.files
    num_panels = 3 if has_lidar else 2

    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 5))
    if num_panels == 2:
        axes = list(axes)

    # Camera RGB
    axes[0].imshow(rgb)
    axes[0].set_title(f"Camera RGB (step {step})")
    axes[0].axis("off")

    # Depth map (if available)
    if "camera_depth" in data.files:
        depth = data["camera_depth"][mid]
        im = axes[1].imshow(depth, cmap="plasma")
        axes[1].set_title("Depth Map (meters)")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
    else:
        axes[1].text(0.5, 0.5, "Depth not available", ha="center", va="center", fontsize=12)
        axes[1].set_title("Depth Map")
        axes[1].axis("off")

    # LiDAR top-down (if available)
    if has_lidar:
        points = data["lidar_points"][mid]
        axes[2].scatter(points[:, 0], points[:, 1], s=0.5, c=points[:, 2], cmap="viridis")
        axes[2].set_title("LiDAR Point Cloud (top view)")
        axes[2].set_xlabel("X (m)")
        axes[2].set_ylabel("Y (m)")
        axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensor_visualization.png"), dpi=150)
    plt.close()
    print("Saved: sensor_visualization.png")


def visualize_trajectory(csv_path: str, output_dir: str):
    """Plot joint trajectory over time."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    num_joints = data.shape[1]
    time = np.arange(len(data)) / 240.0

    fig, ax = plt.subplots(figsize=(12, 5))
    for j in range(num_joints):
        ax.plot(time, data[:, j], label=f"Joint {j+1}")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Joint Angle (radians)")
    ax.set_title("Robot Joint Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "joint_trajectory.png"), dpi=150)
    plt.close()
    print("Saved: joint_trajectory.png")


def visualize_lidar_3d(data_path: str, output_dir: str):
    """3D scatter plot of LiDAR point cloud."""
    data = np.load(data_path, allow_pickle=False)

    if "lidar_points" not in data.files:
        print("Skipped: lidar_3d.png (no LiDAR data)")
        return

    steps = data["steps"]
    mid = len(steps) // 2
    points = data["lidar_points"][mid]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=0.3, c=points[:, 2], cmap="coolwarm",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("LiDAR 3D Point Cloud")

    plt.savefig(os.path.join(output_dir, "lidar_3d.png"), dpi=150)
    plt.close()
    print("Saved: lidar_3d.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Results directory")
    parser.add_argument("--output", default=None, help="Output directory for plots")
    args = parser.parse_args()

    output_dir = args.output or args.input
    os.makedirs(output_dir, exist_ok=True)

    sensor_path = os.path.join(args.input, "sensor_data.npz")
    traj_path = os.path.join(args.input, "joint_trajectory.csv")

    if os.path.exists(sensor_path):
        visualize_sensors(sensor_path, output_dir)
        visualize_lidar_3d(sensor_path, output_dir)

    if os.path.exists(traj_path):
        visualize_trajectory(traj_path, output_dir)

    print(f"\nVisualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
