"""Sensor setup for Isaac Sim: camera and LiDAR."""

from isaacsim.sensors.camera import Camera
from isaacsim.sensors.rtx_lidar import LidarRtx


def setup_camera(prim_path: str, resolution: tuple = (640, 480)) -> Camera:
    """Configure and return a camera sensor."""
    camera = Camera(
        prim_path=prim_path,
        resolution=resolution,
        frequency=30,
    )
    camera.set_focal_length(1.93)
    camera.set_horizontal_aperture(2.4)
    camera.set_clipping_range(0.01, 5.0)
    camera.initialize()
    return camera


def setup_lidar(prim_path: str, config: dict = None) -> LidarRtx:
    """Configure and return a LiDAR sensor."""
    config = config or {}
    lidar = LidarRtx(
        prim_path=prim_path,
        rotation_frequency=config.get("rotation_hz", 20.0),
        horizontal_fov=config.get("horizontal_fov", 360.0),
        vertical_fov=config.get("vertical_fov", 30.0),
        horizontal_resolution=config.get("horizontal_resolution", 0.4),
        vertical_resolution=config.get("vertical_resolution", 2.0),
        max_range=config.get("max_range", 10.0),
        min_range=config.get("min_range", 0.1),
    )
    lidar.initialize()
    return lidar
