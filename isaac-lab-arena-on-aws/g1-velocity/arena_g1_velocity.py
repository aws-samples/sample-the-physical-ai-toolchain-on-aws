# SPDX-License-Identifier: MIT-0
"""Isaac Lab Arena environment for the Unitree G1 flat-terrain velocity-tracking policy.

Registers the ``g1_flat_velocity`` environment with Isaac Lab Arena ``release/0.3.0``
without changing Arena's source. The embodiment reuses Isaac Lab's own
``Isaac-Velocity-Flat-G1-v0`` scene, observation, action and command definitions, so a
policy trained with that task sees the same observations and writes the same joint
targets. The task adds Arena metrics:

* ``success_rate``: the robot did not fall and its episode-mean velocity-tracking error
  stayed below ``vel_xy_success_threshold`` (m/s) and ``vel_yaw_success_threshold`` (rad/s).
* ``fall_rate``: the episode ended because the torso touched the ground.
* ``mean_lin_vel_xy_error`` and ``mean_ang_vel_z_error``: episode-mean tracking errors.

Import this module only after the Isaac Sim app has started; ``arena_run.py`` does that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

COMMAND_NAME = "base_velocity"
FALL_TERM_NAME = "base_contact"


# ---------------------------------------------------------------------------------------
# Configuration: every key here can be set under ``environment:`` in the Experiment YAML.
# ---------------------------------------------------------------------------------------
@dataclass
class G1FlatVelocityEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the G1 flat-terrain velocity-tracking evaluation."""

    episode_length_s: float = 10.0
    """Episode length in simulated seconds. Training used 20 s."""
    lin_vel_x_min: float = 0.0
    lin_vel_x_max: float = 1.0
    lin_vel_y_min: float = -0.5
    lin_vel_y_max: float = 0.5
    ang_vel_z_min: float = -1.0
    ang_vel_z_max: float = 1.0
    rel_standing_envs: float = 0.02
    """Fraction of robots given a zero command."""
    heading_command: bool = True
    """Derive the yaw-rate command from a random heading target, as in training."""
    vel_xy_success_threshold: float = 0.5
    """Success needs an episode-mean linear-velocity error below this value [m/s]."""
    vel_yaw_success_threshold: float = 0.8
    """Success needs an episode-mean yaw-rate error below this value [rad/s]."""
    observation_noise: bool = False
    """Keep the training observation noise. Off by default for a deterministic score."""
    random_pushes: bool = False
    """Keep the training-time random pushes. Off by default."""


# ---------------------------------------------------------------------------------------
# Isaac Lab task configuration, shared by the embodiment and the task.
# ---------------------------------------------------------------------------------------
def _resolve_preset(cfg: Any) -> Any:
    """Return the PhysX ``default`` of an Isaac Lab ``PresetCfg``; pass other configs through."""
    from isaaclab_tasks.utils import PresetCfg

    return cfg.default if isinstance(cfg, PresetCfg) else cfg


def _isaaclab_g1_flat_cfg():
    """Build Isaac Lab's ``Isaac-Velocity-Flat-G1-v0`` environment config."""
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    return G1FlatEnvCfg()


# ---------------------------------------------------------------------------------------
# Embodiment: robot, sensors, ground, actions, observations and reset events.
# ---------------------------------------------------------------------------------------
class G1FlatVelocityEmbodiment(EmbodimentBase):
    """Unitree G1 with Isaac Lab's flat velocity-task scene, actions and observations."""

    name = "g1_flat_velocity"

    def __init__(self, isaaclab_cfg: Any, observation_noise: bool, random_pushes: bool):
        super().__init__()
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.terrains import TerrainImporterCfg

        from isaaclab_arena.utils.configclass import make_configclass

        scene = isaaclab_cfg.scene
        contact_forces = _resolve_preset(scene.contact_forces)
        contact_forces.update_period = isaaclab_cfg.sim.dt
        scene_cfg_type = make_configclass(
            "G1FlatVelocitySceneCfg",
            [
                ("terrain", TerrainImporterCfg, scene.terrain),
                ("robot", ArticulationCfg, scene.robot),
                ("contact_forces", type(contact_forces), contact_forces),
                ("sky_light", AssetBaseCfg, scene.sky_light),
            ],
        )
        self.scene_config = scene_cfg_type()
        self.action_config = isaaclab_cfg.actions
        self.observation_config = isaaclab_cfg.observations
        self.observation_config.policy.enable_corruption = observation_noise

        events = isaaclab_cfg.events
        if not random_pushes:
            events.push_robot = None
            events.base_external_force_torque = None
        self.event_config = events


# ---------------------------------------------------------------------------------------
# Success termination and per-episode recorders.
# ---------------------------------------------------------------------------------------
def _episode_mean_errors(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Episode-mean xy and yaw tracking errors from the velocity command's running sums."""
    term = env.command_manager.get_term(COMMAND_NAME)
    steps = term._step_count.clamp_min(1.0)
    return term._error_xy_sum / steps, term._error_yaw_sum / steps


def velocity_tracking_success(env, xy_threshold: float, yaw_threshold: float) -> torch.Tensor:
    """True on the last step of an episode in which the robot stayed up and tracked well."""
    at_end = env.episode_length_buf >= env.max_episode_length
    fell = env.termination_manager.get_term(FALL_TERM_NAME)
    mean_xy, mean_yaw = _episode_mean_errors(env)
    return at_end & ~fell & (mean_xy < xy_threshold) & (mean_yaw < yaw_threshold)


class _PreResetRecorder(RecorderTerm):
    """Record one value per finished episode, skipping the initial reset of all envs."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.first_reset = True

    def value(self, env_ids) -> torch.Tensor:
        raise NotImplementedError

    def record_pre_reset(self, env_ids):
        if self.first_reset:
            self.first_reset = False
            return None, None
        return self.name, self.value(env_ids)


class FallRecorder(_PreResetRecorder):
    def value(self, env_ids):
        return self._env.termination_manager.get_term(FALL_TERM_NAME)[env_ids].clone()


class LinVelErrorRecorder(_PreResetRecorder):
    def value(self, env_ids):
        return _episode_mean_errors(self._env)[0][env_ids].clone()


class AngVelErrorRecorder(_PreResetRecorder):
    def value(self, env_ids):
        return _episode_mean_errors(self._env)[1][env_ids].clone()


@configclass
class FallRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = FallRecorder
    name: str = "fell"


@configclass
class LinVelErrorRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = LinVelErrorRecorder
    name: str = "lin_vel_xy_error"


@configclass
class AngVelErrorRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = AngVelErrorRecorder
    name: str = "ang_vel_z_error"


def compute_episode_mean(recorded_metric_data: list[np.ndarray]) -> float:
    """Mean of one recorded value per episode."""
    if not recorded_metric_data:
        return 0.0
    values = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in recorded_metric_data])
    return float(np.mean(values))


class _EpisodeMeanMetric(MetricBase):
    recorder_cfg_type: type[RecorderTermCfg]

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return self.recorder_cfg_type(name=self.recorder_term_name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_episode_mean, params={}, recorder_term_name=self.recorder_term_name
        )


class FallRateMetric(_EpisodeMeanMetric):
    name = "fall_rate"
    recorder_term_name = "fell"
    recorder_cfg_type = FallRecorderCfg


class LinVelErrorMetric(_EpisodeMeanMetric):
    name = "mean_lin_vel_xy_error"
    recorder_term_name = "lin_vel_xy_error"
    recorder_cfg_type = LinVelErrorRecorderCfg


class AngVelErrorMetric(_EpisodeMeanMetric):
    name = "mean_ang_vel_z_error"
    recorder_term_name = "ang_vel_z_error"
    recorder_cfg_type = AngVelErrorRecorderCfg


# ---------------------------------------------------------------------------------------
# Per-episode record: written to each run's episode_results JSONL and the Experiment JSON.
# ---------------------------------------------------------------------------------------
def record_velocity_tracking(env, env_id: int) -> dict[str, Any]:
    """Record the command, fall flag and episode-mean tracking errors of one finished episode."""
    term = env.command_manager.get_term(COMMAND_NAME)
    mean_xy, mean_yaw = _episode_mean_errors(env)
    command = term.vel_command_b[env_id].tolist()
    return {
        "fell": bool(env.termination_manager.get_term(FALL_TERM_NAME)[env_id].item()),
        "lin_vel_xy_error": round(float(mean_xy[env_id].item()), 4),
        "ang_vel_z_error": round(float(mean_yaw[env_id].item()), 4),
        "command_lin_vel_x": round(command[0], 3),
        "command_lin_vel_y": round(command[1], 3),
        "command_ang_vel_z": round(command[2], 3),
    }


def _velocity_tracking_episode_term():
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    return EpisodeRecorderTermCfg(func=record_velocity_tracking)


# ---------------------------------------------------------------------------------------
# Task: velocity commands, terminations and metrics.
# ---------------------------------------------------------------------------------------
class G1VelocityTrackingTask(TaskBase):
    """Follow random base-velocity commands without falling."""

    def __init__(self, isaaclab_cfg: Any, cfg: G1FlatVelocityEnvironmentCfg):
        super().__init__(
            episode_length_s=cfg.episode_length_s,
            task_description="Follow the commanded base velocity without falling.",
        )
        from isaaclab.managers import TerminationTermCfg

        from isaaclab_arena.utils.configclass import make_configclass

        commands = isaaclab_cfg.commands
        command = commands.base_velocity
        command.ranges.lin_vel_x = (cfg.lin_vel_x_min, cfg.lin_vel_x_max)
        command.ranges.lin_vel_y = (cfg.lin_vel_y_min, cfg.lin_vel_y_max)
        command.ranges.ang_vel_z = (cfg.ang_vel_z_min, cfg.ang_vel_z_max)
        command.rel_standing_envs = cfg.rel_standing_envs
        command.heading_command = cfg.heading_command
        command.vel_xy_success_threshold = cfg.vel_xy_success_threshold
        command.vel_yaw_success_threshold = cfg.vel_yaw_success_threshold
        # One command for the whole episode, so every episode scores one command.
        command.resampling_time_range = (cfg.episode_length_s, cfg.episode_length_s)
        self.commands_cfg = commands

        terminations = isaaclab_cfg.terminations
        success = TerminationTermCfg(
            func=velocity_tracking_success,
            params={"xy_threshold": cfg.vel_xy_success_threshold, "yaw_threshold": cfg.vel_yaw_success_threshold},
        )
        termination_type = make_configclass(
            "G1VelocityTerminationsCfg",
            [
                ("time_out", TerminationTermCfg, terminations.time_out),
                (FALL_TERM_NAME, TerminationTermCfg, terminations.base_contact),
                ("success", TerminationTermCfg, success),
            ],
        )
        self.termination_cfg = termination_type()

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_commands_cfg(self):
        return self.commands_cfg

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), FallRateMetric(), LinVelErrorMetric(), AngVelErrorMetric()]

    def get_viewer_cfg(self):
        from isaaclab.envs.common import ViewerCfg

        return ViewerCfg(eye=(9.0, 9.0, 6.0), lookat=(0.0, 0.0, 0.5))


# ---------------------------------------------------------------------------------------
# Environment factory, selected with ``environment.type: g1_flat_velocity``.
# ---------------------------------------------------------------------------------------
def _match_training_timing(env_cfg):
    """Use the training task's physics and control rate: dt 0.005 s, decimation 4 (50 Hz)."""
    isaaclab_cfg = _isaaclab_g1_flat_cfg()
    env_cfg.decimation = isaaclab_cfg.decimation
    env_cfg.sim.dt = isaaclab_cfg.sim.dt
    env_cfg.sim.render_interval = isaaclab_cfg.decimation
    env_cfg.sim.physics = _resolve_preset(isaaclab_cfg.sim.physics)
    env_cfg.sim.physics_material = isaaclab_cfg.scene.terrain.physics_material
    return env_cfg


@register_environment
class G1FlatVelocityEnvironment(ArenaEnvironmentFactory[G1FlatVelocityEnvironmentCfg]):
    name: str = "g1_flat_velocity"

    def build(self, cfg: G1FlatVelocityEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        isaaclab_cfg = _isaaclab_g1_flat_cfg()
        embodiment = G1FlatVelocityEmbodiment(isaaclab_cfg, cfg.observation_noise, cfg.random_pushes)
        task = G1VelocityTrackingTask(isaaclab_cfg, cfg)
        return IsaacLabArenaEnvironment(
            name=self.name,
            scene=Scene(assets=[]),
            embodiment=embodiment,
            task=task,
            env_cfg_callback=_match_training_timing,
            episode_recorder_terms={"velocity_tracking": _velocity_tracking_episode_term()},
        )
