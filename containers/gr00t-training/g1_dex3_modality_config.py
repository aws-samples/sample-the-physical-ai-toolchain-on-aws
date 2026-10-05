"""
GR00T modality config for the Unitree G1 Edu + Dex3-1 hands (NEW_EMBODIMENT).

This is the config the N1.7 training job loads via `--modality-config-path`.
`train_entrypoint_n17.py` also keeps an inline copy so the container stays
self-contained (same two-copy convention as `ur3_modality_config.py`) — keep the
two in sync.

Dataset: `pick_place_data`, LeRobot v2.1, `robot_type = "Unitree_G1_Dex3"`,
201 episodes, 30 fps, single camera, 28-DOF joint-space teleop recording.
No conversion is applied: the N1.7 loader reads v2.1 natively and the only
GR00T-specific artefact added is `meta/modality.json`
(written by `training/gr00t/prepare_g1_dex3_dataset.py`).

Index map — `observation.state` and `action` are both float32[28] and carry the
SAME joint ordering (verified from the `names` arrays in `meta/info.json`):

    [0:7]   left_arm    kLeftShoulderPitch, kLeftShoulderRoll, kLeftShoulderYaw,
                        kLeftElbow, kLeftWristRoll, kLeftWristPitch, kLeftWristYaw
    [7:14]  right_arm   kRightShoulderPitch, kRightShoulderRoll, kRightShoulderYaw,
                        kRightElbow, kRightWristRoll, kRightWristPitch, kRightWristYaw
    [14:21] left_hand   kLeftHandThumb0, kLeftHandThumb1, kLeftHandThumb2,
                        kLeftHandMiddle0, kLeftHandMiddle1, kLeftHandIndex0, kLeftHandIndex1
    [21:28] right_hand  kRightHandThumb0, kRightHandThumb1, kRightHandThumb2,
                        kRightHandIndex0, kRightHandIndex1, kRightHandMiddle0, kRightHandMiddle1

7 + 7 + 7 + 7 = 28. There is no waist, no legs and no mobile base in this
capture. The right hand's finger ordering differs from the left (Index before
Middle) — that is how the data was recorded; do not "fix" it, the groups are
sliced contiguously either way.

The group names here MUST match the group names in the dataset's
`meta/modality.json`, in BOTH its `state` and `action` sections:
  * `_extract_joint_groups` (lerobot_episode_loader.py) looks each
    `modality_keys` entry up in `modality.json[modality_type]`; a name that is
    missing is only a printed warning, so a typo silently drops a joint group.
  * A RELATIVE action resolves its reference state as
    `state_key = action_config.state_key if action_config.state_key else key`
    followed by `assert state_key in state`
    (state_action_processor.py:377-386). We leave `state_key=None` and rely on
    the state/action group names being identical.

`ActionType.NON_EEF` is MANDATORY for every group here. `ActionType.EEF` routes
through `EndEffectorPose.from_action_format`, which for `XYZ_ROTVEC` does
`translation=data[:3], rotation=data[3:]` and hands the remainder to
`Rotation.from_rotvec` — that requires exactly 3 rotation components and raises
inside scipy on a 7-element joint group. `NON_EEF` routes to `JointPose`, which
is plain elementwise arithmetic over however many joints the group has. This is
also the semantically correct choice: the recording is joint-space position
targets, not a Cartesian end-effector pose (mean |action - state| = 0.05 rad on
episode 0).

Representation per group follows upstream's own
`unitree_g1_full_body_with_waist_height_nav_cmd`
(gr00t/configs/data/embodiment_configs.py), minus the waist/legs/nav groups we
do not have:
  * arms  -> RELATIVE: the model predicts joint deltas from the measured joint
    positions. `generate_rel_stats` produces `meta/relative_stats.json` for
    exactly these groups inside the container.
  * hands -> ABSOLUTE: upstream treats the G1 hand like a gripper
    ("G1 hand is controlled by binary signals like a gripper").

Embodiment tag is NEW_EMBODIMENT (projector slot 10, the documented "finetune
placeholder" head — see getting_started/finetune_new_embodiment.md). The
pretrained `REAL_G1` tag expects EEF 9D wrist poses + a waist group + two video
timesteps, none of which exist here; `UNITREE_G1` is a POSTTRAIN tag and is
already present in MODALITY_CONFIGS, so registering over it trips
`register_modality_config`'s own assertion. Training therefore starts from a
fresh state/action projector on top of the pretrained backbone, so the initial
loss will be higher than an in-distribution embodiment's.

Action horizon is 16 (`delta_indices = range(0, 16)`), matching the shipped UR3
config. N1.7 allows up to 40 (`max_action_horizon` in the base model's
`processor_config.json`) and the shortest episode is 129 frames, so raising it
is possible — it is kept at 16 to stay comparable with the validated runs.

For a different G1 capture, change the group names/order here AND in the
dataset's `meta/modality.json` (via `prepare_g1_dex3_dataset.py`, which derives
the ranges from `info.json` and asserts them), and keep the tag NEW_EMBODIMENT.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.data.embodiment_tags import EmbodimentTag


g1_dex3_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        action_configs=[
            # left_arm [0:7] — joint-space deltas from the measured joint positions.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm [7:14] — same.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand [14:21] — Dex3-1 absolute targets, treated like a gripper.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand [21:28] — same.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_dex3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
