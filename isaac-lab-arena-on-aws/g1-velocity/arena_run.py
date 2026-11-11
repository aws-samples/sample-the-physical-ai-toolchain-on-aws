# SPDX-License-Identifier: MIT-0
"""Run an unmodified Isaac Lab Arena runner with the workshop's G1 velocity environment.

    /isaac-sim/python.sh arena_run.py experiment_runner --headless \
        --experiment_config g1_velocity_experiment.yaml \
        shared.policy.checkpoint_path=<run dir>/model_<N>.pt

Arena imports its own environments lazily, after the Isaac Sim app starts. This launcher
wraps that hook so ``arena_g1_velocity`` is imported at the same moment, which registers
``g1_flat_velocity`` before the Experiment YAML is resolved.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab_arena.evaluation.arena_experiment_config_loader as config_loader  # noqa: E402
import isaaclab_arena_environments.cli as environments_cli  # noqa: E402

_arena_ensure_environments_registered = environments_cli.ensure_environments_registered


def _ensure_environments_registered():
    _arena_ensure_environments_registered()
    import arena_g1_velocity  # noqa: F401


environments_cli.ensure_environments_registered = _ensure_environments_registered
config_loader.ensure_environments_registered = _ensure_environments_registered

RUNNERS = {
    "experiment_runner": "isaaclab_arena.evaluation.experiment_runner",
    "policy_runner": "isaaclab_arena.evaluation.policy_runner",
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in RUNNERS:
        sys.exit(f"usage: arena_run.py {{{'|'.join(RUNNERS)}}} [runner arguments]")
    runner = __import__(RUNNERS[sys.argv[1]], fromlist=["main"])
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    runner.main()
