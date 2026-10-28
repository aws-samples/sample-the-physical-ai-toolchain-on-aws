import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTAINER_DIR = ROOT / "containers" / "isaac-lab"


class IsaacLab3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dockerfile = (CONTAINER_DIR / "Dockerfile").read_text()
        cls.batch_entrypoint = (CONTAINER_DIR / "batch-train-entrypoint.sh").read_text()
        cls.sagemaker_entrypoint = (CONTAINER_DIR / "sm-train-entrypoint.sh").read_text()
        cls.python_entrypoint = (CONTAINER_DIR / "train_entrypoint.py").read_text()
        cls.buildspec = (CONTAINER_DIR / "buildspec.yml").read_text()
        cls.launcher = (ROOT / "training" / "scripts" / "launch_rl.py").read_text()
        cls.gui_guide = (ROOT / "isaac-sim-on-aws" / "gui-mode-guide.md").read_text()

    def test_container_uses_validated_isaac_lab_3_base(self):
        self.assertIn("FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2", self.dockerfile)
        self.assertNotIn("FROM nvcr.io/nvidia/isaac-lab:2.1.0", self.dockerfile)
        self.assertIn("/workspace/isaaclab/_isaac_sim/kit/python/bin/python3", self.dockerfile)

    def test_entrypoints_do_not_use_removed_isaac_sim_python_path(self):
        self.assertNotIn("/isaac-sim/python.sh", self.batch_entrypoint)
        self.assertNotIn("/isaac-sim/python.sh", self.sagemaker_entrypoint)
        self.assertIn('ISAAC_LAB_ROOT="/workspace/isaaclab"', self.batch_entrypoint)
        self.assertIn('ISAAC_LAB_PYTHON="$ISAAC_LAB_ROOT/_isaac_sim/python.sh"', self.batch_entrypoint)
        self.assertIn('ISAAC_LAB_ROOT="/workspace/isaaclab"', self.sagemaker_entrypoint)
        self.assertIn('ISAAC_LAB_PYTHON="$ISAAC_LAB_ROOT/_isaac_sim/python.sh"', self.sagemaker_entrypoint)

    def test_batch_uses_validated_single_gpu_g1_contract(self):
        self.assertIn('TASK="${TASK:-Isaac-Velocity-Flat-G1-v0}"', self.batch_entrypoint)
        self.assertIn("NPROC=${PROC_PER_NODE:-1}", self.batch_entrypoint)
        self.assertIn('export OUTPUT_DIR="/tmp/checkpoints/${JOB_ID}"', self.batch_entrypoint)
        self.assertIn('"$ISAAC_LAB_ROOT/isaaclab.sh" -p', self.batch_entrypoint)
        self.assertIn("seen_files = set()", self.batch_entrypoint)

    def test_sagemaker_python_entrypoint_uses_absolute_training_path(self):
        self.assertIn(
            '"/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py"',
            self.python_entrypoint,
        )
        self.assertIn('env["PRIVACY_CONSENT"] = "Y"', self.python_entrypoint)

    def test_build_and_launcher_use_versioned_g1_tag(self):
        tag = "3.0.0-beta2-g1"
        self.assertIn(f"$ECR_REPO_URI:{tag}", self.buildspec)
        self.assertIn(f"isaac-lab:{tag}", self.launcher)
        self.assertIn('default="Isaac-Velocity-Flat-G1-v0"', self.launcher)

    def test_gui_guide_uses_unified_kit_visualizer(self):
        self.assertIn("./isaaclab.sh play", self.gui_guide)
        self.assertIn("--rl_library rsl_rl", self.gui_guide)
        self.assertIn("--viz kit", self.gui_guide)

    def test_shell_entrypoints_parse(self):
        for script in ("batch-train-entrypoint.sh", "sm-train-entrypoint.sh"):
            result = subprocess.run(
                ["bash", "-n", str(CONTAINER_DIR / script)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
