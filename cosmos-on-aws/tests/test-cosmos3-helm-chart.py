#!/usr/bin/env python3
"""Contract tests for the private Cosmos 3 Helm chart."""

from pathlib import Path
import os
import shutil
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
CHART = HERE.parent / "charts" / "cosmos3"
EXPECTED_SHA256 = "a0660a0494018fff848fda34ecf0e8a823a360502a0254b5f5068f2e1c7f82ff"
HELM = os.environ.get("HELM_BIN") or shutil.which("helm")


class Cosmos3HelmChartTest(unittest.TestCase):
    def helm(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        if HELM is None:
            self.skipTest("Helm is not installed; static chart checks still run")
        return subprocess.run(
            [HELM, *arguments], check=check, capture_output=True, text=True
        )

    @staticmethod
    def value_arguments() -> list[str]:
        return [
            "--set-string", "aws.region=us-east-2",
            "--set-string", "image.ref=vllm/vllm-omni:cosmos3",
            "--set-string", "model.archiveS3Uri=s3://physical-ai-data-123456789012/cosmos3-assets/archives/cosmos3-nano.tar",
            "--set-string", f"model.archiveSha256={EXPECTED_SHA256}",
        ]

    def test_static_templates_cannot_create_public_access_or_credentials(self) -> None:
        templates = "\n".join(
            path.read_text() for path in sorted((CHART / "templates").glob("*"))
        )
        self.assertIn("type: ClusterIP", templates)
        self.assertNotIn("type: LoadBalancer", templates)
        self.assertNotIn("type: NodePort", templates)
        self.assertNotIn("kind: Ingress", templates)
        self.assertNotIn("kind: Secret", templates)
        self.assertNotIn("kind: ServiceAccount", templates)
        self.assertIn("serviceAccountName:", templates)
        self.assertIn("app:", templates)

    def test_chart_lints_and_renders_validated_nano_service(self) -> None:
        lint = self.helm("lint", str(CHART), *self.value_arguments())
        self.assertIn("0 chart(s) failed", lint.stdout)

        rendered = self.helm(
            "template", "cosmos3-g1-server", str(CHART), *self.value_arguments()
        ).stdout
        self.assertEqual(rendered.count("kind: ConfigMap"), 1)
        self.assertEqual(rendered.count("kind: Deployment"), 1)
        self.assertEqual(rendered.count("kind: Service"), 1)
        self.assertIn('app: "cosmos3-g1-server"', rendered)
        self.assertIn('image: "vllm/vllm-omni:cosmos3"', rendered)
        self.assertEqual(rendered.count('nvidia.com/gpu: "1"'), 2)
        self.assertIn("type: ClusterIP", rendered)
        self.assertIn("name: HF_HUB_OFFLINE", rendered)
        self.assertNotIn("kind: Ingress", rendered)
        self.assertNotIn("type: LoadBalancer", rendered)
        self.assertNotIn("type: NodePort", rendered)

    def test_chart_fails_closed_without_archive_values(self) -> None:
        result = self.helm(
            "template", "cosmos3-g1-server", str(CHART),
            "--set-string", "aws.region=us-east-2",
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model.archiveS3Uri is required", result.stderr)

    def test_chart_rejects_unvalidated_archive_checksum(self) -> None:
        arguments = self.value_arguments()
        arguments[-1] = "model.archiveSha256=" + ("0" * 64)
        result = self.helm(
            "template", "cosmos3-g1-server", str(CHART), *arguments, check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("validated Cosmos archive checksum", result.stderr)


if __name__ == "__main__":
    unittest.main()
