#!/usr/bin/env python3
"""Tests for prepare-g1-service.py; no AWS account is used."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "prepare-g1-service.py"
MANIFEST = HERE.parent / "cosmos3-service.yaml"
SPEC = spec_from_file_location("prepare_g1_service", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareG1ServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = {
            "EksClusterName": "physical-ai-dev-cosmos3",
            "CosmosRuntimeImageUri": "vllm/vllm-omni:cosmos3",
            "Cosmos3ArchiveS3Uri": (
                "s3://physical-ai-data-123456789012/"
                "cosmos3-assets/archives/cosmos3-nano.tar"
            ),
            "Cosmos3ArchiveSha256": MODULE.EXPECTED_ARCHIVE_SHA256,
        }

    def test_renders_private_reusable_nano_service(self) -> None:
        rendered = MODULE.render_service(
            MANIFEST.read_text(), "us-east-2", self.outputs
        )
        self.assertIn("kind: Deployment", rendered)
        self.assertIn("name: cosmos3-g1-server", rendered)
        self.assertIn("replicas: 1", rendered)
        self.assertIn("type: Recreate", rendered)
        self.assertIn('name: COSMOS_MODEL, value: "nano"', rendered)
        self.assertEqual(rendered.count('nvidia.com/gpu: "1"'), 2)
        self.assertIn("startupProbe:", rendered)
        self.assertIn("awscli==1.46.2", rendered)
        self.assertIn("kind: Service", rendered)
        self.assertIn("type: ClusterIP", rendered)
        self.assertNotIn("type: LoadBalancer", rendered)
        self.assertNotIn("type: NodePort", rendered)
        self.assertNotIn("kind: Job", rendered)
        self.assertNotIn("<REGION>", rendered)

    def test_allows_distinct_valid_names(self) -> None:
        rendered = MODULE.render_service(
            MANIFEST.read_text(),
            "us-east-2",
            self.outputs,
            deployment_name="cosmos3-server-v2",
            service_name="cosmos3-api-v2",
        )
        self.assertIn("name: cosmos3-server-v2", rendered)
        self.assertIn("name: cosmos3-api-v2", rendered)

    def test_rejects_public_or_stale_configuration(self) -> None:
        public = MANIFEST.read_text().replace("type: ClusterIP", "type: LoadBalancer")
        with self.assertRaisesRegex(ValueError, "ClusterIP only"):
            MODULE.render_service(public, "us-east-2", self.outputs)
        self.outputs["Cosmos3ArchiveSha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stale"):
            MODULE.render_service(MANIFEST.read_text(), "us-east-2", self.outputs)

    def test_rejects_invalid_resource_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Kubernetes DNS label"):
            MODULE.render_service(
                MANIFEST.read_text(),
                "us-east-2",
                self.outputs,
                deployment_name="Invalid_Name",
            )


if __name__ == "__main__":
    unittest.main()
