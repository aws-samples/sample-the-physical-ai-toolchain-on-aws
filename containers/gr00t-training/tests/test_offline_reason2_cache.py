#!/usr/bin/env python3
"""Regression checks for offline Cosmos Reason2 preparation in GR00T Batch."""

from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "batch-train-entrypoint-n17.sh"
REVISION = "9ce19a195e423419c349abfc86fd07178b230561"


class OfflineReason2PreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text()

    def test_pinned_snapshot_is_seeded_under_canonical_hf_cache_key(self) -> None:
        self.assertIn(f'REASON2_REVISION="{REVISION}"', self.source)
        self.assertIn(
            'models--nvidia--Cosmos-Reason2-2B',
            self.source,
        )
        self.assertIn(
            'REASON2_SNAPSHOT_DIR="${REASON2_CACHE_DIR}/snapshots/${REASON2_REVISION}"',
            self.source,
        )

    def test_main_ref_has_no_newline_and_is_byte_checked(self) -> None:
        self.assertRegex(
            self.source,
            re.compile(
                r"printf '%s' \"\$REASON2_REVISION\" > "
                r'\"\$\{REASON2_CACHE_DIR\}/refs/main\"'
            ),
        )
        self.assertIn('wc -c < "${REASON2_CACHE_DIR}/refs/main"', self.source)
        self.assertNotIn("printf '%s\\n' \"$REASON2_REVISION\"", self.source)

    def test_canonical_backbone_discriminator_is_not_rewritten(self) -> None:
        preparation = self.source.split("# --- Download dataset from S3 ---", 1)[0]
        self.assertIn('canonical = "nvidia/Cosmos-Reason2-2B"', preparation)
        self.assertNotIn("value.replace(source, reason)", preparation)
        self.assertIn("Preserved canonical Cosmos Reason2 identity", preparation)

    def test_only_processor_loading_is_redirected_to_local_snapshot(self) -> None:
        preparation = self.source.split("# --- Download dataset from S3 ---", 1)[0]
        self.assertIn("processing_gr00t_n1d7.py", preparation)
        self.assertIn(
            'needle = "return Qwen3VLProcessor.from_pretrained(model_name, '
            '**transformers_loading_kwargs)"',
            preparation,
        )
        self.assertIn("Redirected Qwen3-VL processor loading", preparation)
        self.assertIn("local_snapshot!r", preparation)


if __name__ == "__main__":
    unittest.main()
