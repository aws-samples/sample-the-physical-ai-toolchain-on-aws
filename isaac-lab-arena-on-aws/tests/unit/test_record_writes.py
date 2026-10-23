"""Actual concurrent status writers; no AWS access or service fixtures."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("arena_support", ROOT / "scripts/resources/support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class StatusRecordTests(unittest.TestCase):
    def test_observers_save_complete_status_records_concurrently(self):
        """Concurrent observers can save/read status without torn JSON or lost temporary files."""
        parent = ROOT / "local-dev"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="record-writes-", dir=parent) as folder:
            path = Path(folder) / "status.json"
            support.save(path, {"writer": -1})
            ready = threading.Barrier(6)

            def writer(number):
                ready.wait(timeout=10)
                for step in range(100):
                    value = {"writer": number, "step": step, "content": str(number) * 10000}
                    if number % 2:
                        support.save_bytes(path, json.dumps(value).encode())
                    else:
                        support.save(path, value)
                    actual = support.load(path)
                    self.assertEqual(actual["content"], str(actual["writer"]) * 10000)

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(writer, number) for number in range(6)]
                for future in futures:
                    future.result(timeout=30)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_downloaded_report_keeps_its_exact_bytes(self):
        """Atomic report saving preserves the bytes checked against the S3 receipt hash."""
        parent = ROOT / "local-dev"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="record-writes-", dir=parent) as folder:
            path = Path(folder) / "report.json"
            body = b'{"message":"original spacing", "value": 1.0}\n'
            support.save_bytes(path, body)
            self.assertEqual(path.read_bytes(), body)

    def test_failed_serialization_preserves_previous_status(self):
        """A bad replacement leaves the previous readable record and no temporary file."""
        parent = ROOT / "local-dev"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="record-writes-", dir=parent) as folder:
            path = Path(folder) / "status.json"
            original = {"state": "last known good"}
            support.save(path, original)
            circular = {}
            circular["self"] = circular
            with self.assertRaises(ValueError):
                support.save(path, circular)
            self.assertEqual(support.load(path), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == "__main__":
    unittest.main(verbosity=2)
