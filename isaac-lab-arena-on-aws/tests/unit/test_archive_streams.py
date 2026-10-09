"""Exercise the real SDK stream over localhost; no AWS resources or credentials."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import socket
import threading
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, IncompleteReadError, ResponseStreamingError

SOURCE = Path(__file__).resolve().parents[2] / "scripts/resources/archive.py"
SPEC = importlib.util.spec_from_file_location("archive_under_test", SOURCE)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)
DATA = bytes(range(256)) * (96 * 1024)  # Three full 8 MiB chunks, with a known digest.
VERSION = "immutable-local-version"


class ObjectHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        requested = self.headers.get("Range")
        self.server.requests.append({"range": requested, "version": query.get("versionId"),
                                     "owner": self.headers.get("x-amz-expected-bucket-owner")})
        mode = self.server.mode
        if mode == "denied":
            body = b"<Error><Code>AccessDenied</Code><Message>Test denial</Message></Error>"
            self.send_response(403)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        start = int(requested.split("=")[1].split("-")[0]) if requested else 0
        if mode == "ignore-range":
            start = 0
        size = len(DATA) - start
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(size - 1 if mode == "wrong-size" else size))
        self.send_header("x-amz-version-id", "another-version" if mode == "wrong-version" else VERSION)
        if start:
            range_start = start + 1 if mode == "wrong-range" else start
            self.send_header("Content-Range", f"bytes {range_start}-{len(DATA) - 1}/{len(DATA)}")
        self.end_headers()
        try:
            if mode == "always-break":
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            if len(self.server.requests) == 1 and mode in {"break-once", "ignore-range", "wrong-range"}:
                self.wfile.write(DATA[:8 * 1024 * 1024 + 17])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            self.wfile.write(DATA[start:])
        except (BrokenPipeError, ConnectionResetError):
            pass


class ArchiveStreamTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ObjectHandler)
        self.server.mode, self.server.requests = "complete", []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = boto3.Session().client(
            "s3", endpoint_url=f"http://127.0.0.1:{self.server.server_port}", region_name="us-east-1",
            config=Config(signature_version=UNSIGNED, retries={"max_attempts": 0},
                          connect_timeout=1, read_timeout=1))
        self.wait = patch.object(archive.time, "sleep")
        self.wait.start()

    def tearDown(self):
        self.wait.stop()
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def digest(self):
        return archive.digest(self.client, "123456789012", "test-bucket", "checkpoint", VERSION, len(DATA))

    def test_interrupted_stream_resumes_without_missing_or_repeating_bytes(self):
        self.server.mode = "break-once"
        self.assertEqual(self.digest(), hashlib.sha256(DATA).hexdigest())
        self.assertEqual(len(self.server.requests), 2)
        self.assertIsNone(self.server.requests[0]["range"])
        offset = int(self.server.requests[1]["range"].split("=")[1].split("-")[0])
        # SDKs can yield the short final fragment or raise before delivering it.
        # Resume only the undelivered part; the complete digest above must agree.
        self.assertIn(offset, {8 * 1024 * 1024, 8 * 1024 * 1024 + 17})
        self.assertTrue(all(r["version"] == [VERSION] and r["owner"] == "123456789012"
                            for r in self.server.requests))

    def test_resumed_stream_must_honor_the_requested_range(self):
        self.server.mode = "ignore-range"
        with self.assertRaisesRegex(ValueError, "Version or size|byte range"):
            self.digest()
        self.assertEqual(len(self.server.requests), 2)

    def test_resumed_stream_cannot_claim_a_different_start_byte(self):
        self.server.mode = "wrong-range"
        with self.assertRaisesRegex(ValueError, "byte range"):
            self.digest()
        self.assertEqual(len(self.server.requests), 2)

    def test_another_version_is_not_retried_or_accepted(self):
        self.server.mode = "wrong-version"
        with self.assertRaisesRegex(ValueError, "Version or size"):
            self.digest()
        self.assertEqual(len(self.server.requests), 1)

    def test_wrong_size_is_not_retried_or_accepted(self):
        self.server.mode = "wrong-size"
        with self.assertRaisesRegex(ValueError, "Version or size"):
            self.digest()
        self.assertEqual(len(self.server.requests), 1)

    def test_repeated_stream_failures_stop_after_three_retries(self):
        self.server.mode = "always-break"
        with self.assertRaises((IncompleteReadError, ResponseStreamingError)):
            self.digest()
        self.assertEqual(len(self.server.requests), 4)

    def test_access_denied_is_not_treated_as_a_stream_interruption(self):
        self.server.mode = "denied"
        with self.assertRaises(ClientError):
            self.digest()
        self.assertEqual(len(self.server.requests), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
