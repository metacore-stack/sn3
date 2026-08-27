"""Tests for the king downloader.

The interesting failures are all partial-transfer failures, so these tests run a
real HTTP server on localhost and interrupt it in the ways a 220 GB transfer
actually gets interrupted: a server that drops the connection, a server that
ignores Range, and a file whose bytes do not match the manifest.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import shutil
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

from validate_checkpoint.download import (
    DISK_MARGIN_BYTES,
    KingDownloader,
    human,
    sha256_file,
)
from validate_checkpoint.errors import KingUnavailableError, ValidationError
from validate_checkpoint.king import KingFile, KingReference


def make_payload(name: str, size: int) -> bytes:
    """Deterministic pseudo-random bytes, so hashes are stable across runs."""
    out = bytearray()
    block = 0
    while len(out) < size:
        out += hashlib.sha256(f"{name}:{block}".encode()).digest()
        block += 1
    return bytes(out[:size])


class RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``server.payloads`` with Range support and optional sabotage."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 - silence the test output
        pass

    def do_GET(self):  # noqa: N802
        name = self.path.lstrip("/")
        body = self.server.payloads.get(name)
        if body is None:
            self.send_error(404)
            return

        self.server.requests.append((name, self.headers.get("Range")))

        start = 0
        header = self.headers.get("Range")
        partial = False
        if header and self.server.honour_range:
            start = int(header.split("=", 1)[1].split("-", 1)[0])
            partial = True

        chunk = body[start:]
        # Sabotage: cut the response short to simulate a dropped connection.
        limit = self.server.truncate_after.get(name)
        if limit is not None:
            chunk = chunk[:limit]

        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(chunk)))
        if partial:
            self.send_header(
                "Content-Range", f"bytes {start}-{start + len(chunk) - 1}/{len(body)}"
            )
        self.end_headers()
        self.wfile.write(chunk)


class ServerFixture:
    def __init__(self, payloads: dict[str, bytes]):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        self.server.payloads = payloads
        self.server.truncate_after = {}
        self.server.honour_range = True
        self.server.requests = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class DownloadTestCase(unittest.TestCase):
    FILES = {"a.safetensors": 40_000, "b.safetensors": 25_000, "config.json": 512}

    def setUp(self):
        self.payloads = {n: make_payload(n, s) for n, s in self.FILES.items()}
        self.fixture = ServerFixture(self.payloads)
        self.addCleanup(self.fixture.stop)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.king = KingReference(
            digest="d" * 64,
            files={
                name: KingFile(
                    path=name,
                    sha256=hashlib.sha256(body).hexdigest(),
                    size=len(body),
                )
                for name, body in self.payloads.items()
            },
            source=self.fixture.url,
            model_name="test-king",
        )

    def downloader(self, **kwargs) -> KingDownloader:
        return KingDownloader(self.king, self.tmp / "king", timeout=10.0, **kwargs)

    # -- planning ----------------------------------------------------------

    def test_plan_reports_everything_missing_initially(self):
        plan = self.downloader().plan()
        self.assertEqual(len(plan.missing), 3)
        self.assertEqual(len(plan.present), 0)
        self.assertEqual(plan.bytes_needed, sum(self.FILES.values()))
        self.assertEqual(plan.total_bytes, sum(self.FILES.values()))

    def test_plan_counts_a_partial_file_as_resumable(self):
        d = self.downloader()
        d.destination.mkdir(parents=True, exist_ok=True)
        d.partial_path("a.safetensors").write_bytes(self.payloads["a.safetensors"][:10_000])
        plan = d.plan()
        self.assertEqual(plan.partial, {"a.safetensors": 10_000})
        self.assertEqual(plan.bytes_needed, sum(self.FILES.values()) - 10_000)

    def test_plan_discards_an_oversized_partial(self):
        d = self.downloader()
        d.destination.mkdir(parents=True, exist_ok=True)
        d.partial_path("a.safetensors").write_bytes(b"x" * 90_000)
        plan = d.plan()
        self.assertEqual(plan.partial, {})
        self.assertFalse(d.partial_path("a.safetensors").exists())

    def test_verify_present_deletes_a_corrupt_file_of_the_right_size(self):
        d = self.downloader()
        d.destination.mkdir(parents=True, exist_ok=True)
        target = d.local_path("a.safetensors")
        target.write_bytes(b"\x00" * self.FILES["a.safetensors"])

        # Same size, so a cheap plan trusts it.
        self.assertEqual(len(d.plan().present), 1)
        # Hashing catches it.
        plan = d.plan(verify_present=True)
        self.assertEqual(len(plan.present), 0)
        self.assertFalse(target.exists())

    def test_refuses_when_the_disk_is_too_small(self):
        d = self.downloader()
        d.plan()  # create the directory
        original = shutil.disk_usage

        def tiny(path):
            usage = original(path)
            return type(usage)(usage.total, usage.used, 1024)

        with unittest.mock.patch("validate_checkpoint.download.shutil.disk_usage", tiny):
            self.assertFalse(d.plan().enough_disk)
            with self.assertRaises(ValidationError) as ctx:
                d.fetch()
        self.assertIn("free", str(ctx.exception))

    def test_disk_margin_is_counted_on_top_of_the_transfer(self):
        d = self.downloader()
        d.plan()
        needed = sum(self.FILES.values())
        original = shutil.disk_usage

        def just_short(path):
            usage = original(path)
            return type(usage)(usage.total, usage.used, needed + DISK_MARGIN_BYTES - 1)

        with unittest.mock.patch(
            "validate_checkpoint.download.shutil.disk_usage", just_short
        ):
            self.assertFalse(d.plan().enough_disk)

    def test_local_directory_reference_is_rejected(self):
        local = KingReference(
            digest="", files=self.king.files, source=str(self.tmp)
        )
        with self.assertRaises(KingUnavailableError):
            KingDownloader(local, self.tmp / "out")

    def test_empty_reference_is_rejected(self):
        with self.assertRaises(KingUnavailableError):
            KingDownloader(KingReference(digest="x" * 64, source="http://x"), self.tmp)

    # -- fetching ----------------------------------------------------------

    def test_fetch_downloads_and_verifies_every_file(self):
        d = self.downloader()
        results = d.fetch()
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.complete for r in results))
        for name, body in self.payloads.items():
            self.assertEqual(d.local_path(name).read_bytes(), body)
        self.assertEqual(d.verify(), [])

    def test_fetch_leaves_no_part_files_behind(self):
        d = self.downloader()
        d.fetch()
        self.assertEqual(list(d.destination.glob("*.part")), [])

    def test_progress_callbacks_account_for_every_byte(self):
        d = self.downloader()
        seen = []
        files = []
        d.fetch(on_bytes=lambda done, total: seen.append((done, total)), on_file=files.append)
        self.assertEqual(seen[-1][0], sum(self.FILES.values()))
        self.assertEqual(seen[-1][1], sum(self.FILES.values()))
        self.assertEqual(len(files), 3)

    def test_second_fetch_is_a_no_op(self):
        d = self.downloader()
        d.fetch()
        before = len(self.fixture.server.requests)
        results = d.fetch()
        self.assertEqual(len(self.fixture.server.requests), before)
        self.assertTrue(all(r.complete for r in results))

    def test_resume_after_a_dropped_connection(self):
        """The scenario this module exists for: die mid-transfer, then continue."""
        self.fixture.server.truncate_after = {"a.safetensors": 12_000}
        d = self.downloader(workers=1)

        with self.assertRaises(ValidationError):
            d.fetch()
        # The partial survived; the completed files did too.
        self.assertTrue(d.partial_path("a.safetensors").is_file())
        self.assertEqual(d.partial_path("a.safetensors").stat().st_size, 12_000)
        self.assertTrue(d.local_path("b.safetensors").is_file())

        # Now the server behaves. Only the missing tail should be requested.
        self.fixture.server.truncate_after = {}
        self.fixture.server.requests.clear()
        results = d.fetch()

        ranges = dict(self.fixture.server.requests)
        self.assertEqual(ranges, {"a.safetensors": "bytes=12000-"})
        self.assertTrue(all(r.complete for r in results))
        self.assertEqual(
            d.local_path("a.safetensors").read_bytes(), self.payloads["a.safetensors"]
        )
        resumed = next(r for r in results if r.path == "a.safetensors")
        self.assertEqual(resumed.resumed_from, 12_000)

    def test_resumed_hash_covers_the_bytes_from_the_first_attempt(self):
        """A resume must hash the whole file, not only this run's bytes.

        If the digest were seeded only with the tail, a corrupt prefix written by
        the first attempt would pass verification and become the model.
        """
        d = self.downloader(workers=1)
        d.destination.mkdir(parents=True, exist_ok=True)
        # A partial with the right length but wrong content.
        d.partial_path("a.safetensors").write_bytes(b"\x00" * 12_000)

        with self.assertRaises(ValidationError) as ctx:
            d.fetch()
        self.assertIn("sha256 mismatch", str(ctx.exception))
        self.assertFalse(d.partial_path("a.safetensors").exists())
        self.assertFalse(d.local_path("a.safetensors").exists())

    def test_server_ignoring_range_restarts_cleanly(self):
        """Some CDNs answer 200 to a Range request. Appending would corrupt."""
        d = self.downloader(workers=1)
        d.destination.mkdir(parents=True, exist_ok=True)
        d.partial_path("a.safetensors").write_bytes(self.payloads["a.safetensors"][:9_000])
        self.fixture.server.honour_range = False

        results = d.fetch()
        self.assertTrue(all(r.complete for r in results))
        self.assertEqual(
            d.local_path("a.safetensors").read_bytes(), self.payloads["a.safetensors"]
        )

    def test_a_missing_file_fails_loudly(self):
        del self.fixture.server.payloads["b.safetensors"]
        d = self.downloader(workers=1)
        with self.assertRaises(ValidationError) as ctx:
            d.fetch()
        self.assertIn("b.safetensors", str(ctx.exception))
        # The others still landed, so a retry is cheap.
        self.assertTrue(d.local_path("a.safetensors").is_file())

    def test_hash_mismatch_removes_the_file(self):
        bad = dict(self.king.files)
        bad["a.safetensors"] = KingFile("a.safetensors", "0" * 64, self.FILES["a.safetensors"])
        king = KingReference(digest="d" * 64, files=bad, source=self.fixture.url)
        d = KingDownloader(king, self.tmp / "king", workers=1, timeout=10.0)

        with self.assertRaises(ValidationError):
            d.fetch()
        self.assertFalse(d.local_path("a.safetensors").exists())
        self.assertFalse(d.partial_path("a.safetensors").exists())

    # -- verification ------------------------------------------------------

    def test_verify_reports_missing_and_corrupt_files(self):
        d = self.downloader()
        d.fetch()
        d.local_path("b.safetensors").unlink()
        d.local_path("a.safetensors").write_bytes(b"\x00" * self.FILES["a.safetensors"])
        self.assertEqual(d.verify(), ["a.safetensors", "b.safetensors"])

    def test_sha256_file_matches_hashlib(self):
        path = self.tmp / "blob"
        body = make_payload("blob", 100_000)
        path.write_bytes(body)
        self.assertEqual(sha256_file(path), hashlib.sha256(body).hexdigest())

    def test_human_is_readable(self):
        self.assertEqual(human(512), "512.0 B")
        self.assertEqual(human(1536), "1.5 KiB")
        self.assertTrue(human(220_000_000_000).endswith("GiB"))


class PlanSummaryTestCase(unittest.TestCase):
    def test_summary_is_json_serialisable(self):
        king = KingReference(
            digest="a" * 64,
            files={"x": KingFile("x", "0" * 64, 10)},
            source="http://example.invalid",
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan = KingDownloader(king, Path(tmp)).plan()
            json.dumps(plan.summary())
            self.assertEqual(plan.summary()["missing"], 1)


if __name__ == "__main__":
    unittest.main()
