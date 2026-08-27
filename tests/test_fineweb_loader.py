"""Tests for fineweb_loader. Run with: python -m unittest discover -s tests -t .

Shard fixtures are synthesised so the suite stays fast and offline, but the
manifest tests run against the real 125,441-entry inventory when it is present
at state/fineweb-manifest.json, since its quirks are the whole point.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fineweb_loader import cli
from fineweb_loader.cache import ShardCache
from fineweb_loader.errors import (
    BudgetExceededError,
    ContaminationError,
    IntegrityError,
    LoaderError,
    ManifestError,
    NpyFormatError,
    ShardNotFoundError,
)
from fineweb_loader.loader import FineWebLoader
from fineweb_loader.manifest import (
    NPY_HEADER_BYTES,
    ShardManifest,
    canonical_sha256,
)
from fineweb_loader.npyio import Shard, read_header
from fineweb_loader.refs import SequenceRef, SequenceSet

REAL_MANIFEST = Path.home() / "Documents" / "sn3" / "state" / "fineweb-manifest.json"
REAL_DIGEST = "130273b000ef130cba39d9f9f467d12498dc7fc7d7e8043132e3bcb584848013"
SEQ_LEN = 2048


# -- helpers ----------------------------------------------------------------


def write_npy(path: Path, tokens: list[int]) -> int:
    """Write a minimal v1.0 uint32 .npy, padded to a 128-byte header like the real ones."""
    header = f"{{'descr': '<u4', 'fortran_order': False, 'shape': ({len(tokens)},), }}"
    prefix = len(b"\x93NUMPY") + 2 + 2
    pad = -(prefix + len(header) + 1) % 64
    header = header + " " * pad + "\n"
    body = b"".join(int(t).to_bytes(4, "little") for t in tokens)
    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY")
        handle.write(bytes([1, 0]))
        handle.write(len(header).to_bytes(2, "little"))
        handle.write(header.encode("latin1"))
        handle.write(body)
    return path.stat().st_size


def synth_manifest(root: Path, shards: dict[str, int], seq_len: int = 8):
    """Build a directory of shards plus a manifest describing them."""
    shard_dir = root / "finewebedu" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, n_sequences in shards.items():
        tokens = list(range(n_sequences * seq_len))
        path = shard_dir / name
        size = write_npy(path, tokens)
        rows.append(
            {
                "key": f"shards/{name}",
                "n_tokens": len(tokens),
                "size_bytes": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {
        "version": "shard-pipeline-v1",
        "tokenizer": "XiaomiMiMo/MiMo-V2.5-Pro",
        "dtype": "uint32",
        "seq_len": seq_len,
        "total_shards": len(rows),
        "total_tokens": sum(r["n_tokens"] for r in rows),
        "shard_prefix": "finewebedu/shards/",
        "shards": rows,
    }
    return ShardManifest(payload, source="synthetic")


def name_for(crawl: str, part: int, index: int) -> str:
    return f"finewebedu__{crawl}__part{part}__shard_{index:06d}.npy"


# -- npy reader -------------------------------------------------------------


class NpyReaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_header_is_128_bytes_like_the_real_shards(self):
        path = self.root / "a.npy"
        size = write_npy(path, list(range(64)))
        self.assertEqual(size - 64 * 4, NPY_HEADER_BYTES)

    def test_reads_shape_and_offset(self):
        path = self.root / "a.npy"
        write_npy(path, list(range(32)))
        header = read_header(path)
        self.assertEqual(header.shape, (32,))
        self.assertEqual(header.descr, "<u4")
        self.assertEqual(header.data_offset, NPY_HEADER_BYTES)

    def test_sequences_read_back_exactly(self):
        path = self.root / "a.npy"
        write_npy(path, list(range(4 * 8)))
        with Shard(path, seq_len=8) as shard:
            self.assertEqual(shard.n_sequences, 4)
            self.assertFalse(shard.has_ragged_tail)
            self.assertEqual(shard.tolist(0), list(range(0, 8)))
            self.assertEqual(shard.tolist(3), list(range(24, 32)))
            self.assertEqual(shard.tolist(-1), list(range(24, 32)))

    def test_ragged_tail_is_not_addressable(self):
        path = self.root / "a.npy"
        write_npy(path, list(range(20)))  # 2 whole sequences of 8, 4 left over
        with Shard(path, seq_len=8) as shard:
            self.assertEqual(shard.n_sequences, 2)
            self.assertTrue(shard.has_ragged_tail)
            with self.assertRaises(IndexError):
                shard.sequence(2)

    def test_rejects_non_npy(self):
        path = self.root / "bad.npy"
        path.write_bytes(b"not a numpy file at all")
        with self.assertRaises(NpyFormatError):
            read_header(path)

    def test_rejects_wrong_dtype(self):
        path = self.root / "f8.npy"
        header = "{'descr': '<f8', 'fortran_order': False, 'shape': (4,), }"
        pad = -(10 + len(header) + 1) % 64
        header = header + " " * pad + "\n"
        with path.open("wb") as handle:
            handle.write(b"\x93NUMPY" + bytes([1, 0]))
            handle.write(len(header).to_bytes(2, "little"))
            handle.write(header.encode("latin1"))
            handle.write(b"\x00" * 32)
        with self.assertRaises(NpyFormatError):
            read_header(path)


# -- manifest ---------------------------------------------------------------


class SyntheticManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manifest = synth_manifest(
            self.root,
            {
                name_for("CC-MAIN-2020-10", 0, 0): 16,
                name_for("CC-MAIN-2021-10", 0, 1): 16,
                name_for("CC-MAIN-2021-10", 1, 2): 2,
            },
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_parses_entries(self):
        self.assertEqual(len(self.manifest), 3)
        entry = self.manifest.lookup(name_for("CC-MAIN-2020-10", 0, 0))
        self.assertEqual(entry.crawl, "CC-MAIN-2020-10")
        self.assertEqual(entry.part, "part0")
        self.assertEqual(entry.sequences(self.manifest.seq_len), 16)

    def test_lookup_by_key_or_name(self):
        name = name_for("CC-MAIN-2020-10", 0, 0)
        self.assertEqual(self.manifest.lookup(name).name, name)
        self.assertEqual(self.manifest.lookup(f"shards/{name}").name, name)
        with self.assertRaises(ShardNotFoundError):
            self.manifest.lookup("nope.npy")

    def test_url_resolution_joins_prefix_namespace_to_key(self):
        entry = self.manifest.lookup(name_for("CC-MAIN-2020-10", 0, 0))
        url = self.manifest.url_for(entry, root="https://example.test")
        self.assertEqual(url, f"https://example.test/finewebedu/shards/{entry.name}")

    def test_verify_passes_on_consistent_manifest(self):
        self.assertEqual(self.manifest.verify(expected_seq_len=8), [])

    def test_verify_catches_token_total_mismatch(self):
        payload = json.loads(json.dumps(self.manifest.payload))
        payload["total_tokens"] += 1
        problems = ShardManifest(payload).verify()
        self.assertTrue(any("total_tokens" in p for p in problems))

    def test_verify_catches_impossible_size(self):
        payload = json.loads(json.dumps(self.manifest.payload))
        payload["shards"][0]["size_bytes"] = 4
        problems = ShardManifest(payload).verify()
        self.assertTrue(any("smaller than" in p for p in problems))

    def test_full_shards_filter_excludes_short_ones(self):
        full = self.manifest.full_shards(minimum=16)
        self.assertEqual(len(full), 2)

    def test_empty_manifest_is_rejected(self):
        with self.assertRaises(ManifestError):
            ShardManifest({"shards": []})

    def test_canonical_hash_is_order_independent(self):
        a = {"b": 1, "a": [1, 2]}
        b = {"a": [1, 2], "b": 1}
        self.assertEqual(canonical_sha256(a), canonical_sha256(b))


@unittest.skipUnless(REAL_MANIFEST.exists(), "real manifest not synced")
class RealManifestTests(unittest.TestCase):
    """The published inventory's actual quirks."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = ShardManifest.load(REAL_MANIFEST)

    def test_canonical_digest_matches_published_value(self):
        self.assertEqual(self.manifest.digest, REAL_DIGEST)

    def test_raw_byte_hash_does_not_match(self):
        raw = REAL_MANIFEST.read_bytes()
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(), REAL_DIGEST)

    def test_verifies_clean(self):
        self.assertEqual(
            self.manifest.verify(
                expected_digest=REAL_DIGEST,
                expected_seq_len=2048,
                expected_tokenizer="XiaomiMiMo/MiMo-V2.5-Pro",
                expected_dtype="uint32",
            ),
            [],
        )

    def test_totals(self):
        stats = self.manifest.stats()
        self.assertEqual(stats.total_shards, 125441)
        self.assertEqual(stats.total_tokens, 1567352367104)
        self.assertEqual(stats.total_sequences, 765308773)
        self.assertEqual(stats.crawls, 110)

    def test_shard_sizes_are_not_uniform(self):
        stats = self.manifest.stats()
        self.assertEqual(stats.max_sequences, 6144)
        self.assertEqual(stats.min_sequences, 3)
        self.assertEqual(stats.short_shards, 580)

    def test_header_overhead_is_uniform_128(self):
        self.assertEqual(self.manifest.header_overhead, NPY_HEADER_BYTES)


# -- refs and sequence sets -------------------------------------------------


class SequenceRefTests(unittest.TestCase):
    def test_round_trip(self):
        ref = SequenceRef("finewebedu__CC-MAIN-2020-10__part0__shard_000001.npy", 42)
        self.assertEqual(SequenceRef.parse(str(ref)), ref)

    def test_rejects_garbage(self):
        for text in ("", "no-index", "shard#", "shard#abc"):
            with self.assertRaises(ValueError):
                SequenceRef.parse(text)

    def test_is_hashable_and_ordered(self):
        a = SequenceRef("a.npy", 1)
        b = SequenceRef("a.npy", 2)
        self.assertLess(a, b)
        self.assertEqual(len({a, b, SequenceRef("a.npy", 1)}), 2)


class SequenceSetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        shards = {}
        for crawl in ("CC-MAIN-2019-10", "CC-MAIN-2020-10", "CC-MAIN-2021-10"):
            for part in range(2):
                shards[name_for(crawl, part, part)] = 32
        self.manifest = synth_manifest(self.root, shards)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, **kw):
        params = dict(
            name="val", seed=1, n_shards=3, per_shard=4, full_shards_only=False
        )
        params.update(kw)
        return SequenceSet.build(self.manifest, **params)

    def test_deterministic_for_a_given_seed(self):
        self.assertEqual(self._build().refs, self._build().refs)

    def test_different_seeds_differ(self):
        self.assertNotEqual(self._build(seed=1).refs, self._build(seed=2).refs)

    def test_shape_and_stratification(self):
        holdout = self._build(n_shards=3, per_shard=4)
        self.assertEqual(len(holdout), 12)
        self.assertEqual(len(holdout.shards()), 3)
        # One shard per crawl where possible, rather than clustering.
        self.assertEqual(len(holdout.crawls()), 3)

    def test_exclusion_produces_disjoint_sets(self):
        first = self._build(name="a", seed=1)
        second = self._build(name="b", seed=1, exclude=[first])
        self.assertEqual(first.overlaps(second), set())

    def test_rejects_impossible_request(self):
        with self.assertRaises(LoaderError):
            self._build(n_shards=99)
        with self.assertRaises(LoaderError):
            self._build(per_shard=999)

    def test_duplicate_refs_rejected(self):
        ref = SequenceRef("a.npy", 1)
        with self.assertRaises(LoaderError):
            SequenceSet(
                name="bad",
                refs=(ref, ref),
                seed=0,
                manifest_sha256="x",
                seq_len=8,
                created="",
                strategy="test",
            )

    def test_save_load_round_trip(self):
        holdout = self._build()
        path = holdout.save(self.root / "val.json")
        restored = SequenceSet.load(path)
        self.assertEqual(restored.refs, holdout.refs)
        self.assertEqual(restored.seed, holdout.seed)
        self.assertEqual(restored.manifest_sha256, holdout.manifest_sha256)


# -- cache ------------------------------------------------------------------


class CacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.origin = self.root / "origin"
        self.names = [name_for("CC-MAIN-2020-10", 0, i) for i in range(3)]
        self.manifest = synth_manifest(self.origin, {n: 16 for n in self.names})
        self.bucket = self.origin.as_uri()

    def tearDown(self):
        self._tmp.cleanup()

    def _cache(self, budget: int = 10**9) -> ShardCache:
        return ShardCache(
            self.root / "cache",
            self.manifest,
            budget_bytes=budget,
            bucket_root=self.bucket,
        )

    def test_fetch_verify_and_reuse(self):
        cache = self._cache()
        entry = self.manifest.lookup(self.names[0])
        path = cache.ensure(entry)
        self.assertTrue(path.exists())
        self.assertTrue(cache.has(entry))
        self.assertEqual(cache.used_bytes, entry.size_bytes)
        self.assertEqual(cache.ensure(entry), path)  # served from cache

    def test_corrupt_download_is_rejected_and_not_kept(self):
        payload = json.loads(json.dumps(self.manifest.payload))
        payload["shards"][0]["sha256"] = "0" * 64
        manifest = ShardManifest(payload)
        cache = ShardCache(
            self.root / "cache2",
            manifest,
            budget_bytes=10**9,
            bucket_root=self.bucket,
        )
        entry = manifest.lookup(self.names[0])
        with self.assertRaises(IntegrityError):
            cache.ensure(entry)
        self.assertFalse(cache.path_for(entry).exists())
        self.assertFalse(cache.has(entry))

    def test_wrong_size_is_rejected(self):
        payload = json.loads(json.dumps(self.manifest.payload))
        payload["shards"][0]["size_bytes"] += 10
        manifest = ShardManifest(payload)
        cache = ShardCache(
            self.root / "cache3",
            manifest,
            budget_bytes=10**9,
            bucket_root=self.bucket,
        )
        with self.assertRaises(IntegrityError):
            cache.ensure(manifest.lookup(self.names[0]))

    def test_lru_eviction_respects_budget(self):
        entry = self.manifest.lookup(self.names[0])
        cache = self._cache(budget=int(entry.size_bytes * 2.5))
        for name in self.names:
            cache.ensure(name)
        self.assertLessEqual(cache.used_bytes, cache.budget_bytes)
        self.assertEqual(len(cache.entries()), 2)
        self.assertFalse(cache.has(self.names[0]))  # oldest evicted
        self.assertTrue(cache.has(self.names[2]))

    def test_budget_smaller_than_one_shard_raises(self):
        cache = self._cache(budget=16)
        with self.assertRaises(BudgetExceededError):
            cache.ensure(self.names[0])

    def test_verify_local_evicts_rotten_file(self):
        cache = self._cache()
        entry = self.manifest.lookup(self.names[0])
        path = cache.ensure(entry)
        self.assertTrue(cache.verify_local(entry))
        path.write_bytes(b"corrupted")
        self.assertFalse(cache.verify_local(entry))
        self.assertFalse(cache.has(entry))

    def test_index_survives_reopen(self):
        cache = self._cache()
        cache.ensure(self.names[0])
        self.assertTrue(self._cache().has(self.names[0]))

    def test_corrupt_index_is_ignored(self):
        cache = self._cache()
        cache.ensure(self.names[0])
        cache.index_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self._cache().used_bytes, 0)


# -- loader -----------------------------------------------------------------


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.origin = self.root / "origin"
        self.names = [name_for("CC-MAIN-2020-10", 0, i) for i in range(2)]
        self.manifest = synth_manifest(self.origin, {n: 16 for n in self.names})
        self.cache = ShardCache(
            self.root / "cache",
            self.manifest,
            budget_bytes=10**9,
            bucket_root=self.origin.as_uri(),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _holdout(self, name="val", refs=None) -> SequenceSet:
        refs = refs or tuple(SequenceRef(self.names[0], i) for i in range(4))
        return SequenceSet(
            name=name,
            refs=refs,
            seed=0,
            manifest_sha256=self.manifest.digest,
            seq_len=self.manifest.seq_len,
            created="",
            strategy="test",
        )

    def test_reads_requested_sequences_in_order(self):
        refs = [
            SequenceRef(self.names[0], 3),
            SequenceRef(self.names[1], 1),
            SequenceRef(self.names[0], 0),
        ]
        with FineWebLoader(self.manifest, self.cache) as loader:
            rows = loader.sequences(refs)
            self.assertEqual(len(rows), 3)
            self.assertEqual(list(rows[0]), list(range(24, 32)))
            self.assertEqual(list(rows[2]), list(range(0, 8)))
            self.assertEqual(loader.stats.sequences_read, 3)
            self.assertEqual(loader.stats.shards_opened, 2)

    def test_shard_opened_once_per_loader(self):
        refs = [SequenceRef(self.names[0], i) for i in range(5)]
        with FineWebLoader(self.manifest, self.cache) as loader:
            loader.sequences(refs)
            loader.sequences(refs)
            self.assertEqual(loader.stats.shards_opened, 1)

    def test_batches_preserve_order_and_size(self):
        refs = [SequenceRef(self.names[0], i) for i in range(7)]
        with FineWebLoader(self.manifest, self.cache) as loader:
            sizes = [len(chunk) for chunk, _ in loader.batches(refs, batch_size=3)]
            self.assertEqual(sizes, [3, 3, 1])

    def test_contamination_raises(self):
        holdout = self._holdout()
        with FineWebLoader(self.manifest, self.cache, holdouts=[holdout]) as loader:
            self.assertEqual(loader.excluded_count, 4)
            with self.assertRaises(ContaminationError):
                loader.sequences(
                    [SequenceRef(self.names[0], 1)], allow_holdout=False
                )

    def test_evaluation_may_read_holdout(self):
        holdout = self._holdout()
        with FineWebLoader(self.manifest, self.cache, holdouts=[holdout]) as loader:
            rows = loader.sequences(list(holdout.refs))
            self.assertEqual(len(rows), 4)

    def test_training_stream_never_yields_holdout_refs(self):
        holdout = self._holdout()
        banned = holdout.as_set()
        with FineWebLoader(self.manifest, self.cache, holdouts=[holdout]) as loader:
            seen = set()
            for chunk, _ in loader.training_stream(
                seed=7, shards=[self.names[0]], batch_size=4
            ):
                seen.update(chunk)
            self.assertEqual(seen & banned, set())
            self.assertEqual(len(seen), 16 - 4)

    def test_training_stream_is_deterministic(self):
        with FineWebLoader(self.manifest, self.cache) as loader:
            first = [
                tuple(c) for c, _ in loader.training_stream(seed=3, shards=self.names)
            ]
        with FineWebLoader(self.manifest, self.cache) as loader:
            second = [
                tuple(c) for c, _ in loader.training_stream(seed=3, shards=self.names)
            ]
        self.assertEqual(first, second)

    def test_training_stream_refuses_when_everything_is_held_out(self):
        everything = self._holdout(
            refs=tuple(SequenceRef(self.names[0], i) for i in range(16))
        )
        with FineWebLoader(self.manifest, self.cache, holdouts=[everything]) as loader:
            with self.assertRaises(LoaderError):
                list(loader.training_stream(seed=1, shards=[self.names[0]]))

    def test_holdout_from_a_different_manifest_is_refused(self):
        stale = SequenceSet(
            name="stale",
            refs=(SequenceRef(self.names[0], 0),),
            seed=0,
            manifest_sha256="deadbeef",
            seq_len=self.manifest.seq_len,
            created="",
            strategy="test",
        )
        with FineWebLoader(self.manifest, self.cache) as loader:
            with self.assertRaises(LoaderError):
                loader.add_holdout(stale)


# -- cli --------------------------------------------------------------------


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.origin = self.root / "origin"
        shards = {}
        for crawl in ("CC-MAIN-2019-10", "CC-MAIN-2020-10", "CC-MAIN-2021-10"):
            shards[name_for(crawl, 0, 0)] = 16
        manifest = synth_manifest(self.origin, shards)
        manifest.save(self.root / "fineweb-manifest.json")
        self.base = ["--root", str(self.root)]

    def tearDown(self):
        self._tmp.cleanup()

    def test_manifest_verify_and_stats(self):
        self.assertEqual(cli.main(["manifest", "verify", *self.base, "--expect-seq-len", "8"]), 0)
        self.assertEqual(cli.main(["manifest", "stats", *self.base, "--crawls"]), 0)

    def test_manifest_verify_detects_wrong_digest(self):
        code = cli.main(
            ["manifest", "verify", *self.base, "--expect-digest", "0" * 64,
             "--expect-seq-len", "8"]
        )
        self.assertEqual(code, 3)

    def test_holdout_build_list_show_check(self):
        self.assertEqual(
            cli.main(["holdout", "build", *self.base, "--name", "a", "--seed", "1",
                      "--shards", "2", "--per-shard", "3", "--include-short-shards"]), 0)
        self.assertEqual(
            cli.main(["holdout", "build", *self.base, "--name", "b", "--seed", "2",
                      "--shards", "2", "--per-shard", "3", "--include-short-shards"]), 0)
        self.assertEqual(cli.main(["holdout", "list", *self.base]), 0)
        self.assertEqual(cli.main(["holdout", "show", *self.base, "a"]), 0)
        self.assertEqual(cli.main(["holdout", "check", *self.base]), 0)

    def test_holdouts_built_via_cli_are_disjoint(self):
        cli.main(["holdout", "build", *self.base, "--name", "a", "--seed", "1",
                  "--shards", "3", "--per-shard", "8", "--include-short-shards"])
        cli.main(["holdout", "build", *self.base, "--name", "b", "--seed", "1",
                  "--shards", "3", "--per-shard", "8", "--include-short-shards"])
        a = SequenceSet.load(self.root / "holdouts" / "a.json")
        b = SequenceSet.load(self.root / "holdouts" / "b.json")
        self.assertEqual(a.overlaps(b), set(), "same seed must still yield disjoint sets")

    def test_missing_manifest_is_usage_error(self):
        code = cli.main(["manifest", "stats", "--root", str(self.root / "empty")])
        self.assertEqual(code, 2)

    def test_unknown_shard_is_usage_error(self):
        code = cli.main(["shard", "inspect", *self.base, "nope.npy"])
        self.assertEqual(code, 2)

    def test_cache_status_on_empty_cache(self):
        self.assertEqual(cli.main(["cache", "status", *self.base]), 0)


if __name__ == "__main__":
    unittest.main()
