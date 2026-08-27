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
from dataclasses import replace
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


# -- multi-corpus (2026-08-27 contract) -------------------------------------

from fineweb_loader.corpus import (  # noqa: E402
    Corpus,
    CorpusSet,
    DatasetConfig,
    SourceSpec,
    build_blended_holdout,
    source_targets,
    split_by_corpus,
)
from fineweb_loader.loader import BlendedLoader  # noqa: E402

LIVE_PROPORTIONS = [0.22, 0.26, 0.52]
LIVE_NAMES = ["finewebedu", "automathtext-v2", "dclm-baseline-1.0"]


class SourceTargetsTests(unittest.TestCase):
    """Largest-remainder apportionment, matching the validator exactly."""

    def test_live_split_of_2000(self):
        self.assertEqual(source_targets(2000, LIVE_PROPORTIONS), [440, 520, 1040])

    def test_always_sums_to_total(self):
        for total in (1, 7, 100, 999, 2000, 25000):
            self.assertEqual(sum(source_targets(total, LIVE_PROPORTIONS)), total)

    def test_remainder_goes_to_largest_fraction(self):
        # 10 * [1/3, 1/3, 1/3] -> floors 3,3,3 with remainder 1 to index 0.
        self.assertEqual(source_targets(10, [1 / 3, 1 / 3, 1 / 3]), [4, 3, 3])

    def test_differs_from_naive_rounding(self):
        # round() would give 1+1+1=3 here; apportionment must total 4.
        proportions = [0.34, 0.33, 0.33]
        self.assertEqual(sum(source_targets(4, proportions)), 4)

    def test_single_source(self):
        self.assertEqual(source_targets(2000, [1.0]), [2000])


def make_dataset_config(names=LIVE_NAMES, proportions=LIVE_PROPORTIONS, delta=0.1):
    return DatasetConfig(
        config_version="cfg-v1",
        dataset_label="-".join(names),
        delta_threshold=delta,
        eval_n=2000,
        sources=tuple(
            SourceSpec(
                name=n,
                proportion=p,
                manifest_url=f"https://example.test/{n}/manifest.json",
                manifest_sha256="",
                tokenizer="XiaomiMiMo/MiMo-V2.5-Pro",
                sequence_length=8,
                dtype="uint32",
            )
            for n, p in zip(names, proportions)
        ),
    )


class DatasetConfigTests(unittest.TestCase):
    def test_targets_match_the_live_split(self):
        self.assertEqual(
            make_dataset_config().targets(),
            {"finewebedu": 440, "automathtext-v2": 520, "dclm-baseline-1.0": 1040},
        )

    def test_base_url_is_derived_from_the_manifest_url(self):
        spec = make_dataset_config().sources[1]
        self.assertEqual(spec.base_url, "https://example.test/automathtext-v2")

    def test_check_catches_bad_proportions(self):
        bad = make_dataset_config(proportions=[0.2, 0.2, 0.2])
        self.assertTrue(any("sum to" in p for p in bad.check()))

    def test_check_catches_tokenizer_disagreement(self):
        config = make_dataset_config()
        sources = list(config.sources)
        sources[0] = replace(sources[0], tokenizer="other/tokenizer")
        mixed = replace(config, sources=tuple(sources))
        self.assertTrue(any("tokenizer" in p for p in mixed.check()))

    def test_from_payload_round_trip(self):
        payload = {
            "config_version": "abc",
            "dataset_label": "blend",
            "delta_threshold": 0.1,
            "eval_n": 2000,
            "sources": [
                {"name": "a", "proportion": 0.5, "manifest_url": "https://x/a/manifest.json"},
                {"name": "b", "proportion": 0.5, "manifest_url": "https://x/b/manifest.json"},
            ],
        }
        config = DatasetConfig.from_payload(payload)
        self.assertEqual(config.names, ["a", "b"])
        self.assertEqual(config.delta_threshold, 0.1)
        self.assertEqual(config.targets(10), {"a": 5, "b": 5})


def build_corpus_set(root: Path, seq_len: int = 8) -> CorpusSet:
    """Three synthetic corpora with realistic, corpus-prefixed shard names."""
    config = make_dataset_config()
    corpora = []
    for spec in config.sources:
        origin = root / spec.name
        shards = {
            f"{spec.name}__group{g}__part0__shard_{i:06d}.npy": 16
            for g in range(3)
            for i in range(2)
        }
        manifest = synth_manifest(origin, shards, seq_len=seq_len)
        manifest.base_url = origin.as_uri()
        cache = ShardCache(
            root / "cache" / spec.name,
            manifest,
            budget_bytes=10**9,
            bucket_root=origin.as_uri(),
        )
        corpora.append(Corpus(spec=spec, manifest=manifest, cache=cache))
    return CorpusSet(config, corpora)


class CorpusSetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.corpora = build_corpus_set(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_resolves_a_shard_to_its_corpus_by_prefix(self):
        name = "dclm-baseline-1.0__group1__part0__shard_000000.npy"
        self.assertEqual(self.corpora.corpus_of(name).name, "dclm-baseline-1.0")

    def test_hyphenated_corpus_names_resolve(self):
        # dclm-baseline-1.0 contains dots and hyphens; the split is on "__".
        for name in self.corpora.names:
            shard = f"{name}__group0__part0__shard_000000.npy"
            self.assertEqual(self.corpora.corpus_of(shard).name, name)

    def test_unknown_shard_raises(self):
        with self.assertRaises(ShardNotFoundError):
            self.corpora.corpus_of("nosuchcorpus__g__p__s.npy")

    def test_lookup_returns_corpus_and_entry(self):
        name = "finewebedu__group0__part0__shard_000000.npy"
        corpus, entry = self.corpora.lookup(name)
        self.assertEqual(corpus.name, "finewebedu")
        self.assertEqual(entry.name, name)
        self.assertEqual(entry.corpus, "finewebedu")

    def test_stats_cover_every_source(self):
        stats = self.corpora.stats()
        self.assertEqual(set(stats["sources"]), set(LIVE_NAMES))
        self.assertEqual(stats["targets"]["dclm-baseline-1.0"], 1040)

    def test_verify_flags_a_missing_source(self):
        partial = CorpusSet(self.corpora.config, [self.corpora["finewebedu"]])
        self.assertTrue(any("not loaded" in p for p in partial.verify()))


class BlendedHoldoutTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.corpora = build_corpus_set(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_proportions_match_the_validator(self):
        holdout = build_blended_holdout(
            self.corpora, name="blend", seed=1, total=100, per_shard=8,
            full_shards_only=False,
        )
        by_corpus = {k: len(v) for k, v in split_by_corpus(holdout).items()}
        self.assertEqual(len(holdout), 100)
        self.assertEqual(by_corpus, self.corpora.targets(100))

    def test_deterministic(self):
        a = build_blended_holdout(self.corpora, name="b", seed=7, total=60, per_shard=8, full_shards_only=False)
        b = build_blended_holdout(self.corpora, name="b", seed=7, total=60, per_shard=8, full_shards_only=False)
        self.assertEqual(a.refs, b.refs)

    def test_different_seeds_differ(self):
        a = build_blended_holdout(self.corpora, name="b", seed=1, total=60, per_shard=8, full_shards_only=False)
        b = build_blended_holdout(self.corpora, name="b", seed=2, total=60, per_shard=8, full_shards_only=False)
        self.assertNotEqual(a.refs, b.refs)

    def test_disjoint_from_an_excluded_set(self):
        first = build_blended_holdout(
            self.corpora, name="a", seed=1, total=60, per_shard=8,
            full_shards_only=False,
        )
        second = build_blended_holdout(
            self.corpora, name="b", seed=1, total=60, per_shard=8, exclude=[first],
            full_shards_only=False,
        )
        self.assertEqual(first.overlaps(second), set())

    def test_spans_every_corpus(self):
        holdout = build_blended_holdout(
            self.corpora, name="b", seed=3, total=100, per_shard=8,
            full_shards_only=False,
        )
        self.assertEqual(sorted(split_by_corpus(holdout)), sorted(LIVE_NAMES))


class BlendedLoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.corpora = build_corpus_set(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _refs(self, per_corpus=2):
        out = []
        for name in self.corpora.names:
            shard = f"{name}__group0__part0__shard_000000.npy"
            out.extend(SequenceRef(shard, i) for i in range(per_corpus))
        return out

    def test_reads_across_corpora_in_order(self):
        refs = self._refs()
        with BlendedLoader(self.corpora) as loader:
            rows = loader.sequences(refs)
            self.assertEqual(len(rows), len(refs))
            self.assertEqual(loader.stats.sequences_read, len(refs))

    def test_contamination_guard_spans_corpora(self):
        banned = self._refs(per_corpus=1)
        holdout = SequenceSet(
            name="h",
            refs=tuple(banned),
            seed=0,
            manifest_sha256="cfg-v1",
            seq_len=8,
            created="",
            strategy="test",
        )
        with BlendedLoader(self.corpora, holdouts=[holdout]) as loader:
            self.assertEqual(loader.excluded_count, len(banned))
            with self.assertRaises(ContaminationError):
                loader.sequences(banned, allow_holdout=False)
            # evaluation may still read them
            self.assertEqual(len(loader.sequences(banned)), len(banned))

    def test_training_stream_respects_proportions(self):
        shards = [
            f"{name}__group{g}__part0__shard_{i:06d}.npy"
            for name in self.corpora.names
            for g in range(3)
            for i in range(2)
        ]
        proportions = {s.name: s.proportion for s in self.corpora.config.sources}
        with BlendedLoader(self.corpora) as loader:
            seen = []
            for chunk, _ in loader.training_stream(
                seed=1, shards=shards, batch_size=16, proportions=proportions
            ):
                seen.extend(chunk)
        counts: dict[str, int] = {}
        for ref in seen:
            counts[ref.shard.split("__", 1)[0]] = (
                counts.get(ref.shard.split("__", 1)[0], 0) + 1
            )
        total = sum(counts.values())
        for name, share in proportions.items():
            self.assertAlmostEqual(counts[name] / total, share, delta=0.05)

    def test_training_stream_excludes_holdouts(self):
        banned = self._refs(per_corpus=4)
        holdout = SequenceSet(
            name="h", refs=tuple(banned), seed=0, manifest_sha256="cfg-v1",
            seq_len=8, created="", strategy="test",
        )
        shards = sorted({r.shard for r in banned})
        with BlendedLoader(self.corpora, holdouts=[holdout]) as loader:
            seen = set()
            for chunk, _ in loader.training_stream(seed=2, shards=shards, batch_size=8):
                seen.update(chunk)
            self.assertEqual(seen & holdout.as_set(), set())
