"""Tests for validate_checkpoint.

Fixtures are synthetic so the suite is fast and offline, but the contract tests
read the real chain.toml when the clone is present, since its 42 locked keys and
six hashes are the thing being enforced.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from validate_checkpoint import cli
from validate_checkpoint.checks import sha256_file
from validate_checkpoint.contract import (
    MAX_UPLOAD_BYTES,
    Contract,
    find_chain_toml,
)
from validate_checkpoint.errors import ContractError, SafetensorsFormatError
from validate_checkpoint.king import KingFile, KingReference
from validate_checkpoint.report import Layer, Report, Status
from validate_checkpoint.safetensors_io import read_header, scan_nonfinite
from validate_checkpoint.validator import Options, validate

REAL_CHAIN = Path.home() / "Documents" / "teutonic" / "chain.toml"

CONFIG = {
    "architectures": ["MiMoV2ForCausalLM"],
    "model_type": "mimo_v2",
    "vocab_size": 152576,
    "hidden_size": 3072,
    "num_hidden_layers": 45,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_group": 1,
    "topk_method": "noaux_tc",
}


# -- fixture helpers --------------------------------------------------------


def write_safetensors(path: Path, tensors: dict, *, truncate: int = 0) -> None:
    """Write a minimal valid .safetensors file."""
    header, blob = {}, b""
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [len(blob), len(blob) + len(payload)],
        }
        blob += payload
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    body = struct.pack("<Q", len(raw)) + raw + blob
    path.write_bytes(body[: len(body) - truncate] if truncate else body)


def bf16(value: int, count: int) -> bytes:
    return struct.pack("<H", value) * count


BF16_ONE = 0x3F80
BF16_NAN = 0x7FC0
BF16_INF = 0x7F80


def make_contract(root: Path, contract_files: dict[str, bytes]) -> Contract:
    """A synthetic chain.toml describing the given contract files."""
    hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in contract_files.items()
    }
    lines = [
        "[chain]",
        'name = "teutonic-TEST-1B"',
        'seed_repo = "owner/teutonic-TEST-1B-genesis"',
        'repo_pattern = "^[^/]+/teutonic-TEST-1B-.+$"',
        "",
        "[arch]",
        'module = "teutonic.archs.mimo"',
        'extra_lock_keys = ["n_routed_experts", "num_experts_per_tok", "n_group", "topk_method"]',
        "",
        "[seed]",
        "[seed.contract_files]",
    ]
    lines += [f'"{name}" = "{digest}"' for name, digest in sorted(hashes.items())]
    lines += ["", "[evaluation]", "n = 2000", "delta_threshold = 0.5"]
    path = root / "chain.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return Contract.load(path)


def build_checkpoint(root: Path, *, config: dict | None = None) -> tuple[Path, Contract, KingReference]:
    """A clean, self-consistent checkpoint plus a matching contract and king."""
    model = root / "model"
    model.mkdir(parents=True, exist_ok=True)

    config_bytes = json.dumps(config or CONFIG).encode()
    contract_files = {
        "config.json": config_bytes,
        "tokenizer.json": b'{"tokenizer": true}',
        "tokenizer_config.json": b"{}",
        "chat_template.jinja.txt": b"template",
        "configuration_mimo_v2.py": b"# configuration\n",
        "modeling_mimo_v2.py": b"# modeling\n",
    }
    for name, data in contract_files.items():
        (model / name).write_bytes(data)

    write_safetensors(
        model / "model-00001-of-00002.safetensors",
        {"model.layers.0.w": ("BF16", (2, 2), bf16(BF16_ONE, 4))},
    )
    write_safetensors(
        model / "model-00002-of-00002.safetensors",
        {"model.layers.1.w": ("BF16", (2, 2), bf16(BF16_ONE, 4))},
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "model.layers.0.w": "model-00001-of-00002.safetensors",
                    "model.layers.1.w": "model-00002-of-00002.safetensors",
                },
            }
        )
    )

    contract = make_contract(root, contract_files)
    files = {
        p.relative_to(model).as_posix(): KingFile(
            p.relative_to(model).as_posix(), sha256_file(p), p.stat().st_size
        )
        for p in sorted(model.rglob("*"))
        if p.is_file()
    }
    king = KingReference(
        digest="k" * 64,
        files=files,
        config=json.loads(config_bytes),
        index=json.loads((model / "model.safetensors.index.json").read_text()),
        source="synthetic",
    )
    return model, contract, king


def statuses(report: Report) -> dict[str, Status]:
    return {c.name: c.status for c in report.checks}


# -- safetensors ------------------------------------------------------------


class SafetensorsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_parses_header(self):
        path = self.root / "a.safetensors"
        write_safetensors(path, {"w": ("BF16", (2, 3), bf16(BF16_ONE, 6))})
        header = read_header(path)
        self.assertEqual(header.names, ["w"])
        self.assertEqual(header.tensors[0].shape, (2, 3))
        self.assertEqual(header.tensors[0].n_bytes, 12)
        self.assertEqual(header.problems(), [])

    def test_detects_truncation(self):
        path = self.root / "t.safetensors"
        write_safetensors(path, {"w": ("BF16", (8,), bf16(BF16_ONE, 8))}, truncate=6)
        problems = read_header(path).problems()
        self.assertTrue(any("truncated" in p for p in problems))

    def test_detects_shape_byte_mismatch(self):
        path = self.root / "m.safetensors"
        write_safetensors(path, {"w": ("BF16", (100,), bf16(BF16_ONE, 4))})
        self.assertTrue(any("reserves" in p for p in read_header(path).problems()))

    def test_rejects_non_safetensors(self):
        path = self.root / "bad.safetensors"
        path.write_bytes(b"nope")
        with self.assertRaises(SafetensorsFormatError):
            read_header(path)

    def test_rejects_absurd_header_length(self):
        path = self.root / "huge.safetensors"
        path.write_bytes(struct.pack("<Q", 2**60) + b"{}")
        with self.assertRaises(SafetensorsFormatError):
            read_header(path)

    def test_mtp_detection_is_case_insensitive_substring(self):
        path = self.root / "mtp.safetensors"
        write_safetensors(
            path,
            {
                "model.MTP.head.weight": ("BF16", (2,), bf16(BF16_ONE, 2)),
                "model.layers.0.w": ("BF16", (2,), bf16(BF16_ONE, 2)),
            },
        )
        header = read_header(path)
        flagged = [t.name for t in header.tensors if t.is_mtp()]
        self.assertEqual(flagged, ["model.MTP.head.weight"])

    def test_nonfinite_scan_finds_nan_and_inf(self):
        path = self.root / "n.safetensors"
        write_safetensors(
            path,
            {
                "clean": ("BF16", (4,), bf16(BF16_ONE, 4)),
                "nan": ("BF16", (4,), bf16(BF16_NAN, 4)),
                "inf": ("BF16", (2,), bf16(BF16_INF, 2)),
            },
        )
        found = dict(scan_nonfinite(read_header(path)))
        self.assertNotIn("clean", found)
        self.assertEqual(found["nan"], 4)
        self.assertEqual(found["inf"], 2)


# -- contract ---------------------------------------------------------------


class ContractTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.contract = make_contract(self.root, {"config.json": b"{}"})

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_contract_files_and_lock_keys(self):
        self.assertIn("config.json", self.contract.contract_files)
        self.assertIn("n_routed_experts", self.contract.locked_config_keys)
        self.assertIn("vocab_size", self.contract.locked_config_keys)  # generic

    def test_name_matching_handles_the_character_class(self):
        # The pattern contains [^/], so naive splitting on "/" is wrong.
        self.assertTrue(self.contract.name_matches("teutonic-TEST-1B-demo-001"))
        self.assertTrue(self.contract.name_matches("owner/teutonic-TEST-1B-demo"))
        self.assertFalse(self.contract.name_matches("some-other-model"))
        self.assertFalse(self.contract.name_matches("owner/wrong-prefix"))

    def test_missing_contract_files_section_raises(self):
        path = self.root / "empty.toml"
        path.write_text('[chain]\nname = "x"\n')
        with self.assertRaises(ContractError):
            Contract.load(path)

    def test_missing_file_raises(self):
        with self.assertRaises(ContractError):
            Contract.load(self.root / "nope.toml")


@unittest.skipUnless(REAL_CHAIN.exists(), "teutonic clone not present")
class RealContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = Contract.load(REAL_CHAIN)

    def test_generation_and_six_contract_files(self):
        self.assertEqual(self.contract.name, "teutonic-II-110B")
        self.assertEqual(len(self.contract.contract_files), 6)
        self.assertEqual(
            self.contract.contract_files["modeling_mimo_v2.py"],
            "08ee07cc45353e4bf01c7e2d207764f561f24c14d2bc258497928cd81f0c3f37",
        )

    def test_locked_key_count(self):
        # 12 generic + 30 arch-specific, no overlap.
        self.assertEqual(self.contract.n_locked_keys, 42)
        for key in ("topk_method", "n_group", "layer_types", "vocab_size"):
            self.assertIn(key, self.contract.locked_config_keys)

    def test_live_evaluation_settings(self):
        # The bar dropped from 0.5 to 0.1 on 2026-08-27, in the same change that
        # replaced the single FineWeb-Edu corpus with a three-way blend. This
        # assertion is deliberately pinned: if it fails again, the contract moved
        # again and every offline judgement needs re-checking.
        self.assertEqual(self.contract.eval_n, 2000)
        self.assertEqual(self.contract.delta_threshold, 0.1)
        self.assertEqual(
            self.contract.dataset_label,
            "finewebedu-automathtext-v2-dclm-baseline-1.0",
        )
        self.assertIn(226, self.contract.initial_weight_uids)

    def test_real_name_pattern(self):
        self.assertTrue(self.contract.name_matches("teutonic-II-110B-A7B-5ek5koe5-v5"))
        self.assertFalse(self.contract.name_matches("my-model"))

    def test_autodetect_finds_it(self):
        self.assertEqual(find_chain_toml(), REAL_CHAIN)


# -- king -------------------------------------------------------------------


class KingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_from_directory_with_manifest(self):
        d = self.root / "king"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "manifest.json").write_text(
            json.dumps({"files": [{"path": "config.json", "sha256": "ab", "size": 2}]})
        )
        king = KingReference.from_directory(d)
        self.assertEqual(king.sha256_for("config.json"), "ab")
        self.assertEqual(king.size_for("config.json"), 2)

    def test_from_directory_without_manifest_walks(self):
        d = self.root / "king"
        d.mkdir()
        (d / "a.safetensors").write_bytes(b"x" * 10)
        king = KingReference.from_directory(d)
        self.assertEqual(king.shard_names, ["a.safetensors"])
        self.assertEqual(king.size_for("a.safetensors"), 10)

    def test_bad_digest_rejected(self):
        from validate_checkpoint.errors import KingUnavailableError

        with self.assertRaises(KingUnavailableError):
            KingReference.from_digest("tooshort")


# -- checks, end to end -----------------------------------------------------


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.model, self.contract, self.king = build_checkpoint(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, **kw) -> Report:
        return validate(
            self.model, contract=self.contract, king=self.king, options=Options(**kw)
        )

    def test_clean_checkpoint_passes(self):
        # A real challenger has trained weights, so at least one shard differs
        # from the king. An untouched copy is covered separately below.
        write_safetensors(
            self.model / "model-00002-of-00002.safetensors",
            {"model.layers.1.w": ("BF16", (2, 2), bf16(0x3F00, 4))},
        )
        report = self._run(
            hash_shards=True, finite=True, model_name="teutonic-TEST-1B-demo-001"
        )
        self.assertFalse(report.would_reject, [c.detail for c in report.failures])
        # Nothing skipped: a pre-submission run must leave no rule unchecked.
        self.assertTrue(report.determinate, [c.name for c in report.skipped])
        self.assertEqual(report.verdict, "CLEAN")

    def test_reordered_config_fails_on_bytes_not_values(self):
        # The classic save_pretrained failure: identical values, different bytes.
        original = json.loads((self.model / "config.json").read_text())
        (self.model / "config.json").write_text(json.dumps(original, indent=2, sort_keys=True))
        report = self._run()
        self.assertIs(statuses(report)["contract files byte-identical"], Status.FAIL)
        self.assertIn("GenesisContractMismatch", report.error_codes)
        # The values are untouched, so the config lock itself still passes.
        # Byte identity and value identity are different gates.
        lock = next(c for c in report.checks if "locked config" in c.name)
        self.assertIs(lock.status, Status.PASS)

    def test_missing_contract_file(self):
        (self.model / "tokenizer_config.json").unlink()
        report = self._run()
        self.assertIs(statuses(report)["contract files present"], Status.FAIL)
        self.assertIn("GenesisContractMismatch", report.error_codes)

    def test_stray_file_is_undeclared(self):
        (self.model / "training_args.bin").write_bytes(b"junk")
        report = self._run()
        check = next(c for c in report.checks if c.name.startswith("inventory"))
        self.assertIs(check.status, Status.FAIL)
        self.assertIn("ArtifactIntegrityError", report.error_codes)
        self.assertTrue(any("training_args.bin" in i for i in check.items))

    def test_reserved_manifest_name(self):
        (self.model / "manifest.json").write_text("{}")
        report = self._run()
        self.assertIs(statuses(report)["manifest.json not present"], Status.FAIL)

    def test_symlink_rejected(self):
        (self.model / "link.json").symlink_to(self.model / "config.json")
        report = self._run()
        self.assertIs(statuses(report)["no symlinks"], Status.FAIL)

    def test_extra_python_file_rejected(self):
        (self.model / "custom_patch.py").write_text("# nope\n")
        report = self._run()
        self.assertIs(statuses(report)["only permitted python files"], Status.FAIL)

    def test_mtp_tensor_rejected(self):
        write_safetensors(
            self.model / "model-00002-of-00002.safetensors",
            {
                "model.layers.1.w": ("BF16", (2, 2), bf16(BF16_ONE, 4)),
                "model.mtp.head": ("BF16", (2,), bf16(BF16_ONE, 2)),
            },
        )
        report = self._run()
        self.assertIs(statuses(report)["no MTP tensors"], Status.FAIL)

    def test_dangling_index_reference(self):
        (self.model / "model-00002-of-00002.safetensors").unlink()
        report = self._run()
        self.assertIs(statuses(report)["index references existing shards"], Status.FAIL)

    def test_orphaned_shard(self):
        write_safetensors(
            self.model / "model-00003-of-00002.safetensors",
            {"model.layers.9.w": ("BF16", (2,), bf16(BF16_ONE, 2))},
        )
        report = self._run()
        self.assertIs(statuses(report)["no orphaned shards"], Status.FAIL)

    def test_config_value_change_fails_the_lock(self):
        config = dict(CONFIG, n_group=4)
        (self.model / "config.json").write_text(json.dumps(config))
        report = self._run()
        check = next(c for c in report.checks if "locked config" in c.name)
        self.assertIs(check.status, Status.FAIL)
        self.assertTrue(any("n_group" in i for i in check.items))

    def test_unchanged_weights_are_a_copy(self):
        report = self._run(hash_shards=True)
        # build_checkpoint made the king from these exact files.
        check = statuses(report)["weights differ from the king"]
        self.assertIs(check, Status.FAIL)

    def test_changed_shard_passes_copy_detection(self):
        write_safetensors(
            self.model / "model-00002-of-00002.safetensors",
            {"model.layers.1.w": ("BF16", (2, 2), bf16(0x3F00, 4))},
        )
        report = self._run(hash_shards=True)
        self.assertIs(statuses(report)["weights differ from the king"], Status.PASS)
        self.assertTrue(any(c.name == "changed-shard coverage" for c in report.checks))

    def test_nonfinite_weights_fail(self):
        write_safetensors(
            self.model / "model-00002-of-00002.safetensors",
            {"model.layers.1.w": ("BF16", (2, 2), bf16(BF16_NAN, 4))},
        )
        report = self._run(finite=True)
        self.assertIs(statuses(report)["weights are finite"], Status.FAIL)

    def test_truncated_shard_detected(self):
        write_safetensors(
            self.model / "model-00002-of-00002.safetensors",
            {"model.layers.1.w": ("BF16", (64,), bf16(BF16_ONE, 64))},
            truncate=40,
        )
        report = self._run()
        self.assertIs(statuses(report)["shard data offsets are sane"], Status.FAIL)

    def test_skips_are_reported_without_a_king(self):
        report = validate(self.model, contract=self.contract, king=None)
        self.assertFalse(report.determinate)
        self.assertTrue(report.skipped)
        self.assertFalse(report.would_reject)

    def test_missing_directory(self):
        report = validate(self.root / "nope", contract=self.contract, king=self.king)
        self.assertTrue(report.would_reject)

    def test_layers_mark_fatality_correctly(self):
        self.assertTrue(Layer.PREFLIGHT.recoverable)
        self.assertFalse(Layer.INGEST.recoverable)
        self.assertFalse(Layer.EVALUATOR.recoverable)
        (self.model / "training_args.bin").write_bytes(b"x")
        report = self._run()
        self.assertTrue(report.fatal_failures)


# -- cli --------------------------------------------------------------------


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.model, self.contract, _ = build_checkpoint(self.root)
        self.chain = str(self.contract.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_clean_run_without_king_is_undetermined(self):
        code = cli.main(["check", str(self.model), "--chain", self.chain])
        self.assertEqual(code, 2)

    def test_failure_exits_one(self):
        (self.model / "manifest.json").write_text("{}")
        code = cli.main(["check", str(self.model), "--chain", self.chain])
        self.assertEqual(code, 1)

    def test_json_output_parses(self):
        import contextlib, io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["check", str(self.model), "--chain", self.chain, "--json"])
        payload = json.loads(buf.getvalue())
        self.assertIn("checks", payload)
        self.assertIn("verdict", payload)

    def test_contract_command(self):
        self.assertEqual(cli.main(["contract", "--chain", self.chain]), 0)

    def test_king_without_reference_is_usage_error(self):
        self.assertEqual(cli.main(["king", "--chain", self.chain]), 3)

    def test_missing_chain_is_usage_error(self):
        code = cli.main(
            ["check", str(self.model), "--chain", str(self.root / "absent.toml")]
        )
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()


class ReuseLimitTests(unittest.TestCase):
    """The safetensors reuse limit added on 2026-08-27."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_combined_digest_matches_the_validator_construction(self):
        import hashlib

        from validate_checkpoint.reuse import safetensors_digest_from_file_digests

        digests = {
            "model-00002.safetensors": "b" * 64,
            "model-00001.safetensors": "a" * 64,
        }
        # engine.py: sorted by name, name || NUL || raw 32 bytes.
        expected = hashlib.sha256()
        for name in sorted(digests):
            expected.update(name.encode("utf-8"))
            expected.update(b"\0")
            expected.update(bytes.fromhex(digests[name]))
        self.assertEqual(
            safetensors_digest_from_file_digests(digests), expected.hexdigest()
        )

    def test_digest_is_order_independent_but_name_sensitive(self):
        from validate_checkpoint.reuse import safetensors_digest_from_file_digests

        a = {"x.safetensors": "a" * 64, "y.safetensors": "b" * 64}
        b = {"y.safetensors": "b" * 64, "x.safetensors": "a" * 64}
        self.assertEqual(
            safetensors_digest_from_file_digests(a),
            safetensors_digest_from_file_digests(b),
        )
        c = {"z.safetensors": "a" * 64, "y.safetensors": "b" * 64}
        self.assertNotEqual(
            safetensors_digest_from_file_digests(a),
            safetensors_digest_from_file_digests(c),
        )

    def test_rejects_malformed_digests(self):
        from validate_checkpoint.reuse import safetensors_digest_from_file_digests

        with self.assertRaises(FileNotFoundError):
            safetensors_digest_from_file_digests({})
        with self.assertRaises(ValueError):
            safetensors_digest_from_file_digests({"a": "not-a-digest"})

    def test_ledger_counts_and_limits(self):
        from validate_checkpoint.reuse import MAX_COMPLETED_EVALS, SubmissionLedger

        self.assertEqual(MAX_COMPLETED_EVALS, 3)
        ledger = SubmissionLedger.load(self.root)
        digest = "c" * 64
        self.assertEqual(ledger.uses(digest), 0)
        self.assertFalse(ledger.would_exceed(digest))
        for i in range(MAX_COMPLETED_EVALS):
            ledger.record(digest, self.root / f"ckpt-{i}", hotkey=f"hk{i}")
        self.assertEqual(ledger.uses(digest), MAX_COMPLETED_EVALS)
        self.assertEqual(ledger.remaining(digest), 0)
        self.assertTrue(ledger.would_exceed(digest))

    def test_ledger_persists(self):
        from validate_checkpoint.reuse import SubmissionLedger

        digest = "d" * 64
        SubmissionLedger.load(self.root).record(digest, "somewhere")
        self.assertEqual(SubmissionLedger.load(self.root).uses(digest), 1)

    def test_corrupt_ledger_is_ignored(self):
        from validate_checkpoint.reuse import LEDGER_FILENAME, SubmissionLedger

        (self.root / LEDGER_FILENAME).write_text("{not json", encoding="utf-8")
        self.assertEqual(SubmissionLedger.load(self.root).entries, [])

    def test_check_fails_at_the_limit(self):
        from validate_checkpoint.reuse import SubmissionLedger, snapshot_safetensors_digest

        model, contract, king = build_checkpoint(self.root)
        write_safetensors(
            model / "model-00002-of-00002.safetensors",
            {"model.layers.1.w": ("BF16", (2, 2), bf16(0x3F00, 4))},
        )
        digest, _ = snapshot_safetensors_digest(model)
        ledger = SubmissionLedger.load(self.root)
        for _ in range(3):
            ledger.record(digest, model)

        report = validate(
            model,
            contract=contract,
            king=king,
            options=Options(hash_shards=True, ledger=ledger),
        )
        check = next(c for c in report.checks if "reuse limit" in c.name)
        self.assertIs(check.status, Status.FAIL)
        self.assertIn("safetensors_reuse_limit", report.error_codes)

    def test_check_passes_for_unseen_weights(self):
        from validate_checkpoint.reuse import SubmissionLedger

        model, contract, king = build_checkpoint(self.root)
        write_safetensors(
            model / "model-00002-of-00002.safetensors",
            {"model.layers.1.w": ("BF16", (2, 2), bf16(0x3F00, 4))},
        )
        report = validate(
            model,
            contract=contract,
            king=king,
            options=Options(hash_shards=True, ledger=SubmissionLedger.load(self.root)),
        )
        check = next(c for c in report.checks if "reuse limit" in c.name)
        self.assertIs(check.status, Status.PASS)
