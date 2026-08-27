"""The individual rules, each mirroring something the real system enforces.

Every function takes a :class:`~validate_checkpoint.report.Report` and appends to
it, so a run produces one ordered account of what was examined.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .contract import MAX_UPLOAD_BYTES, RESERVED_FILENAME, Contract
from .errors import SafetensorsFormatError
from .king import KingReference
from .report import Layer, Report, Status
from .safetensors_io import SafetensorsHeader, read_header, scan_nonfinite, scannable_dtypes

CHUNK = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def walk_files(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file() or p.is_symlink())


# -- layer 1: what the miner CLI refuses to upload --------------------------


def check_tree(report: Report, root: Path) -> list[Path]:
    """Mirrors ``miner/upload_model.py:model_paths``."""
    if not root.is_dir():
        report.fail("directory exists", Layer.PREFLIGHT, f"{root} is not a directory")
        return []

    entries = walk_files(root)
    symlinks = [p.relative_to(root).as_posix() for p in entries if p.is_symlink()]
    if symlinks:
        report.fail(
            "no symlinks",
            Layer.PREFLIGHT,
            f"{len(symlinks)} symlink(s); the uploader aborts on the first one",
            items=symlinks[:10],
        )
    else:
        report.ok("no symlinks", Layer.PREFLIGHT)

    files = [p for p in entries if not p.is_symlink()]
    relatives = [p.relative_to(root).as_posix() for p in files]

    if RESERVED_FILENAME in relatives:
        report.fail(
            "manifest.json not present",
            Layer.PREFLIGHT,
            f"{RESERVED_FILENAME} is reserved; the CLI writes it during upload",
        )
    else:
        report.ok("manifest.json not present", Layer.PREFLIGHT)

    if not files:
        report.fail("directory is not empty", Layer.PREFLIGHT, "no files found")
    else:
        report.ok("directory is not empty", Layer.PREFLIGHT, f"{len(files)} files")

    return files


def check_size(report: Report, files: Iterable[Path]) -> int:
    """Mirrors ``access/storage.py:MAX_MINER_UPLOAD_BYTES``."""
    total = sum(p.stat().st_size for p in files)
    gb = total / 1e9
    if total > MAX_UPLOAD_BYTES:
        report.fail(
            "upload under 250 GB",
            Layer.INGEST,
            f"{gb:.2f} GB exceeds the {MAX_UPLOAD_BYTES / 1e9:.0f} GB cap",
        )
    else:
        headroom = (MAX_UPLOAD_BYTES - total) / 1e9
        report.ok("upload under 250 GB", Layer.INGEST, f"{gb:.2f} GB ({headroom:.1f} GB spare)")
    return total


# -- layer 3: the genesis contract ------------------------------------------


def check_contract_files(report: Report, root: Path, contract: Contract) -> None:
    """Mirrors ``access/storage.py:_validate_genesis_contract`` plus the re-hash.

    This produces ``GenesisContractMismatch``, the most common real failure. It
    is nearly always ``save_pretrained`` rewriting ``config.json`` -- reordered
    keys or a bumped ``transformers_version`` -- after which the bytes differ.
    """
    missing: list[str] = []
    mismatched: list[str] = []

    for name, expected in sorted(contract.contract_files.items()):
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = sha256_file(path)
        if actual != expected:
            mismatched.append(f"{name} (have {actual[:12]}…, want {expected[:12]}…)")

    if missing:
        report.fail(
            "contract files present",
            Layer.INGEST,
            f"{len(missing)} of {len(contract.contract_files)} missing",
            error_code="GenesisContractMismatch",
            items=missing,
        )
    else:
        report.ok(
            "contract files present",
            Layer.INGEST,
            f"all {len(contract.contract_files)} present",
        )

    if mismatched:
        report.fail(
            "contract files byte-identical",
            Layer.INGEST,
            "hashes differ from chain.toml; copy these files from the king "
            "after every save_pretrained",
            error_code="GenesisContractMismatch",
            items=mismatched,
        )
    elif not missing:
        report.ok("contract files byte-identical", Layer.INGEST, "all hashes match chain.toml")


def check_allowed_code_files(report: Report, root: Path, contract: Contract) -> None:
    """Only the two declared modules may ship. ``archs/mimo`` declares them."""
    present = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py"))
    unexpected = [p for p in present if p not in contract.allowed_code_files]
    if unexpected:
        report.fail(
            "only permitted python files",
            Layer.EVALUATOR,
            f"allowed: {', '.join(contract.allowed_code_files)}",
            items=unexpected,
        )
    else:
        report.ok("only permitted python files", Layer.EVALUATOR, f"{len(present)} present")


# -- layer 3: inventory -----------------------------------------------------


def check_inventory(
    report: Report, root: Path, files: list[Path], king: KingReference | None
) -> None:
    """Mirrors the exact set comparison that raises ``ArtifactIntegrityError``.

    The uploaded object set must equal the manifest's file set: no undeclared
    extras, no missing declarations.
    """
    relatives = {p.relative_to(root).as_posix() for p in files}

    duplicates_lower: dict[str, list[str]] = {}
    for name in relatives:
        duplicates_lower.setdefault(name.lower(), []).append(name)
    collisions = [v for v in duplicates_lower.values() if len(v) > 1]
    if collisions:
        report.warn(
            "no case-only filename collisions",
            Layer.INGEST,
            "paths differing only in case can collide in object storage",
            items=[", ".join(sorted(c)) for c in collisions[:5]],
        )
    else:
        report.ok("no case-only filename collisions", Layer.INGEST)

    if king is None or not king.paths:
        report.skip(
            "inventory matches the king's file set",
            Layer.INGEST,
            "no king reference; pass --king-digest or --king",
        )
        return

    expected = set(king.paths)
    missing = sorted(expected - relatives)
    undeclared = sorted(relatives - expected)

    if missing or undeclared:
        detail = f"missing={len(missing)} undeclared={len(undeclared)}"
        report.fail(
            "inventory matches the king's file set",
            Layer.INGEST,
            detail,
            error_code="ArtifactIntegrityError",
            items=[f"missing: {m}" for m in missing[:8]]
            + [f"undeclared: {u}" for u in undeclared[:8]],
        )
    else:
        report.ok(
            "inventory matches the king's file set",
            Layer.INGEST,
            f"{len(expected)} files, exactly as the king declares",
        )


# -- layer 4: weights -------------------------------------------------------


def load_headers(report: Report, root: Path) -> dict[str, SafetensorsHeader]:
    headers: dict[str, SafetensorsHeader] = {}
    broken: list[str] = []
    for path in sorted(root.rglob("*.safetensors")):
        name = path.relative_to(root).as_posix()
        try:
            headers[name] = read_header(path)
        except SafetensorsFormatError as exc:
            broken.append(f"{name}: {exc}")
    if broken:
        report.fail("safetensors headers parse", Layer.EVALUATOR, "unreadable shard(s)", items=broken)
    elif headers:
        total = sum(len(h.tensors) for h in headers.values())
        report.ok(
            "safetensors headers parse",
            Layer.EVALUATOR,
            f"{len(headers)} shards, {total:,} tensors",
        )
    else:
        report.fail("safetensors headers parse", Layer.EVALUATOR, "no .safetensors files found")
    return headers


def check_mtp(report: Report, headers: dict[str, SafetensorsHeader]) -> None:
    """Mirrors ``engine.py:reject_mtp_checkpoint_weights`` -- substring, case-insensitive."""
    offenders = [
        f"{shard}: {t.name}"
        for shard, header in sorted(headers.items())
        for t in header.tensors
        if t.is_mtp()
    ]
    if offenders:
        report.fail(
            "no MTP tensors",
            Layer.EVALUATOR,
            "multi-token-prediction weights are rejected before scoring",
            items=offenders[:8],
        )
    else:
        report.ok("no MTP tensors", Layer.EVALUATOR)


def check_header_structure(report: Report, headers: dict[str, SafetensorsHeader]) -> None:
    """Truncated shards and impossible offsets, from headers alone."""
    problems = [
        f"{shard}: {problem}"
        for shard, header in sorted(headers.items())
        for problem in header.problems()
    ]
    if problems:
        report.fail(
            "shard data offsets are sane",
            Layer.EVALUATOR,
            "a truncated or inconsistent shard",
            items=problems[:8],
        )
    else:
        report.ok("shard data offsets are sane", Layer.EVALUATOR, f"{len(headers)} shards")


def check_dtypes(report: Report, headers: dict[str, SafetensorsHeader]) -> None:
    dtypes: set[str] = set()
    for header in headers.values():
        dtypes |= header.dtypes
    if not dtypes:
        report.skip("weights are bf16", Layer.LOCAL, "no tensors found")
    elif dtypes == {"BF16"}:
        report.ok("weights are bf16", Layer.LOCAL, "BF16 throughout")
    else:
        report.warn(
            "weights are bf16",
            Layer.LOCAL,
            f"mixed dtypes: {', '.join(sorted(dtypes))}; the king ships bf16",
        )


def check_index(report: Report, root: Path, headers: dict[str, SafetensorsHeader]) -> None:
    """Every weight_map target must exist, and every shard must be referenced."""
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        report.fail("weight index present", Layer.EVALUATOR, "model.safetensors.index.json missing")
        return
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.fail("weight index parses", Layer.EVALUATOR, str(exc))
        return
    report.ok("weight index present", Layer.EVALUATOR)

    weight_map: dict[str, str] = dict(index.get("weight_map") or {})
    if not weight_map:
        report.fail("weight index is populated", Layer.EVALUATOR, "weight_map is empty")
        return

    referenced = set(weight_map.values())
    present = set(headers)

    dangling = sorted(referenced - present)
    orphaned = sorted(present - referenced)
    if dangling:
        report.fail(
            "index references existing shards",
            Layer.EVALUATOR,
            f"{len(dangling)} shard(s) referenced but absent",
            items=dangling[:8],
        )
    else:
        report.ok("index references existing shards", Layer.EVALUATOR, f"{len(referenced)} shards")

    if orphaned:
        report.fail(
            "no orphaned shards",
            Layer.EVALUATOR,
            f"{len(orphaned)} shard(s) present but unreferenced",
            error_code="ArtifactIntegrityError",
            items=orphaned[:8],
        )
    else:
        report.ok("no orphaned shards", Layer.EVALUATOR)

    declared = set(weight_map)
    actual = {t.name for header in headers.values() for t in header.tensors}
    missing_tensors = sorted(declared - actual)
    extra_tensors = sorted(actual - declared)
    if missing_tensors:
        report.fail(
            "index tensor names resolve",
            Layer.EVALUATOR,
            f"{len(missing_tensors)} tensor(s) in the index are absent from the shards",
            items=missing_tensors[:8],
        )
    elif extra_tensors:
        report.warn(
            "index tensor names resolve",
            Layer.EVALUATOR,
            f"{len(extra_tensors)} tensor(s) in shards are not in the index",
            items=extra_tensors[:8],
        )
    else:
        report.ok("index tensor names resolve", Layer.EVALUATOR, f"{len(declared):,} tensors")


# -- layer 4: config lock ---------------------------------------------------


def check_config_lock(
    report: Report, root: Path, contract: Contract, king: KingReference | None
) -> None:
    """All 43 locked keys, plus ``architectures``.

    The key list is read from chain.toml, so it tracks upstream changes without
    a release here.
    """
    config_path = root / "config.json"
    if not config_path.is_file():
        report.fail("config.json present", Layer.EVALUATOR, "missing")
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.fail("config.json parses", Layer.EVALUATOR, str(exc))
        return

    if king is None or not king.config:
        report.skip(
            f"{contract.n_locked_keys} locked config keys match",
            Layer.EVALUATOR,
            "no king config available; pass --king-digest or --king",
        )
        return

    sentinel = object()
    differences: list[str] = []

    king_arch = king.config.get("architectures", [])
    our_arch = config.get("architectures", [])
    if king_arch and our_arch and king_arch != our_arch:
        differences.append(f"architectures: {our_arch} != {king_arch}")

    for key in contract.locked_config_keys:
        theirs = king.config.get(key, sentinel)
        ours = config.get(key, sentinel)
        if theirs is sentinel and ours is sentinel:
            continue
        if theirs != ours:
            differences.append(
                f"{key}: {_show(ours, sentinel)} != {_show(theirs, sentinel)}"
            )

    if differences:
        report.fail(
            f"{contract.n_locked_keys} locked config keys match",
            Layer.EVALUATOR,
            f"{len(differences)} key(s) differ from the king",
            items=differences[:10],
        )
    else:
        report.ok(
            f"{contract.n_locked_keys} locked config keys match",
            Layer.EVALUATOR,
            f"{contract.n_locked_keys} keys plus architectures",
        )


def _show(value: Any, sentinel: Any) -> str:
    if value is sentinel:
        return "<absent>"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "…"


# -- layer 4: copy detection ------------------------------------------------


def check_weights_changed(
    report: Report, root: Path, king: KingReference | None, *, hash_shards: bool
) -> None:
    """Mirrors ``policy.py:decide_model_copy``.

    Rejection needs *every* shard digest to match the king's, so partial-freeze
    training is safe. A checkpoint where nothing changed is not.
    """
    if king is None or not king.shard_names:
        report.skip("weights differ from the king", Layer.EVALUATOR, "no king reference")
        return
    if not hash_shards:
        report.skip(
            "weights differ from the king",
            Layer.EVALUATOR,
            "shard hashing not requested; pass --hash-shards",
        )
        return

    changed: list[str] = []
    identical: list[str] = []
    absent: list[str] = []
    for name in king.shard_names:
        path = root / name
        if not path.is_file():
            absent.append(name)
            continue
        expected = king.sha256_for(name)
        actual = sha256_file(path)
        (identical if actual == expected else changed).append(name)

    if absent:
        report.fail(
            "weights differ from the king",
            Layer.EVALUATOR,
            f"{len(absent)} shard(s) absent, cannot compare",
            items=absent[:8],
        )
        return

    if not changed:
        report.fail(
            "weights differ from the king",
            Layer.EVALUATOR,
            "every shard is byte-identical to the king; this is a copy",
            items=identical[:8],
        )
    else:
        report.ok(
            "weights differ from the king",
            Layer.EVALUATOR,
            f"{len(changed)} of {len(king.shard_names)} shards changed",
        )
        if identical:
            report.warn(
                "changed-shard coverage",
                Layer.LOCAL,
                f"{len(identical)} shard(s) untouched — expected for staged "
                "unfreezing, surprising for a full run",
                items=identical[:8],
            )


# -- layer 4: reuse limit ---------------------------------------------------


def check_reuse_limit(
    report: Report, root: Path, ledger, *, hash_shards: bool
) -> str | None:
    """Mirrors ``engine.py:reject_reused_safetensors``.

    The validator refuses a fourth evaluation of the same combined safetensors
    digest. It keeps that count server-side, so all that can be done here is
    consult a local record of what has already been sent.
    """
    from .reuse import MAX_COMPLETED_EVALS, snapshot_safetensors_digest

    if not hash_shards:
        report.skip(
            "safetensors reuse limit",
            Layer.EVALUATOR,
            "needs shard hashes; pass --hash-shards or --thorough",
        )
        return None
    try:
        digest, _ = snapshot_safetensors_digest(root)
    except FileNotFoundError as exc:
        report.fail("safetensors reuse limit", Layer.EVALUATOR, str(exc))
        return None

    if ledger is None:
        report.ok(
            "safetensors reuse limit",
            Layer.EVALUATOR,
            f"digest {digest[:16]}\u2026 (no local ledger to compare against)",
        )
        return digest

    uses = ledger.uses(digest)
    if uses >= MAX_COMPLETED_EVALS:
        report.fail(
            "safetensors reuse limit",
            Layer.EVALUATOR,
            f"these exact weights are recorded as submitted {uses} time(s); "
            f"the validator allows {MAX_COMPLETED_EVALS}",
            error_code="safetensors_reuse_limit",
        )
    elif uses:
        report.warn(
            "safetensors reuse limit",
            Layer.EVALUATOR,
            f"submitted {uses}/{MAX_COMPLETED_EVALS} time(s) already; "
            f"{ledger.remaining(digest)} left for these exact weights",
        )
    else:
        report.ok(
            "safetensors reuse limit",
            Layer.EVALUATOR,
            f"digest {digest[:16]}\u2026 not previously submitted",
        )
    return digest


# -- layer 4: numerics ------------------------------------------------------


def check_finite(report: Report, headers: dict[str, SafetensorsHeader], *, enabled: bool) -> None:
    """NaN or Inf makes a checkpoint unscoreable.

    Streams the files; no model load and no GPU. Slow on a full checkpoint, so
    it is opt-in.
    """
    if not enabled:
        report.skip("weights are finite", Layer.LOCAL, "not requested; pass --finite")
        return
    offenders: list[str] = []
    skipped_dtypes: set[str] = set()
    for shard, header in sorted(headers.items()):
        _, skipped = scannable_dtypes(header)
        skipped_dtypes |= skipped
        for name, count in scan_nonfinite(header):
            offenders.append(f"{shard}: {name} ({count:,} non-finite)")
    if offenders:
        report.fail(
            "weights are finite",
            Layer.LOCAL,
            "NaN or Inf present; the model cannot be scored",
            items=offenders[:8],
        )
    else:
        detail = f"{len(headers)} shards scanned"
        if skipped_dtypes:
            detail += f"; dtypes not scanned: {', '.join(sorted(skipped_dtypes))}"
        report.ok("weights are finite", Layer.LOCAL, detail)


# -- naming -----------------------------------------------------------------


def check_name(report: Report, contract: Contract, model_name: str | None) -> None:
    if not model_name:
        report.skip(
            "submission name matches repo_pattern",
            Layer.INGEST,
            "no --name given",
        )
        return
    if contract.name_matches(model_name):
        report.ok(
            "submission name matches repo_pattern",
            Layer.INGEST,
            f"{model_name!r} satisfies {contract.repo_pattern}",
        )
    else:
        report.fail(
            "submission name matches repo_pattern",
            Layer.INGEST,
            f"{model_name!r} does not match {contract.repo_pattern}",
        )
