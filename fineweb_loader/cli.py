"""Command-line interface for the FineWeb-Edu shard loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .cache import DEFAULT_BUDGET_BYTES, ShardCache
from .errors import (
    EXIT_FAILED,
    EXIT_INTEGRITY,
    EXIT_OK,
    EXIT_USAGE,
    IntegrityError,
    LoaderError,
    ManifestError,
    ShardNotFoundError,
)
from .corpus import (
    DATASET_CONFIG_URL,
    CorpusSet,
    DatasetConfig,
    build_blended_holdout,
    split_by_corpus,
)
from .loader import BlendedLoader, FineWebLoader
from .manifest import DEFAULT_MANIFEST_URL, ShardManifest
from .npyio import NUMPY_AVAILABLE
from .refs import SequenceSet

DEFAULT_ROOT = Path.home() / "Documents" / "sn3" / "state"


def _root(args) -> Path:
    return Path(args.root).expanduser() if args.root else DEFAULT_ROOT


def _manifest_path(args) -> Path:
    return _root(args) / "fineweb-manifest.json"


def _dataset_config_path(args) -> Path:
    return _root(args) / "dataset-config.json"


def _corpora(args) -> CorpusSet:
    config = DatasetConfig.load(_dataset_config_path(args))
    return CorpusSet.open(
        config,
        _root(args),
        budget_bytes=int(args.budget * 1024**3) if args.budget else DEFAULT_BUDGET_BYTES,
    )


def _holdout_dir(args) -> Path:
    return _root(args) / "holdouts"


def _load_manifest(args) -> ShardManifest:
    return ShardManifest.load(_manifest_path(args))


def _cache(args, manifest: ShardManifest) -> ShardCache:
    return ShardCache(
        _root(args) / "cache",
        manifest,
        budget_bytes=int(args.budget * 1024**3) if args.budget else DEFAULT_BUDGET_BYTES,
    )


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


# -- manifest ---------------------------------------------------------------


def cmd_manifest_sync(args) -> int:
    print(f"downloading {args.url} …")
    manifest, raw = ShardManifest.download(args.url, timeout=args.timeout)
    path = _manifest_path(args)
    manifest.save(path)
    print(f"  {len(raw):,} bytes -> {path}")
    print(f"  canonical digest {manifest.digest}")
    problems = manifest.verify(
        expected_digest=args.expect_digest,
        expected_seq_len=args.expect_seq_len,
        expected_tokenizer=args.expect_tokenizer,
    )
    if problems:
        for problem in problems:
            print(f"  ! {problem}")
        return EXIT_INTEGRITY
    print("  verified: internal totals consistent")
    return EXIT_OK


def cmd_manifest_verify(args) -> int:
    manifest = _load_manifest(args)
    print(f"canonical digest {manifest.digest}")
    problems = manifest.verify(
        expected_digest=args.expect_digest,
        expected_seq_len=args.expect_seq_len,
        expected_tokenizer=args.expect_tokenizer,
    )
    if problems:
        for problem in problems:
            print(f"  ! {problem}")
        return EXIT_INTEGRITY
    print("  OK — nothing inconsistent")
    return EXIT_OK


def cmd_manifest_stats(args) -> int:
    manifest = _load_manifest(args)
    stats = manifest.stats()
    print(f"source            {manifest.source}")
    print(f"tokenizer         {manifest.tokenizer}")
    print(f"source revision   {manifest.source_revision}")
    print(f"dtype / seq_len   {manifest.dtype} / {manifest.seq_len}")
    print(f"canonical digest  {manifest.digest}")
    print()
    print(f"shards            {stats.total_shards:,}")
    print(f"crawls            {stats.crawls}")
    print(f"tokens            {stats.total_tokens:,}")
    print(f"sequences         {stats.total_sequences:,}")
    print(f"total size        {_human(stats.total_bytes)}  ({stats.total_bytes / 1e12:.2f} TB)")
    print(f"npy header/shard  {manifest.header_overhead} bytes")
    print()
    print(f"sequences/shard   min {stats.min_sequences} max {stats.max_sequences}")
    print(
        f"full shards       {stats.full_shards:,} "
        f"({stats.full_shards / stats.total_shards * 100:.1f}%)"
    )
    print(
        f"short (<2000)     {stats.short_shards:,} "
        f"({stats.short_shards / stats.total_shards * 100:.1f}%)"
    )
    if args.crawls:
        print("\nshards per crawl:")
        for crawl, shards in manifest.by_crawl().items():
            print(f"  {crawl:22} {len(shards):>6,}")
    return EXIT_OK


# -- corpus -----------------------------------------------------------------


def cmd_corpus_sync(args) -> int:
    """Download the live dataset configuration and every source inventory."""
    import json as _json
    import urllib.request as _url

    root = _root(args)
    req = _url.Request(args.url, headers={"User-Agent": "fineweb-loader/1.0"})
    payload = _json.loads(_url.urlopen(req, timeout=args.timeout).read().decode())
    config = DatasetConfig.from_payload(payload, source_url=args.url)
    config.save(_dataset_config_path(args), payload)

    print(f"dataset       {config.dataset_label}")
    print(f"config        {config.config_version}")
    print(f"delta / n     {config.delta_threshold} / {config.eval_n}")
    print(f"sources       {len(config.sources)}")
    for problem in config.check():
        print(f"  ! {problem}")

    print()
    for spec in config.sources:
        target = root / "manifests" / f"{spec.name}.json"
        if target.is_file() and not args.force:
            manifest = ShardManifest.load(target, base_url=spec.base_url)
            status = "cached"
        else:
            print(f"  downloading {spec.name} …", flush=True)
            manifest, _ = ShardManifest.download(
                spec.manifest_url, timeout=args.timeout, base_url=spec.base_url
            )
            manifest.save(target)
            status = "fetched"
        ok = manifest.digest == spec.manifest_sha256
        print(
            f"  {spec.name:22} {spec.proportion:>5.2f}  {len(manifest):>7,} shards  "
            f"{status:8} digest {'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            print(f"      expected {spec.manifest_sha256}")
            print(f"      computed {manifest.digest}")
            return EXIT_INTEGRITY

    print(f"\n  per-evaluation split of n={config.eval_n}:")
    for name, count in config.targets().items():
        print(f"    {name:22} {count:>5}")
    return EXIT_OK


def cmd_corpus_status(args) -> int:
    corpora = _corpora(args)
    stats = corpora.stats()
    config = corpora.config
    print(f"dataset       {config.dataset_label}")
    print(f"config        {config.config_version}")
    print(f"delta / n     {config.delta_threshold} / {config.eval_n}")
    print()
    print(f"  {'corpus':24} {'share':>6} {'shards':>9} {'tokens':>16} {'size':>10} {'cached':>7} {'draw':>6}")
    for name, s in stats["sources"].items():
        print(
            f"  {name:24} {s['proportion']:>6.2f} {s['shards']:>9,} {s['tokens']:>16,} "
            f"{_human(s['bytes']):>10} {s['cached']:>7} {stats['targets'].get(name, 0):>6}"
        )
    t = stats["total"]
    print(f"  {'TOTAL':24} {'':>6} {t['shards']:>9,} {t['tokens']:>16,} {_human(t['bytes']):>10}")
    problems = corpora.verify()
    print()
    if problems:
        for p in problems:
            print(f"  ! {p}")
        return EXIT_INTEGRITY
    print("  all sources verified against the live configuration")
    return EXIT_OK


# -- shards -----------------------------------------------------------------


def cmd_shard_fetch(args) -> int:
    manifest = _load_manifest(args)
    cache = _cache(args, manifest)
    entry = manifest.lookup(args.key)
    print(f"{entry.name}")
    print(f"  url    {manifest.url_for(entry)}")
    print(f"  size   {_human(entry.size_bytes)}  ({entry.sequences(manifest.seq_len)} sequences)")

    last = [-1]

    def progress(done: int, total: int) -> None:
        share = int(done * 40 / max(total, 1))
        if share != last[0]:
            last[0] = share
            sys.stdout.write(f"\r  [{'#' * share}{'.' * (40 - share)}] {_human(done)}")
            sys.stdout.flush()

    path = cache.ensure(entry, progress=None if args.quiet else progress)
    if not args.quiet:
        sys.stdout.write("\r" + " " * 70 + "\r")
    print(f"  ok     {path}")
    print(f"  cache  {_human(cache.used_bytes)} / {_human(cache.budget_bytes)}")
    return EXIT_OK


def cmd_shard_inspect(args) -> int:
    manifest = _load_manifest(args)
    cache = _cache(args, manifest)
    entry = manifest.lookup(args.key)
    if not cache.has(entry) and not args.fetch:
        print(f"{entry.name} is not cached; pass --fetch to download it")
        return EXIT_USAGE
    with FineWebLoader(manifest, cache) as loader:
        shard = loader.open_shard(entry.name)
        print(f"{entry.name}")
        print(f"  crawl / part     {entry.crawl} / {entry.part}")
        print(f"  tokens           {shard.n_tokens:,}")
        print(f"  sequences        {shard.n_sequences:,}")
        print(f"  ragged tail      {shard.has_ragged_tail}")
        print(f"  manifest tokens  {entry.n_tokens:,}")
        print(f"  backend          {'numpy' if NUMPY_AVAILABLE else 'stdlib mmap'}")
        head = shard.tolist(0)[:16]
        print(f"  first sequence   {head} …")
        tail = shard.tolist(shard.n_sequences - 1)[:16]
        print(f"  last sequence    {tail} …")
    return EXIT_OK


def cmd_shard_verify(args) -> int:
    manifest = _load_manifest(args)
    cache = _cache(args, manifest)
    entries = cache.entries()
    if not entries:
        print("cache is empty")
        return EXIT_OK
    bad = 0
    for cached in entries:
        ok = cache.verify_local(cached.key)
        print(f"  {'OK  ' if ok else 'BAD '} {cached.name}")
        bad += 0 if ok else 1
    print(f"\n{len(entries) - bad}/{len(entries)} verified")
    return EXIT_OK if bad == 0 else EXIT_INTEGRITY


# -- holdouts ---------------------------------------------------------------


def cmd_holdout_build(args) -> int:
    manifest = _load_manifest(args)
    existing = []
    directory = _holdout_dir(args)
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            other = SequenceSet.load(path)
            if other.name != args.name:
                existing.append(other)

    holdout = SequenceSet.build(
        manifest,
        name=args.name,
        seed=args.seed,
        n_shards=args.shards,
        per_shard=args.per_shard,
        full_shards_only=not args.include_short_shards,
        exclude=existing,
        notes=tuple(args.note or ()),
    )
    path = holdout.save(directory / f"{args.name}.json")
    print(f"built {holdout.name}")
    print(f"  sequences       {len(holdout):,}")
    print(f"  shards          {len(holdout.shards())}")
    print(f"  crawls          {len(holdout.crawls())}")
    print(f"  seed            {holdout.seed}")
    print(f"  manifest        {holdout.manifest_sha256[:16]}…")
    print(f"  strategy        {holdout.strategy}")
    if existing:
        print(f"  disjoint from   {', '.join(o.name for o in existing)}")
    print(f"  written to      {path}")
    return EXIT_OK


def cmd_holdout_blend(args) -> int:
    """Build a holdout mirroring the validator's corpus proportions."""
    corpora = _corpora(args)
    existing = []
    directory = _holdout_dir(args)
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            other = SequenceSet.load(path)
            if other.name != args.name:
                existing.append(other)

    holdout = build_blended_holdout(
        corpora,
        name=args.name,
        seed=args.seed,
        total=args.total,
        per_shard=args.per_shard,
        exclude=existing,
        notes=tuple(args.note or ()),
    )
    path = holdout.save(directory / f"{args.name}.json")
    by_corpus = split_by_corpus(holdout)
    print(f"built {holdout.name}")
    print(f"  sequences       {len(holdout):,}")
    print(f"  shards          {len(holdout.shards())}")
    print(f"  config version  {holdout.manifest_sha256[:24]}…")
    print("  per corpus:")
    target = corpora.targets(args.total or corpora.config.eval_n)
    for name, refs in by_corpus.items():
        share = len(refs) / len(holdout)
        print(
            f"    {name:24} {len(refs):>6} sequences  {share:>6.2%}  "
            f"(validator draws {target.get(name, 0)})"
        )
    if existing:
        print(f"  disjoint from   {', '.join(o.name for o in existing)}")
    print(f"  written to      {path}")
    return EXIT_OK


def cmd_holdout_list(args) -> int:
    directory = _holdout_dir(args)
    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    if not paths:
        print("no holdouts yet; run 'holdout build'")
        return EXIT_OK
    for path in paths:
        holdout = SequenceSet.load(path)
        print(
            f"  {holdout.name:16} {len(holdout):>7,} sequences  "
            f"{len(holdout.shards()):>4} shards  "
            f"{len(holdout.crawls()):>4} crawls  seed={holdout.seed}"
        )
    return EXIT_OK


def cmd_holdout_show(args) -> int:
    holdout = SequenceSet.load(_holdout_dir(args) / f"{args.name}.json")
    print(json.dumps({k: v for k, v in holdout.to_dict().items() if k != "refs"}, indent=2))
    print(f"\nfirst {min(args.limit, len(holdout))} refs:")
    for ref in holdout.refs[: args.limit]:
        print(f"  {ref}")
    return EXIT_OK


def cmd_holdout_check(args) -> int:
    """Confirm every pair of holdouts is disjoint."""
    directory = _holdout_dir(args)
    sets = [SequenceSet.load(p) for p in sorted(directory.glob("*.json"))]
    if len(sets) < 2:
        print("fewer than two holdouts; nothing to compare")
        return EXIT_OK
    clean = True
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            overlap = left.overlaps(right)
            status = "OK" if not overlap else f"OVERLAP {len(overlap)}"
            print(f"  {left.name:16} vs {right.name:16} {status}")
            clean = clean and not overlap
    return EXIT_OK if clean else EXIT_INTEGRITY


# -- cache ------------------------------------------------------------------


def cmd_cache_status(args) -> int:
    manifest = _load_manifest(args)
    cache = _cache(args, manifest)
    entries = cache.entries()
    print(f"root    {cache.root}")
    print(f"used    {_human(cache.used_bytes)} / {_human(cache.budget_bytes)}")
    print(f"shards  {len(entries)}")
    for cached in entries[-args.limit :]:
        print(f"  {cached.name:52} {_human(cached.size_bytes)}")
    return EXIT_OK


def cmd_cache_prune(args) -> int:
    manifest = _load_manifest(args)
    cache = _cache(args, manifest)
    before = cache.used_bytes
    freed = cache.prune()
    print(f"freed {_human(freed)}; now {_human(cache.used_bytes)} (was {_human(before)})")
    return EXIT_OK


# -- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fineweb",
        description="Reproducible, verified access to the SN3 FineWeb-Edu shards.",
    )
    parser.add_argument("--version", action="version", version=f"fineweb-loader {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", help="state directory (default ~/Documents/sn3/state)")
    common.add_argument("--budget", type=float, help="cache budget in GiB")

    expect = argparse.ArgumentParser(add_help=False)
    expect.add_argument("--expect-digest", help="required canonical manifest sha256")
    expect.add_argument("--expect-seq-len", type=int, default=2048)
    expect.add_argument("--expect-tokenizer", default=None)

    sub = parser.add_subparsers(dest="group", required=True)

    # manifest
    m = sub.add_parser("manifest", help="shard inventory").add_subparsers(
        dest="command", required=True
    )
    p = m.add_parser("sync", parents=[common, expect], help="download and verify")
    p.add_argument("--url", default=DEFAULT_MANIFEST_URL)
    p.add_argument("--timeout", type=float, default=120.0)
    p.set_defaults(func=cmd_manifest_sync)

    p = m.add_parser("verify", parents=[common, expect], help="verify the local copy")
    p.set_defaults(func=cmd_manifest_verify)

    p = m.add_parser("stats", parents=[common], help="inventory statistics")
    p.add_argument("--crawls", action="store_true", help="break down by crawl")
    p.set_defaults(func=cmd_manifest_stats)

    # corpus
    c0 = sub.add_parser("corpus", help="the multi-source evaluation corpus").add_subparsers(
        dest="command", required=True
    )
    p = c0.add_parser("sync", parents=[common], help="download config and every inventory")
    p.add_argument("--url", default=DATASET_CONFIG_URL)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--force", action="store_true", help="re-download cached inventories")
    p.set_defaults(func=cmd_corpus_sync)

    p = c0.add_parser("status", parents=[common], help="sources, sizes and draw sizes")
    p.set_defaults(func=cmd_corpus_status)

    # shard
    s = sub.add_parser("shard", help="individual shards").add_subparsers(
        dest="command", required=True
    )
    p = s.add_parser("fetch", parents=[common], help="download and verify one shard")
    p.add_argument("key")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_shard_fetch)

    p = s.add_parser("inspect", parents=[common], help="shape and first tokens")
    p.add_argument("key")
    p.add_argument("--fetch", action="store_true", help="download if not cached")
    p.set_defaults(func=cmd_shard_inspect)

    p = s.add_parser("verify", parents=[common], help="re-hash every cached shard")
    p.set_defaults(func=cmd_shard_verify)

    # holdout
    h = sub.add_parser("holdout", help="frozen validation sets").add_subparsers(
        dest="command", required=True
    )
    p = h.add_parser("build", parents=[common], help="build a stratified holdout")
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--shards", type=int, default=40)
    p.add_argument("--per-shard", type=int, default=128)
    p.add_argument("--include-short-shards", action="store_true")
    p.add_argument("--note", action="append")
    p.set_defaults(func=cmd_holdout_build)

    p = h.add_parser("blend", parents=[common], help="holdout across all corpora")
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total", type=int, help="sequences in total (default: eval_n)")
    p.add_argument("--per-shard", type=int, default=128)
    p.add_argument("--note", action="append")
    p.set_defaults(func=cmd_holdout_blend)

    p = h.add_parser("list", parents=[common], help="list holdouts")
    p.set_defaults(func=cmd_holdout_list)

    p = h.add_parser("show", parents=[common], help="show one holdout")
    p.add_argument("name")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_holdout_show)

    p = h.add_parser("check", parents=[common], help="assert holdouts are disjoint")
    p.set_defaults(func=cmd_holdout_check)

    # cache
    c = sub.add_parser("cache", help="local shard cache").add_subparsers(
        dest="command", required=True
    )
    p = c.add_parser("status", parents=[common], help="usage and contents")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_cache_status)

    p = c.add_parser("prune", parents=[common], help="evict down to budget")
    p.set_defaults(func=cmd_cache_prune)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except IntegrityError as exc:
        print(f"integrity error: {exc}", file=sys.stderr)
        return EXIT_INTEGRITY
    except (ManifestError, ShardNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except LoaderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
