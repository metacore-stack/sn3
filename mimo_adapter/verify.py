"""Proof that the patched router is safe to train against.

Order matters. Numerical parity comes before anything else: a router that is
subtly wrong still produces plausible losses, and you would not discover the
mistake until a GPU budget had already been spent on it.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import ParityError
from .patch import RoutingRecorder, gates, set_gates_eval, trainable_routing

# Parameter groups, matched against parameter names.
GROUPS = {
    "router": (".gate.weight",),
    "router_bias": (".e_score_correction_bias",),
    "routed_experts": (".experts.",),
    "shared_expert": (".shared_experts.",),
    "attention": (".self_attn.",),
    "layernorm": ("norm",),
    "embedding": ("embed_tokens",),
    "lm_head": ("lm_head",),
}

PARITY_TOLERANCE = 0.0  # exact: the same ops on the same inputs


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", **data: Any) -> Check:
        check = Check(name, passed, detail, data)
        self.checks.append(check)
        return check

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail, **c.data}
                for c in self.checks
            ],
        }


def _batch(model, config, *, batch: int = 2, length: int = 16, seed: int = 7):
    import torch

    torch.manual_seed(seed)
    return torch.randint(0, int(config.vocab_size), (batch, length))


def group_of(name: str) -> str:
    for group, needles in GROUPS.items():
        if any(needle in name for needle in needles):
            return group
    return "other"


# -- the checks -------------------------------------------------------------


def check_guard(report: Report, arch, model, config) -> None:
    """The shipped gate must refuse to run in training mode.

    If this does not fire, the miniature is not exercising the real path and
    every result below is meaningless.
    """
    import torch

    input_ids = _batch(model, config)
    model.train()
    try:
        model(input_ids=input_ids)
    except ValueError as exc:
        if "noaux_tc" in str(exc):
            report.add(
                "shipped gate refuses training mode",
                True,
                str(exc).strip()[:100],
            )
        else:
            report.add("shipped gate refuses training mode", False, f"unexpected: {exc}")
    except Exception as exc:  # noqa: BLE001
        report.add(
            "shipped gate refuses training mode", False, f"{type(exc).__name__}: {exc}"
        )
    else:
        report.add(
            "shipped gate refuses training mode",
            False,
            "no ValueError raised — the miniature may not use noaux_tc routing",
        )
    finally:
        model.eval()


def check_eval_parity(report: Report, arch, model, config, *, samples: int = 3) -> None:
    """Patched and original must agree exactly in eval mode.

    This is the check the whole module exists for.
    """
    import torch

    model.eval()
    worst = 0.0
    mismatched_routes = 0
    for i in range(samples):
        input_ids = _batch(model, config, seed=100 + i)
        with torch.no_grad():
            baseline = model(input_ids=input_ids).logits.clone()
        with trainable_routing(arch):
            with torch.no_grad():
                patched = model(input_ids=input_ids).logits.clone()
        delta = (baseline - patched).abs().max().item()
        worst = max(worst, delta)
        if delta > 0:
            mismatched_routes += 1

    passed = worst <= PARITY_TOLERANCE
    report.add(
        "patched routing is numerically identical in eval",
        passed,
        f"max |Δlogit| = {worst:.3e} over {samples} inputs"
        + ("" if passed else f"; {mismatched_routes} sample(s) diverged"),
        max_abs_delta=worst,
        samples=samples,
    )
    if not passed:
        raise ParityError(
            f"patched router diverges from the original by {worst:.3e}; "
            "do not train against it"
        )


def check_eval_mode_trick(report: Report, arch, model, config) -> None:
    """The cheaper alternative: force gates to eval inside a training model."""
    import torch

    model.train()
    n = set_gates_eval(model)
    try:
        input_ids = _batch(model, config)
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        grads = sum(
            1
            for _, p in model.named_parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )
        report.add(
            "gates-to-eval trick also permits backward",
            grads > 0,
            f"{n} gates set to eval; {grads} parameters received gradient",
            gates=n,
            params_with_grad=grads,
        )
    except Exception as exc:  # noqa: BLE001
        report.add(
            "gates-to-eval trick also permits backward",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()


def check_gradients(report: Report, arch, model, config) -> dict[str, Any]:
    """Backward must reach the parameter groups training depends on."""
    import torch

    model.zero_grad(set_to_none=True)
    model.train()
    stats: dict[str, dict[str, Any]] = {}
    try:
        with trainable_routing(arch):
            input_ids = _batch(model, config)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            finite_loss = bool(torch.isfinite(loss).item())
            report.add(
                "forward produces a finite loss in training mode",
                finite_loss,
                f"loss = {loss.item():.6f}",
                loss=loss.item(),
            )
            loss.backward()
    except Exception as exc:  # noqa: BLE001
        report.add("backward completes", False, f"{type(exc).__name__}: {exc}")
        model.eval()
        return {}
    report.add("backward completes", True, "no exception")

    nonfinite: list[str] = []
    for name, param in model.named_parameters():
        group = group_of(name)
        entry = stats.setdefault(
            group, {"params": 0, "with_grad": 0, "nonzero": 0, "grad_norm": 0.0}
        )
        entry["params"] += 1
        if param.grad is None:
            continue
        entry["with_grad"] += 1
        if not torch.isfinite(param.grad).all().item():
            nonfinite.append(name)
            continue
        norm = param.grad.norm().item()
        entry["grad_norm"] += norm
        if norm > 0:
            entry["nonzero"] += 1

    report.add(
        "all gradients are finite",
        not nonfinite,
        "no NaN or Inf in any gradient"
        if not nonfinite
        else f"{len(nonfinite)} parameter(s) have non-finite gradients",
        nonfinite=nonfinite[:5],
    )

    for group in ("router", "routed_experts", "shared_expert", "attention", "lm_head"):
        entry = stats.get(group)
        if not entry or entry["params"] == 0:
            continue
        report.add(
            f"gradient reaches {group}",
            entry["nonzero"] > 0,
            f"{entry['nonzero']}/{entry['params']} parameters with non-zero gradient, "
            f"total norm {entry['grad_norm']:.4e}",
            **entry,
        )

    bias = stats.get("router_bias")
    if bias and bias["params"]:
        # Documented, expected behaviour rather than a fault: the correction bias
        # steers selection only, and selection is not differentiable.
        report.add(
            "router bias receives no gradient (expected)",
            bias["nonzero"] == 0,
            "e_score_correction_bias is updated by a load-balancing rule you run "
            "yourself, not by backprop",
            **bias,
        )

    model.zero_grad(set_to_none=True)
    model.eval()
    return stats


def check_expert_coverage(
    report: Report, arch, model, config, *, batch: int = 4, length: int = 32
) -> RoutingRecorder:
    """How many experts a batch actually touches.

    With top-k of n, most experts receive nothing from a small batch. This is
    the number that determines how large a real training batch must be.
    """
    import torch

    recorder = RoutingRecorder()
    model.eval()
    with trainable_routing(arch, recorder=recorder):
        with torch.no_grad():
            model(input_ids=_batch(model, config, batch=batch, length=length))

    summary = recorder.summary()
    report.add(
        "routing statistics collected",
        recorder.experts_touched > 0,
        f"{recorder.experts_touched}/{recorder.n_experts} experts touched "
        f"({recorder.coverage * 100:.0f}%) across {recorder.calls} MoE layers, "
        f"imbalance {summary['imbalance']}",
        **summary,
    )
    return recorder


def check_determinism(report: Report, arch, model, config) -> None:
    """Two identical forwards under the patch must agree exactly."""
    import torch

    model.eval()
    input_ids = _batch(model, config, seed=31)
    with trainable_routing(arch):
        with torch.no_grad():
            first = model(input_ids=input_ids).logits.clone()
            second = model(input_ids=input_ids).logits.clone()
    delta = (first - second).abs().max().item()
    report.add(
        "patched routing is deterministic",
        delta == 0.0,
        f"max |Δ| = {delta:.3e} between repeated forwards",
    )


def check_optimizer_step(report: Report, arch, model, config) -> None:
    """One real step must change weights and keep the model finite."""
    import torch

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    try:
        with trainable_routing(arch):
            input_ids = _batch(model, config, seed=55)
            loss = model(input_ids=input_ids, labels=input_ids).loss
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except Exception as exc:  # noqa: BLE001
        report.add("optimizer step succeeds", False, f"{type(exc).__name__}: {exc}")
        model.eval()
        return

    changed = sum(
        1 for old, new in zip(before, trainable) if not torch.equal(old, new.detach())
    )
    finite = all(torch.isfinite(p).all().item() for p in trainable)
    report.add(
        "optimizer step succeeds",
        changed > 0 and finite,
        f"{changed}/{len(trainable)} tensors changed; all finite = {finite}",
        changed=changed,
        total=len(trainable),
    )
    model.eval()


def check_roundtrip(report: Report, arch, model, config) -> None:
    """A checkpoint trained under the patch must load under the *original* code.

    This is what the validator does, and it is the reason the patch must never
    touch the file on disk.
    """
    import torch

    model.train()
    try:
        with trainable_routing(arch):
            input_ids = _batch(model, config, seed=77)
            loss = model(input_ids=input_ids, labels=input_ids).loss
            loss.backward()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except Exception as exc:  # noqa: BLE001
        report.add("trained checkpoint round-trips", False, f"training failed: {exc}")
        model.eval()
        return
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "challenger"
        try:
            model.save_pretrained(path, safe_serialization=True)
        except Exception as exc:  # noqa: BLE001
            report.add("trained checkpoint round-trips", False, f"save failed: {exc}")
            return

        # Reload with the patch NOT active — exactly the validator's situation.
        try:
            reloaded = arch.causal_lm_cls.from_pretrained(path)
            reloaded.eval()
            input_ids = _batch(model, config, seed=91)
            with torch.no_grad():
                original_logits = model(input_ids=input_ids).logits
                reloaded_logits = reloaded(input_ids=input_ids).logits
        except Exception as exc:  # noqa: BLE001
            report.add(
                "trained checkpoint round-trips",
                False,
                f"reload under unpatched code failed: {type(exc).__name__}: {exc}",
            )
            return

        delta = (original_logits - reloaded_logits).abs().max().item()
        shards = sorted(p.name for p in path.glob("*.safetensors"))
        report.add(
            "trained checkpoint round-trips",
            math.isclose(delta, 0.0, abs_tol=1e-5),
            f"reloaded under unpatched code; max |Δlogit| = {delta:.3e}; "
            f"files: {', '.join(shards) or 'none'}",
            max_abs_delta=delta,
        )


def check_patch_restores(report: Report, arch) -> None:
    """The class must be left exactly as found, including after an exception."""
    original = arch.gate_cls.forward
    try:
        with trainable_routing(arch):
            raise RuntimeError("deliberate")
    except RuntimeError:
        pass
    report.add(
        "patch restores the original forward",
        arch.gate_cls.forward is original,
        "restored even when the block raises",
    )


# -- orchestration ----------------------------------------------------------


def run_all(arch, model, config, *, include_slow: bool = True) -> Report:
    """Every check, in dependency order."""
    report = Report()
    check_patch_restores(report, arch)
    check_guard(report, arch, model, config)
    check_eval_parity(report, arch, model, config)
    check_determinism(report, arch, model, config)
    check_gradients(report, arch, model, config)
    check_expert_coverage(report, arch, model, config)
    check_eval_mode_trick(report, arch, model, config)
    check_optimizer_step(report, arch, model, config)
    if include_slow:
        check_roundtrip(report, arch, model, config)
    return report
