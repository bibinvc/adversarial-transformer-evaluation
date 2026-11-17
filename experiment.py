"""Small, auditable adversarial-transformer experiment; CPU and synthetic data only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

PREFIX = 3
FILLERS = tuple(range(8))
BEGIN, OP_START, OPERAND_START, END, QUERY = 8, 9, 12, 28, 29
VOCAB, LENGTH = 30, 11


@dataclass(frozen=True)
class Config:
    seed: int = 17
    train_size: int = 512
    fit_size: int = 48
    test_size: int = 96
    train_steps: int = 100
    epsilon: float = 0.75
    pgd_steps: int = 16
    gcg_steps: int = 8
    bootstrap_samples: int = 2000


def deterministic(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def dataset(config: Config) -> dict[str, tuple[Tensor, Tensor]]:
    """No body is repeated across splits; labels depend only on the opcode."""
    rng = random.Random(config.seed)
    seen: set[tuple[int, ...]] = set()
    result = {}
    for name, size in (("train", config.train_size), ("attack_fit", config.fit_size),
                       ("test", config.test_size)):
        rows, labels = [], []
        while len(rows) < size:
            label = len(rows) % 3
            operands = tuple(rng.randrange(16) + OPERAND_START for _ in range(4))
            body = (BEGIN, OP_START + label, *operands, END, QUERY)
            if body in seen:
                continue
            seen.add(body)
            # A deliberately imperfect correlation exposes shortcut learning.
            prefix = [label if rng.random() < 0.7 else rng.randrange(8) for _ in range(PREFIX)]
            rows.append(prefix + list(body))
            labels.append(label)
        result[name] = (torch.tensor(rows, dtype=torch.long), torch.tensor(labels))
    return result


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads, self.width = heads, width
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.attention: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        split = lambda t: t.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        scores = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        self.attention = scores.masked_fill(future, float("-inf")).softmax(-1)
        attended = (self.attention @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.out(attended)
        return x + self.ff(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, width: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.width, self.heads, self.layers = width, heads, layers
        self.embedding = nn.Embedding(VOCAB, width)
        self.position = nn.Parameter(torch.randn(LENGTH, width) * 0.02)
        self.blocks = nn.ModuleList(AttentionBlock(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 3)

    def forward(self, ids: Tensor | None = None, embeddings: Tensor | None = None) -> dict[str, Tensor]:
        if embeddings is None:
            if ids is None:
                raise ValueError("ids or embeddings are required")
            embeddings = self.embedding(ids)
        x = embeddings + self.position
        captured: dict[str, Tensor] = {}
        # A genuine PyTorch forward hook captures the final residual stream.
        handle = self.blocks[-1].register_forward_hook(lambda module, args, output: captured.update(residual=output))
        try:
            for block in self.blocks:
                x = block(x)
        finally:
            handle.remove()
        return {"logits": self.classifier(self.norm(x[:, -1])),
                "residual": captured["residual"], "attention": self.blocks[-1].attention}


def freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def train_model(model: TinyTransformer, rows: Tensor, labels: Tensor, seed: int, steps: int,
                teacher: TinyTransformer | None = None, randomize_prefix: bool = False) -> TinyTransformer:
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    for _ in range(steps):
        ix = torch.randint(len(rows), (min(64, len(rows)),), generator=generator)
        x, y = rows[ix].clone(), labels[ix]
        if randomize_prefix:
            x[:, :PREFIX] = torch.randint(8, (len(x), PREFIX), generator=generator)
        logits = model(x)["logits"]
        loss = F.cross_entropy(logits, y)
        if teacher is not None:
            with torch.no_grad():
                target = (teacher(x)["logits"] / 2).softmax(-1)
            soft_loss = F.kl_div((logits / 2).log_softmax(-1), target, reduction="batchmean") * 4
            loss = 0.5 * loss + 0.5 * soft_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return freeze(model)


def entropy(probabilities: Tensor) -> Tensor:
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)


def js_divergence(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    p, q = p_logits.softmax(-1), q_logits.softmax(-1)
    m = ((p + q) / 2).clamp_min(1e-12)
    return 0.5 * ((p * (p.clamp_min(1e-12).log() - m.log())).sum(-1)
                  + (q * (q.clamp_min(1e-12).log() - m.log())).sum(-1))


def state_loss(output: dict[str, Tensor], clean: dict[str, Tensor]) -> Tensor:
    drift = 1 - F.cosine_similarity(output["residual"][:, -1], clean["residual"][:, -1], dim=-1)
    attention_entropy = entropy(output["attention"][:, :, -1]).mean(-1) / math.log(LENGTH)
    return drift + 0.1 * attention_entropy + 0.1 * js_divergence(output["logits"], clean["logits"])


def objective(output: dict[str, Tensor], clean: dict[str, Tensor], labels: Tensor, kind: str) -> Tensor:
    ce = F.cross_entropy(output["logits"], labels, reduction="none")
    if kind == "ce":
        return ce.mean()
    if kind == "state":
        return state_loss(output, clean).mean()
    if kind == "combined":
        return (ce + 0.5 * state_loss(output, clean)).mean()
    raise ValueError(kind)


def detach_output(output: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach() for key, value in output.items()}


def project(delta: Tensor, epsilon: float) -> Tensor:
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    result = delta.clamp(-epsilon, epsilon).clone()
    result[:, PREFIX:] = 0
    return result


def embedding_attack(model: TinyTransformer, rows: Tensor, labels: Tensor, epsilon: float,
                     steps: int = 1, kind: str = "ce", seed: int = 0) -> Tensor:
    """Untargeted FGSM when steps=1; random-start projected gradient ascent otherwise."""
    base = model.embedding(rows).detach()
    if epsilon == 0:
        return torch.zeros_like(base)
    with torch.no_grad():
        clean = detach_output(model(embeddings=base))
    generator = torch.Generator().manual_seed(seed)
    delta = torch.zeros_like(base)
    if steps > 1:
        delta = project((torch.rand(base.shape, generator=generator) * 2 - 1) * epsilon, epsilon)
    step_size = epsilon if steps == 1 else epsilon / 4
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        loss = objective(model(embeddings=base + delta), clean, labels, kind)
        gradient, = torch.autograd.grad(loss, delta, only_inputs=True)
        delta = project(delta + step_size * gradient.sign(), epsilon)
    return delta.detach()


def random_matched(delta: Tensor, seed: int) -> Tensor:
    """Rademacher perturbations exactly match each attack's L2 norm and preserve its Linf budget."""
    generator = torch.Generator().manual_seed(seed)
    result = torch.zeros_like(delta)
    amplitude = delta[:, :PREFIX].flatten(1).norm(dim=-1) / math.sqrt(PREFIX * delta.shape[-1])
    signs = torch.randint(2, delta[:, :PREFIX].shape, generator=generator) * 2 - 1
    result[:, :PREFIX] = signs * amplitude[:, None, None]
    return result


def apply_trigger(rows: Tensor, trigger: Tensor) -> Tensor:
    result = rows.clone()
    result[:, :PREFIX] = trigger
    return result


def trigger_search(model: TinyTransformer, rows: Tensor, labels: Tensor, steps: int = 8,
                   shortlist: int = 4, kind: str = "combined", gradient_guided: bool = True) -> dict:
    """One universal trigger, fitted solely on attack_fit; accepted steps never decrease objective."""
    trigger = torch.full((PREFIX,), 7, dtype=torch.long)
    request_count = 0
    example_queries = 0
    backward_passes = 0
    clean = {}
    if kind != "ce":
        with torch.no_grad():
            clean = detach_output(model(rows))
        request_count += 1
        example_queries += len(rows)

    def score(t: Tensor) -> float:
        nonlocal request_count, example_queries
        request_count += 1
        example_queries += len(rows)
        with torch.no_grad():
            out = model(apply_trigger(rows, t))
            return float(objective(out, clean, labels, kind))

    value = score(trigger)
    history = [value]
    for _ in range(steps):
        candidates: list[tuple[int, int]] = []
        if gradient_guided:
            # Gradient of shared-token loss is summed across the fitting batch.
            embedded = model.embedding(apply_trigger(rows, trigger)).detach().requires_grad_(True)
            out = model(embeddings=embedded)
            request_count += 1
            example_queries += len(rows)
            gradient, = torch.autograd.grad(objective(out, clean, labels, kind), embedded)
            backward_passes += 1
            token_scores = gradient[:, :PREFIX].sum(0) @ model.embedding.weight[:8].detach().T
            for position in range(PREFIX):
                for token in token_scores[position].topk(min(shortlist, 8)).indices.tolist():
                    candidates.append((position, token))
        else:
            # Output-only comparison: enumerate tokens, never use target gradients or states.
            candidates = [(position, token) for position in range(PREFIX) for token in FILLERS]
        best_value, best_trigger = value, trigger
        for position, token in candidates:
            if token == int(trigger[position]):
                continue
            proposal = trigger.clone()
            proposal[position] = token
            proposed_value = score(proposal)
            if proposed_value > best_value + 1e-9:
                best_value, best_trigger = proposed_value, proposal
        trigger, value = best_trigger, best_value
        history.append(value)
        if history[-1] <= history[-2] + 1e-9:
            break
    return {"trigger": trigger.tolist(), "objective_history": history,
            "batch_forward_queries": request_count, "example_logit_queries": example_queries,
            "backward_passes": backward_passes, "gradient_guided": gradient_guided, "objective": kind}


def spectrum(residual: Tensor) -> dict:
    matrix = residual[:, -1].double()
    matrix = matrix - matrix.mean(0, keepdim=True)
    values = torch.linalg.svdvals(matrix)
    energy = values.square()
    if float(energy.sum()) <= 1e-15:
        return {"singular_values": values.tolist(), "effective_rank": 0.0, "leading_energy_fraction": 0.0}
    energy /= energy.sum()
    return {"singular_values": values.tolist(), "effective_rank": float(entropy(energy).exp()),
            "leading_energy_fraction": float(energy[0])}


def metrics(clean: dict[str, Tensor], attacked: dict[str, Tensor], labels: Tensor) -> dict[str, Tensor]:
    before, after = clean["residual"][:, -1], attacked["residual"][:, -1]
    return {"correct": (attacked["logits"].argmax(-1) == labels).float(),
            "loss": F.cross_entropy(attacked["logits"], labels, reduction="none"),
            "residual_relative_drift": (after - before).norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-12),
            "attention_entropy_nats": entropy(attacked["attention"][:, :, -1]).mean(-1),
            "logit_js_nats": js_divergence(clean["logits"], attacked["logits"])}


def bootstrap_mean(values: Tensor, count: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(values), (count, len(values)), generator=generator)
    distribution = values.double()[ix].mean(-1)
    return {"mean": float(values.double().mean()),
            "ci95": [float(distribution.quantile(0.025)), float(distribution.quantile(0.975))]}


def patched_forward(model: TinyTransformer, rows: Tensor, replacement: Tensor | None = None,
                    direction: Tensor | None = None) -> dict[str, Tensor]:
    def patch(module, inputs, output):
        changed = output.clone()
        if replacement is not None:
            changed[:, -1] = replacement
        if direction is not None:
            changed[:, -1] = changed[:, -1] - direction
        return changed
    handle = model.blocks[-1].register_forward_hook(patch)
    try:
        return model(rows)
    finally:
        handle.remove()


def run(config: Config, output_dir: Path) -> dict:
    deterministic(config.seed)
    data = dataset(config)
    train_x, train_y = data["train"]
    fit_x, fit_y = data["attack_fit"]
    test_x, test_y = data["test"]
    source = train_model(TinyTransformer(), train_x, train_y, 101, config.train_steps)
    torch.manual_seed(29)
    target = train_model(TinyTransformer(width=40), train_x, train_y, 202, config.train_steps)
    # New module plus copied state avoids deepcopy of cached non-leaf attention tensors.
    finetuned = TinyTransformer()
    finetuned.load_state_dict(source.state_dict())
    finetuned = train_model(finetuned, train_x, train_y, 303, 30, randomize_prefix=True)
    torch.manual_seed(41)
    distilled = train_model(TinyTransformer(width=24), train_x, train_y, 404, config.train_steps, teacher=source)
    models = {"source": source, "independent_target": target, "fine_tuned_source": finetuned,
              "distilled_student": distilled}
    report = {"config": asdict(config), "runtime": {"python": platform.python_version(), "torch": torch.__version__,
               "device": "cpu", "threads": 1}, "split_sha256": {}, "models": {}, "experiments": {},
              "training": {"source": {"initialization_seed": config.seed, "batch_seed": 101, "steps": config.train_steps},
                           "independent_target": {"initialization_seed": 29, "batch_seed": 202, "steps": config.train_steps},
                           "fine_tuned_source": {"starts_from": "source", "batch_seed": 303, "extra_steps": 30,
                                                 "randomized_training_prefix": True},
                           "distilled_student": {"initialization_seed": 41, "batch_seed": 404, "steps": config.train_steps,
                                                 "teacher": "source", "temperature": 2, "hard_loss_weight": 0.5}},
              "limitations": ["Synthetic opcode classifier, not a generative LLM or binary-analysis system.",
                              "No DRM, proprietary code, closed-weight API, or future architecture evaluated.",
                              "One training seed per model; example-bootstrap intervals exclude training-seed uncertainty.",
                              "Internal-state drift and entropy are proxies; they do not prove semantic confusion.",
                              "Fine-tuning/distillation are small local stress tests, not durability guarantees."]}
    for name, (rows, labels) in data.items():
        serialized = json.dumps([rows.tolist(), labels.tolist()], separators=(",", ":")).encode("utf-8")
        report["split_sha256"][name] = hashlib.sha256(serialized).hexdigest()
    with torch.no_grad():
        clean = detach_output(source(test_x))
        clean_by_model = {name: detach_output(model(test_x)) for name, model in models.items()}
    for name, model in models.items():
        report["models"][name] = {"width": model.width, "layers": model.layers, "heads": model.heads,
                                  "parameters": sum(p.numel() for p in model.parameters()),
                                  "clean_accuracy": float((clean_by_model[name]["logits"].argmax(-1) == test_y).float().mean())}
    raw_rows = []

    def record(name: str, out: dict[str, Tensor], baseline: dict[str, Tensor] = clean,
               extra: dict | None = None) -> None:
        measurements = metrics(baseline, out, test_y)
        baseline_correct = (baseline["logits"].argmax(-1) == test_y).float()
        entry = {key: bootstrap_mean(value, config.bootstrap_samples, 700 + i)
                 for i, (key, value) in enumerate(measurements.items())}
        entry["paired_accuracy_change"] = bootstrap_mean(measurements["correct"] - baseline_correct,
                                                         config.bootstrap_samples, 800)
        entry["residual_spectrum"] = spectrum(out["residual"])
        if extra:
            entry.update(extra)
        report["experiments"][name] = entry
        for index in range(len(test_y)):
            raw_rows.append({"experiment": name, "example_index": index, "label": int(test_y[index]),
                             "baseline_correct": int(baseline_correct[index]),
                             **{key: float(value[index]) for key, value in measurements.items()}})

    record("clean", clean)
    delta_combined = None
    for kind in ("ce", "state", "combined"):
        delta = embedding_attack(source, test_x, test_y, config.epsilon, config.pgd_steps, kind, 500)
        with torch.no_grad():
            out = detach_output(source(embeddings=source.embedding(test_x) + delta))
        record("pgd_" + kind, out, extra={"prefix_linf_max": float(delta.abs().max()),
                                         "prefix_l2_mean": float(delta.flatten(1).norm(dim=-1).mean())})
        if kind == "combined":
            delta_combined = delta
    fgsm = embedding_attack(source, test_x, test_y, config.epsilon, 1, "ce")
    with torch.no_grad():
        record("fgsm_ce", detach_output(source(embeddings=source.embedding(test_x) + fgsm)))
        noise = random_matched(delta_combined, 501)
        record("random_l2_matched", detach_output(source(embeddings=source.embedding(test_x) + noise)),
               extra={"maximum_l2_match_error": float((noise.flatten(1).norm(dim=-1)
                                                        - delta_combined.flatten(1).norm(dim=-1)).abs().max())})
    universal = trigger_search(source, fit_x, fit_y, config.gcg_steps)
    report["universal_trigger_fit"] = universal
    trigger = torch.tensor(universal["trigger"])
    attacked_x = apply_trigger(test_x, trigger)
    with torch.no_grad():
        for name, model in models.items():
            record("universal_" + name, detach_output(model(attacked_x)), clean_by_model[name])
        # The direction is estimated on attack_fit only, never on test examples.
        fit_clean = source(fit_x)["residual"][:, -1]
        fit_attacked = source(apply_trigger(fit_x, trigger))["residual"][:, -1]
        direction = (fit_attacked - fit_clean).mean(0)
        record("universal_source_steering", detach_output(patched_forward(source, attacked_x, direction=direction)))
        record("final_residual_patch_sanity", detach_output(patched_forward(source, attacked_x,
                                                                           replacement=clean["residual"][:, -1])),
               extra={"note": "Restores the final classifier input; an identity sanity check, not a deployable defence."})
        control_prefix = torch.tensor([4, 5, 6])
        record("fixed_control_prefix", detach_output(source(apply_trigger(test_x, control_prefix))))
    black_box = trigger_search(target, fit_x, fit_y, config.gcg_steps, kind="ce", gradient_guided=False)
    report["local_output_only_fit"] = black_box
    with torch.no_grad():
        record("output_only_target", detach_output(target(apply_trigger(test_x, torch.tensor(black_box["trigger"])))),
               clean_by_model["independent_target"])
    source_delta = report["experiments"]["universal_source"]["paired_accuracy_change"]["mean"]
    target_delta = report["experiments"]["universal_independent_target"]["paired_accuracy_change"]["mean"]
    report["transfer_gap_accuracy_change_source_minus_target"] = source_delta - target_delta
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "per_example.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--train-steps", type=int, default=100)
    args = parser.parse_args()
    report = run(Config(train_steps=args.train_steps), args.output)
    print("Model clean accuracies:", {k: round(v["clean_accuracy"], 4) for k, v in report["models"].items()})
    for name, entry in report["experiments"].items():
        print(f"{name}: accuracy={entry['correct']['mean']:.4f}, drift={entry['residual_relative_drift']['mean']:.4f}")
    print("Saved", args.output / "report.json", "and", args.output / "per_example.csv")


if __name__ == "__main__":
    main()
