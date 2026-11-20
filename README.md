# Adversarial Transformer Evaluation

A personal implementation project in PyTorch. It measures how constrained input changes affect a small causal transformer's task accuracy, activations, attention and output distributions. It includes runnable methods, invariant tests, an explicit evaluation protocol, per-example results and uncertainty estimates.

The task is deliberately small: classify the opcode family (`ADD`, `SUB` or `XOR`) in synthetic, fixed-length opcode-like sequences. This is **not** a generative LLM, a reverse-engineering tool, a DRM implementation or an experiment on protected software. Results do not establish the effectiveness of a software defence.

## Implemented

- A two-block causal transformer with explicit attention matrices, embeddings, residual streams and PyTorch forward hooks.
- Input gradients with frozen model weights; FGSM and random-start projected gradient ascent in embedding space.
- Three optimization objectives: task cross-entropy, an internal-state objective, and their weighted combination.
- Universal discrete prefixes fitted using a small GCG-style procedure: gradient-based candidate ranking followed by actual objective evaluations and greedy coordinate replacement. This is a simplified method inspired by GCG, not a reproduction of the full LLM paper.
- Restricted filler-token edits: program/opcode tokens stay fixed. Continuous embedding perturbations are a separate white-box threat model and need not correspond to valid tokens.
- Task accuracy/loss, relative final-residual drift, attention entropy, logit Jensen-Shannon divergence, singular-value spectra and effective rank.
- A random perturbation baseline matched to each example's attack L2 norm, plus a fixed discrete control prefix `[4, 5, 6]`.
- Discrete-prefix transfer from the source to a separately initialized model with a different hidden width; a local output-only coordinate-search comparator with explicit forward/example query counts.
- Fine-tuning and distillation stress tests, an attack-fitting-set-derived steering direction, and a final-residual patch sanity check.
- Paired example-bootstrap intervals and CSV records suitable for independent checking.

## Run

Tested with Python 3.12.14 and PyTorch 2.15.0+cpu. No model checkpoint, external dataset, API key, paid service or GPU is required.

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m unittest -v
.venv\Scripts\python experiment.py --output results
```

On Linux/macOS use `.venv/bin/python` instead. A pinned `torch` version should be used when exact cross-machine reproduction matters; the installed version is recorded in every report. CPU threading and random seeds are fixed. Results may still differ across PyTorch/BLAS/platform versions.

`results/report.json` records the configuration, software versions, split hashes, architectures, method outputs, confidence intervals and scope limits. `results/per_example.csv` contains one row per held-out example and experiment. Model weights are not committed. See [RESULTS.md](RESULTS.md) for the measured default-run summary.

## Evaluation design

The default experiment uses 512 training, 48 attack-fitting and 96 held-out test examples. The immutable program body is unique across these splits. Labels are determined exclusively by the opcode. Training data deliberately contains an imperfect correlation between inert prefix tokens and labels; this provides an auditable setting for investigating shortcut sensitivity. All edits preserve the opcode and body. The synthetic setup is a design choice, not an approximation validated against real software or language tasks.

The source has hidden width 32, the independent transfer target width 40, and the distilled student width 24; each has four attention heads and two blocks. All train on the same training split. The target and student have separate initializations. The fine-tuned model starts from the source weights and receives 30 extra training steps with randomized prefixes. The student uses both hard labels and temperature-2 teacher probabilities. This tests specified local interventions; it does not predict robustness after arbitrary fine-tuning, distillation or architecture changes.

Universal prefixes, the output-only comparator and the steering vector use **only the attack-fitting split**. Held-out examples are not used to fit those shared interventions or select hyperparameters. Per-example white-box embedding attacks use test labels as part of the stated attack oracle, as is conventional for this attack evaluation; they are not deployable defences and do not produce a universally reusable token prefix.

See [METHODOLOGY.md](METHODOLOGY.md) for the objective, norms, transfer-gap definition and interpretation limits.

## Checks and results

The unit suite checks split separation and semantic invariance, causal masking, hook cleanup, nonzero input gradients and a finite-difference comparison, frozen parameters, attack budgets and positions, objective differentiability, matched random norms, distribution metrics, monotonic trigger acceptance, query accounting, deterministic seeds and patch identity.

Generated results should be read together with the model's clean accuracy. A change in activations, attention entropy or spectrum alone is **not proof of degraded task performance**. The report includes both quantities rather than equating them.

## Scope and authorship

This project is intended for study and reproducible portfolio review. No employer system, private data, proprietary binary, commercial LLM, closed-weight API or real DRM stack is involved. There is no publication or prior professional deployment claim. Future work could evaluate language-model tasks, stronger independent training seeds and external model families after suitable compute and a defined authorization/safety scope are available.

## Primary references

- [PyTorch autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html) and [Module forward hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).
- Goodfellow et al., [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), FGSM.
- Madry et al., [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083), constrained iterative attacks and robust optimization.
- Zou et al., [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043), gradient-guided discrete candidate search.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), scaled dot-product attention and transformer structure.
