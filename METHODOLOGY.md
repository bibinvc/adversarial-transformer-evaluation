# Mathematical and evaluation notes

## System and allowed changes

Let `x` be an 11-token sequence and `y` its opcode-family label. The first three tokens are inert fillers. `f_theta(x)` returns three classification logits, `h(x)` the final-block residual stream and `A(x)` the final-block causal attention probabilities. Model parameters `theta` are frozen during attacks.

The default white-box budget is `||delta||_infinity <= 0.75`. Only the first three input embeddings can change. Every PGD update is projected onto that box and zeroed outside the prefix. These are embedding-space sensitivity experiments, not necessarily token-realizable perturbations. FGSM takes one signed-gradient step from zero. PGD takes 16 signed-gradient steps of size `epsilon/4`, from one seeded uniform feasible starting point. It returns the final iterate, without multiple restarts or best-iterate selection; stronger search could find a different result.

Discrete search changes only the first three token IDs, each restricted to the eight filler IDs. The opcode, operands and task label remain unchanged. The same fitted three-token prefix is evaluated across the full held-out test split.

## Objectives

All attacks maximize their specified objective. For the last sequence position:

```text
CE(x') = cross_entropy(f_theta(x'), y)
D(x',x) = 1 - cosine_similarity(h(x')[-1], h(x)[-1])
H_A(x') = mean_over_heads entropy(A(x')[:, -1, :]) / log(sequence_length)
S(x',x) = D(x',x) + 0.1 H_A(x') + 0.1 JS(softmax(f(x')), softmax(f(x)))
combined(x',x) = CE(x') + 0.5 S(x',x)
```

CE-only, S-only and combined objectives use the same epsilon, steps and seeded initialization. The attention term concerns the final query, whose full sequence is causally visible; no forbidden future keys enter the entropy. Entropy and divergence use natural logarithms, so units are nats. The state objective is one operational proxy, not a mathematical definition of all possible model confusion. Its scale/weights are explicit fixed choices, not tuned on the test split.

The discrete gradient method computes gradients with respect to prefix embeddings on the attack-fitting batch, sums them across examples, and ranks filler-token embeddings by the local linear score `gradient dot embedding`. It tests up to four candidate tokens per coordinate with the real objective and accepts only an improving candidate. It stops when no improvement is found or eight iterations are reached. This is a small, explicitly bounded GCG-style approximation, without claims of reproducing a large-model jailbreak result.

The local output-only comparator enumerates the allowed coordinate substitutions on the independent target using **CE only**. Its search does not consume gradients, attention or residuals. The model is locally available for evaluation but the search function uses only logits and labels. Forward counts include every candidate/current-trigger scoring batch; example counts equal batch count times the fitting-batch size. For white-box search, additional gradient forward passes and backward counts are recorded. No commercial API or hidden model was queried.

## Measurements

- Accuracy and cross-entropy measure task performance.
- Relative residual drift is `||h_adv[-1]-h_clean[-1]||_2 / max(||h_clean[-1]||_2, 1e-12)` per example.
- Attention entropy averages the final query's per-head entropies.
- Logit JS divergence compares the clean and changed class-probability distributions.
- Spectral analysis centers final-position residuals across the held-out batch, computes singular values, and normalizes their squared values into an energy distribution. Effective rank is `exp(entropy(energy))`; the constant-matrix convention is zero. Spectra are descriptive, not evidence of a universal collapse mechanism.
- The transfer gap is `accuracy_change_source - accuracy_change_target`, with each change computed against that model's own clean accuracy. It is expressed as a fraction, not percentage points. Positive and negative gaps have no automatic significance interpretation.

Bootstrap intervals resample **paired held-out example indices** 2,000 times with a fixed seed. Accuracy-change intervals therefore compare attacked and clean correctness on the same examples. Other scalar metrics have example-bootstrap mean intervals. They are exploratory percentile intervals; they do not include training-seed uncertainty, model selection uncertainty or multiple-comparison correction. One initialization per source/target/student is a clear limitation.

## Ablations and interpretation

The random embedding comparator uses independent Rademacher directions scaled to match each corresponding combined-PGD perturbation's L2 norm. It stays within the same Linf budget and prefix positions. The fixed control prefix `[4, 5, 6]` is a discrete baseline, not a random draw or a norm match to embedding attacks.

A steering direction is the mean adversarial-minus-clean final residual **on the attack-fitting split**. Subtracting that fixed vector from test residuals checks a simple representation intervention. It is not asserted to recover all task behavior or to be a production defence.

The final-residual patch directly substitutes the clean final classifier input into an attacked forward pass. Restoring clean logits is expected by construction and is tested as an instrumentation identity. It must not be reported as independent evidence that a particular causal subspace has been discovered.

Fine-tuning and distillation results apply only to the specified local models and training procedures. The experiment does not establish transfer to closed-weight LLMs, durability under future model versions, or usefulness against an attacker analyzing real protected code. Those remain unimplemented research questions.
