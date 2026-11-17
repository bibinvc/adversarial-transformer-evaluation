# Measured default run

Run on 17 November 2026 with Python 3.12.14, PyTorch 2.15.0+cpu, one CPU thread and the default configuration. The unit suite passed **24 tests**. These are results from one training realization per model variant, not a benchmark on large language models.

A second default run in the same environment reproduced all 14 experiments' numerical results exactly. This checks local repeatability, not robustness across new seeds or software versions.

All four models classified all 96 clean held-out synthetic examples correctly. The table reports source-model results unless a target is named. Accuracy-change confidence intervals use paired example bootstrapping; they exclude training-seed and model-selection uncertainty.

| Method | Held-out accuracy | Change from clean, percentage points | 95% paired bootstrap interval for change | Mean relative residual drift |
| --- | ---: | ---: | --- | ---: |
| Clean | 100.00% | 0.00 | [0.00, 0.00] | 0.0000 |
| FGSM, CE | 94.79% | -5.21 | [-9.38, -1.04] | 0.4335 |
| PGD, CE | 58.33% | -41.67 | [-52.08, -31.25] | 0.8657 |
| PGD, state objective | 59.38% | -40.63 | [-51.04, -31.25] | 0.8459 |
| PGD, combined objective | 59.38% | -40.63 | [-51.04, -31.25] | 0.8405 |
| Random embedding noise, matched L2 | 100.00% | 0.00 | [0.00, 0.00] | 0.0769 |
| Universal discrete prefix, source | 100.00% | 0.00 | [0.00, 0.00] | 0.3108 |
| Same discrete prefix, independent target | 100.00% | 0.00 | [0.00, 0.00] | 0.3753 |
| Local output-only search, target | 100.00% | 0.00 | [0.00, 0.00] | 0.4383 |

Embedding attacks affect only the three filler-prefix positions, with an Linf budget of 0.75. PGD uses 16 steps. Random noise matches the combined-PGD per-example L2 norms to within approximately 9.54e-7. A change in embedding space need not decode to a valid token sequence.

The shared gradient-guided prefix was `[2, 2, 2]`. Fitting used 42 batched forward evaluations (2,016 example evaluations) and four backward passes on the 48-example fitting split. The output-only target comparator selected `[1, 1, 1]` using 85 batched forward evaluations (4,080 example logit evaluations), with no backward passes.

**The discrete prefixes did not degrade held-out classification accuracy.** Their residual drift is therefore evidence of changed representations only. The source-to-target accuracy-change gap was zero; there is no demonstrated transfer of a successful discrete attack. The independent target, fine-tuned source and distilled student all remained at 100% under the fitted source prefix. This is a useful negative result rather than evidence of future-model durability.

The fixed steering vector reduced the source prefix's mean relative residual drift from 0.3108 to 0.2631; accuracy was already 100% and remained 100%. Final-residual patching restored drift to zero and clean logits as expected by construction. This patch is an instrumentation sanity check, not a deployable defence or a discovered causal explanation.

Detailed values, attention entropies, JS divergences, singular values and all exploratory intervals are in [results/report.json](results/report.json). Individual observations are in [results/per_example.csv](results/per_example.csv). See [METHODOLOGY.md](METHODOLOGY.md) for formulas and the explicit scope limits.
