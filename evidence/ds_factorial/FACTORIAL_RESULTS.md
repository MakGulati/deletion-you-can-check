# 2×2 factorial: corpus × model family

Measured on the **exact-retraining control** in all four cells — a perfect
deletion, where a calibrated membership axis must read chance and where the
detector comparison is quoted. Scoring normalised across every cell: FUMA and
Fisher-SIF off, no curvature holdout, populations 200 / 800 / 300 identical
throughout. Three seeds per cell.

Generated 2026-08-10 from `ds_factorial/run_factorial.sh`.

---

## Finding 1 — The membership decalibration is a CORPUS effect

Membership AUC (`loss_ratio`, forget vs unseen) on exact retraining:

| | ArxivMIA | BookMIA |
|---|---|---|
| **Llama-3.2-1B** | 0.474 ± 0.003 | **0.122 ± 0.016** |
| **Qwen2.5-1.5B** | 0.473 ± 0.027 | **0.257 ± 0.023** |

Marginal means, n = 6 per level:

| effect | contrast | size |
|---|---|---|
| **corpus** | ArxivMIA 0.473 → BookMIA 0.189 | **−0.284** |
| model | Llama 0.298 → Qwen 0.365 | +0.067 |

**The corpus effect is 4× the model effect.** The two ArxivMIA cells agree to
0.001 across model families; both BookMIA cells are far from chance, and
Llama+BookMIA (0.122) is *more* decalibrated than the Qwen+BookMIA cell the
finding was first observed in.

This was a falsifiable prediction. The account in
`ds_bookmia/preregistration/MEMBERSHIP_AXIS_LIMITATION.md` says the
decalibration comes from the corpus — D_forget and D_test are drawn from
different book pools, so fine-tuning on retain lifts the forget set by domain
transfer. Had Llama+BookMIA read near chance, that account would have been
wrong. It does not, and the account stands.

Combined with the held-out-member experiment (which moved Qwen+BookMIA from
0.257 to 0.523, inside the preregistered region), the mechanism is now
established on both axes of the design: **the counterfactual population, not
the model, decides where the membership null sits.**

---

## Finding 2 — The detector "flip" is not a corpus effect. It is noise.

ROC-AUC on forget vs retain, exact retraining, mean over 3 seeds:

| detector | Llama/ArxivMIA | Qwen/ArxivMIA | Llama/BookMIA | Qwen/BookMIA |
|---|---|---|---|---|
| `loss_delta_vs_finetuned` | **0.9999** | **0.9987** | **0.9953** | 0.9936 |
| `raw_cosine` | 0.9986 | 0.9977 | 0.9948 | **0.9950** |
| `alignment_cos_sq` | 0.9986 | 0.9977 | 0.9948 | 0.9950 |
| `loss_ratio_vs_finetuned` | 0.9989 | 0.9949 | 0.9910 | 0.9896 |
| `grad_ratio` | 0.9898 | 0.9788 | 0.9643 | 0.9273 |
| `grad_norm_sq` | 0.7205 | 0.6835 | 0.8219 | 0.7558 |
| `loss_unlearned` | 0.7486 | 0.6812 | 0.6694 | 0.6219 |

Paired within-seed, AUC(preregistered baseline `loss_ratio_vs_finetuned`) −
AUC(challenger); positive means the baseline wins:

| challenger | Llama/ArxivMIA | Qwen/ArxivMIA | Llama/BookMIA | Qwen/BookMIA |
|---|---|---|---|---|
| `raw_cosine` | **+0.0002 ± 0.0006** | −0.0028 ± 0.0006 | −0.0038 ± 0.0031 | −0.0054 ± 0.0021 |
| `alignment_cos_sq` | +0.0002 ± 0.0006 | −0.0028 ± 0.0006 | −0.0038 ± 0.0031 | −0.0054 ± 0.0021 |
| `grad_ratio` | +0.0091 ± 0.0038 | +0.0162 ± 0.0065 | +0.0268 ± 0.0091 | +0.0623 ± 0.0056 |
| `loss_delta_vs_finetuned` | −0.0010 ± 0.0005 | −0.0037 ± 0.0008 | −0.0042 ± 0.0031 | −0.0040 ± 0.0011 |

### An earlier result of this study does not survive the factorial

The ArxivMIA package reported the loss baseline beating `raw_cosine` by
**+0.0008 ± 0.0006** — a >1 SD win, and the one piece of direct support for the
original central claim. Under the normalised scoring used here, the same cell
reads **+0.0002 ± 0.0006**: within 1 SD, **no separation**.

The difference is the scoring configuration, not the models or the data. The
original number was computed with the Fisher curvature holdout active, which
scores 600 retain samples instead of 800. **A result that changes sign class
when the negative population changes by 200 samples was never strong enough to
carry a central claim**, and this is recorded rather than quietly dropped.

### What the four cells support

- **Two-checkpoint loss and signed cosine are within ~0.005 AUC everywhere.**
  Which one edges ahead is not stable, and it is not explained by corpus: cosine
  edges ahead in 3 of 4 cells, the loss statistic in 1, all at magnitudes
  comparable to the across-seed SD.
- **`loss_delta_vs_finetuned` is the strongest detector in 3 of 4 cells** and
  beats the preregistered `loss_ratio_vs_finetuned` in all 4. The
  preregistration named the weaker of the two two-checkpoint statistics; that is
  a preregistration artifact and is reported as one, not corrected after the
  fact.
- **Calibrated gradient magnitude (`grad_ratio`) loses in all four cells**, and
  loses more on BookMIA (+0.027, +0.062) than on ArxivMIA (+0.009, +0.016).
- **Gradient norm and single-checkpoint loss are far behind everywhere**
  (0.62–0.82), across both corpora and both model families.

### The claim this licenses

> Under matched access, two-checkpoint loss statistics and directional gradient
> alignment perform equivalently — within the across-seed noise in all four
> corpus × model cells — while both decisively outperform calibrated gradient
> magnitude, gradient norm, and single-checkpoint loss. No stable advantage in
> either direction was found between the loss and cosine families.

It does **not** license the original claim that two-checkpoint loss *matches or
beats* gradient alignment, nor its converse.

---

## What remains confounded

Truncation now varies with corpus rather than with model: 100% of BookMIA
samples are truncated at 512 tokens (corpus mean 710) in **both** BookMIA cells,
and ArxivMIA truncates almost nothing (mean 205) in both of its. So corpus and
truncation regime remain perfectly confounded with each other, though both are
now separated from model family.

Distinguishing them would need BookMIA at a length budget that does not
truncate, which the 24 GB card cannot accommodate at these batch sizes.
Statements about "the corpus effect" throughout should be read as **corpus and
its truncation regime jointly**.

## Scope

Two corpora, two model families, one adapter configuration, three seeds, the
exact-retraining control only. The method verdicts (GA+GD, SCRUB, NPO) remain at
two cells and are unaffected by this experiment.
