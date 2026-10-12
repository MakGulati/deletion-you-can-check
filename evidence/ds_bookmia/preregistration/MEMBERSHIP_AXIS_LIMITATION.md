# The unseen population is the wrong counterfactual — demonstrated

**Superseded 2026-08-10.** An earlier version of this file recorded the
membership-axis deviation on BookMIA as an *undiagnosed* limitation. The
diagnostic experiment has since been run and the mechanism is established. The
finding is stronger and narrower than the limitation it replaces.

## The observation

Exact retraining is a **perfect deletion** — the forget set was never trained
on — so a calibrated membership axis must read chance on it. Measured with
`loss_ratio` on forget-vs-unseen:

| study | measured on exact retraining |
|---|---|
| ArxivMIA / Llama-3.2-1B | 0.474 ± 0.003 (chance) |
| BookMIA / Qwen2.5-1.5B | **0.257 ± 0.023** |

## The cause, now demonstrated rather than hypothesised

BookMIA's unseen population is `label=0` — a **different set of books** from the
`label=1` forget and retain sets. "Unseen" therefore differed from "forgotten"
in two ways at once: never-trained *and* differently distributed.

Re-running the axis against an unseen population of **held-out members**
(`label=1` records the split never drew, disjoint from forget, retain and the
routine-update slice — never trained on by anything, and identically
distributed):

| condition | vs `label=0` non-members | vs held-out members |
|---|---|---|
| **exact retraining** (perfect deletion) | 0.257 ± 0.023 | **0.523 ± 0.025** |
| exact retraining + routine update | 0.233 ± 0.021 | 0.523 ± 0.018 |
| null refit (nothing deleted) | 0.032 ± 0.004 | 0.097 ± 0.006 |
| routine update (nothing deleted) | 0.039 ± 0.005 | 0.115 ± 0.010 |
| GA+GD weak | 0.110 ± 0.071 | 0.181 ± 0.031 |
| NPO | 0.252 ± 0.022 | 0.346 ± 0.019 |
| SCRUB | 0.445 ± 0.090 | 0.335 ± 0.026 |
| GA+GD aggressive (2/3 seeds collapsed) | 0.572 ± 0.089 | 0.501 ± 0.019 |

**The gold standard returns to chance: 0.523, inside the preregistered
[0.45, 0.55] region.** No threshold had to move. The population definition was
wrong, not the threshold.

The whole axis becomes coherent at once. Nothing-deleted conditions read
strongly member-like (0.097, 0.115); a perfect deletion reads chance; partial
methods sit in between in the order their RNFF implies.

## The methodological error this identifies

Unlearning evaluations routinely inherit their "unseen" population from a
membership-inference benchmark, where non-members are the correct choice. **For
unlearning evaluation it is the wrong counterfactual.** The question is not "did
this record come from the member distribution?" but "does this *forgotten*
record still look like a *trained* one?" — and the only control that answers it
is a record that is never-trained but otherwise identical.

The fix is cheap: draw the unseen population from held-out members of the same
pool. ArxivMIA cannot do it — its 1000 members are fully consumed by the split,
which is why the error was invisible there and why it happened to be benign.
BookMIA has 4935, which is what made the diagnosis possible.

## Why a base-model check does not catch it

The pre-flight asked whether the *pretrained base* separates members from
unseen: it does not (0.4597, matching ArxivMIA's 0.4626), so it passed. The
distortion is created by **fine-tuning on retain**, which improves the model on
`label=1` prose generally and lowers the forget set's loss without the forget
set ever being trained on. Only a model that has actually been fine-tuned —
i.e. the exact-retraining control — exposes it.

A base-model check is necessary but not sufficient for validating a membership
axis on a new corpus.

## The second finding, which the corrected axis makes visible

With the counterfactual fixed, compare two rows:

- **exact retraining** — a perfect deletion — reads **0.523**
- **GA+GD aggressive** — which tripped the collapse guard on 2 of 3 seeds and
  failed utility on all 3 — reads **0.501**

A correctly calibrated membership metric **still cannot tell a perfect deletion
from a destroyed model.** Fixing the counterfactual makes the axis interpretable;
it does not make it sufficient. That is the argument for the conjunction, and it
is now supported by a calibrated measurement rather than by a mis-calibrated one.

## Status and reporting rules

- The primary BookMIA tables keep the **preregistered** `label=0` axis. Nothing
  is retroactively rescored into the headline verdicts.
- This diagnostic is reported beside them as the explanation, clearly marked
  post-hoc: it was designed after the controls were seen.
- The preregistered [0.45, 0.55] region is **not** recentred anywhere. It did
  not need to be.
- **Replication criterion 1 still fails as preregistered** on BookMIA, and is
  reported as failing. The diagnostic explains *why* it failed; it does not
  convert the failure into a pass.
- The object-level conclusion is unaffected: **no approximate method passes the
  conjunction on either corpus, under either the preregistered or the
  measured-null calibration** (verified in `ds_cross_dataset/cross_dataset.json`).

Artifacts: `ds_bookmia/membership_null/membership_null_check.json`,
`deletion_signals/scripts/membership_null_check.py`.
