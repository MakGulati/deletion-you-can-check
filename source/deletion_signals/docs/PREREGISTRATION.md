# Preregistration: GA+GD unlearning-strength sweep

**Committed before the sweep ran.** Everything below — the question, the
thresholds, the decision rule, and the list of outcomes that count as results —
was fixed in advance. The analysis code reads these thresholds from
`sweep_report.py` module constants that mirror this document; changing a verdict
requires changing both, in a commit that says so.

## What the baseline runs established, and what they did not

Three seeds, GA+GD and SCRUB, ArxivMIA + Llama-3.2-1B QLoRA:

- On `forget_vs_unseen`, calibrated `loss_ratio` separates forgotten samples
  from never-trained ones essentially perfectly. Under the preregistered
  higher-is-forget direction that reads **AUC 0.001** — i.e. 0.999 in the
  membership direction: a forgotten sample still has the *lower* loss of a
  fitted member. On that task the metric is measuring **membership**, not
  deletion, and it is reported below 0.5 rather than flipped.
- On `forget_vs_retain` — where both classes are known members, so membership
  cannot explain a result — signed `raw_cosine` reaches 0.982 ± 0.006 while
  `loss_unlearned` sits at 0.548.

But **no seed reached its forget/retain loss-ratio target of 2.0**. The
directional signature is therefore also consistent with unlearning simply not
having finished: a model that has barely moved off θ_finetuned still carries the
fine-tuning geometry, and the forget set is still, in every measurable sense, in
the model.

## The question this sweep asks

> As unlearning strengthens and the membership evidence disappears, does the
> signed cosine still identify the deletion set while retained-data utility
> survives?

## Design, fixed in advance

| | |
|---|---|
| swept knob | `unlearning.ga_gd.ascent_lr` ∈ {1e-5, 2e-5, 5e-5, 1e-4, 2e-4} |
| held fixed | `steps: 100`, `descent_lr: 2.0e-5`, `batch_size: 4`, everything else in `configs/arxivmia_sweep.yaml` |
| seeds | 42, 43, 44 — each reusing **its own** `ds_checkpoints/finetuned_seed{seed}` and **its own** committed `results/seed{seed}/splits.json` |
| early stopping | disabled (`min_steps: 10000`), because it triggers on the very quantity being swept |
| collapse guard | `collapse_loss: 20.0`, NaN protection only; a trip is a data point recorded in `stop_reason`, not a failed run |
| replication unit | the seed |

1e-5 reproduces the baseline. The descent rate currently exceeding the ascent
rate is the likeliest cause of the weak forgetting, which is why the range runs
to 10× the descent rate.

## The decision rule

```
membership gone     loss_ratio AUC on forget_vs_unseen within [0.45, 0.55]

utility preserved   retain ppl(unlearned) <= 1.10 x retain ppl(finetuned)
                    AND test   ppl(unlearned) <= 1.10 x test   ppl(finetuned)

deletion present    raw_cosine AUC on forget_vs_retain >= 0.70
                    AND its bootstrap CI lower bound > 0.50
```

A setting **qualifies** only if all three hold **on all three seeds**. Two
seeds out of three is reported as two out of three and is not rounded up.

Every setting's per-criterion result is reported whether or not it qualifies.
**"Genuinely forgotten" is never concluded from a single metric.**

Notes on the specific choices:

- The membership criterion is a *band*, not a bound, and this is the reason: at
  baseline the oriented AUC is 0.001, so a rule of the form "AUC ≤ 0.55" would
  be satisfied by the very runs that have the strongest possible membership
  evidence. Under the declared direction 0.5 is chance in both senses, so "gone"
  means "at chance", and the band is two-sided.
- `forget_vs_unseen` is the task named `forget_vs_test` in the working notes;
  the population is called `test` and the task is called `forget_vs_unseen`
  throughout the code.
- Perplexity is token-weighted per population, `exp(Σ loss_i·n_i / Σ n_i)`. A
  per-sample average would weight a six-token sample like a five-hundred-token
  one and would not be the perplexity of any corpus.
- The bootstrap interval in the deletion criterion is over samples within one
  run. It guards against a threshold being cleared by noise in a single model;
  it is *not* the across-seed uncertainty, which is reported separately and is
  what any claim about the method rests on.

## The three outcomes, all reportable

The run is designed so that each of these is a result, and the wording is fixed
here so the verdict is mechanical:

1. **Fingerprint persists.** Some strength satisfies all three criteria on every
   seed. The directional signal survives after the membership evidence is gone,
   with utility intact.
2. **Incomplete forgetting.** Where membership evidence disappears and utility
   survives, the deletion signal does not clear its threshold. The baseline
   fingerprint was then consistent with unlearning not having finished, and the
   paper says so.
3. **Inconclusive.** No strength achieved forgetting and utility together. This
   is a statement about the range swept, not about the signal.

## Declared conflicts of interest in the analysis itself

**`raw_cosine` was promoted to the headline after being seen to win.** In the
baseline three seeds it reached 0.982 ± 0.006 on the primary task, and only then
was it preregistered as the primary score — the promotion is recorded with its
date in the change log at the top of `configs/orientations.yaml`.

The consequence is stated once and applies everywhere: this sweep confirms the
score on data that did not select it, and **the baseline three seeds cannot be
used as that confirmation.** Any report that cites both must say which role each
body of data is playing.

`loss_ratio_vs_finetuned` and `loss_delta_vs_finetuned` were added at the same
time, with directions declared for all five tasks before a single row of them
existed.

## Threat model, stated with the result

`raw_cosine` requires **both** the fine-tuned pre-unlearning checkpoint and the
unlearned checkpoint, because ĝ = θ_unlearned − θ_finetuned cannot be formed
without them. It is a **white-box deletion audit** available to a model owner, a
regulator or an authorised verifier — **not** a black-box membership-inference
attack. A party holding only the released model cannot mount it.

The claim is scoped to **gradient-ascent-based unlearning**. The mechanism
argued for the sign — forget samples receive ascent so their gradients point
along ĝ, retain samples receive descent so theirs oppose it — is specific to
GA+GD. SCRUB showed no directional advantage and is out of this sweep.

## Statistics

n = 3 seeds. Means, paired within-seed differences and across-seed SDs lead.
`stats.across_seed_comparison` already refuses to pretend three seeds are many:
below six pairs Wilcoxon cannot reach p < 0.05 at all, and the paired t-test it
falls back to rests on a normality assumption that three points cannot support.
The sweep report computes **no p-values**; none would carry weight at this n.
