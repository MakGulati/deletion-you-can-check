# NPO — the formulation implemented here, and where it departs

Implementation: [`deletion_signals/npo.py`](../npo.py).
Tests: [`deletion_signals/tests/test_npo.py`](../tests/test_npo.py).
Frozen parameters: [`configs/preregistration_v3_fllm.yaml`](../configs/preregistration_v3_fllm.yaml), section `component_a_npo`.

This document records exactly what was implemented, which published variant it
corresponds to, and every place where a choice had to be made that the paper
leaves open. It exists so a reader can check the implementation against the
source rather than against a summary of it.

## The equation

Zhang, Lin, Chen, Sun, Wu and Wu (2024), *Negative Preference Optimization: From
Catastrophic Collapse to Effective Unlearning*, arXiv:2404.05868, **Eq. 3** — the
DPO objective with the preferred-response term removed:

```
L_NPO(θ) = (2/β) · E_{(x,y)~D_forget} [ log( 1 + (π_θ(y|x) / π_ref(y|x))^β ) ]
         = -(2/β) · E [ log σ( -β · r(x,y) ) ]

r(x,y) = log π_θ(y|x) − log π_ref(y|x)
```

Both forms are implemented and are asserted equal to each other, and the first
is asserted against a value computed by hand outside the implementation.

## Sign convention

`r` is **current minus reference**, on the forget sequence. Minimising `L_NPO`
drives `π_θ` **down** on forget records, i.e. drives the forget **loss up**.

A run whose forget loss falls has the sign inverted. That failure mode is silent
— every curve still looks plausible — so it is checked three times: as a
behavioural unit test, as a monotonicity property of the gradient weight, and as
an assertion in the GPU smoke gate.

## The gradient, and the mechanism under test

```
∇_θ L_NPO = 2 · σ(β · r) · ∇_θ log π_θ(y|x)
```

Gradient ascent on the forget log-likelihood, weighted by a factor in `(0, 2)`.

- At initialisation `θ = θ_ref`, so `r = 0` and the weight is `2σ(0) = 1`.
- As forgetting proceeds `r` goes negative, `σ(βr) → 0`, and the weight vanishes.

That decay is NPO's entire contribution over gradient ascent, and whether it
prevents collapse *here* is Component A's question. The batch-mean weight is
recorded in the trajectory at every probe, so the mechanism is observable rather
than assumed.

### The consequence for the learning-rate grid

At step 0 NPO's forget gradient is the same direction as GA's ascent gradient,
and — because both are clipped to `max_grad_norm = 1.0` — the same **step**, not
merely a parallel one. `test_npo_first_step_matches_gradient_ascent_up_to_a_positive_scale`
asserts a cosine of 1.0 and equality after normalisation.

This is what licenses anchoring NPO's preregistered grid on GA+GD's measured
transition (RNFF crossed 0.5 at ascent_lr 2.6e-5; utility failed by 2.7e-5)
rather than on any NPO outcome. Because the weight decays monotonically, NPO at
a given learning rate is weaker per step than GA, so the grid is extended
**upward** to 1.0e-4 rather than downward.

## β

Temperature on the sequence log-ratio.

- `β → 0` freezes the weight at 1 and recovers plain gradient ascent.
- Larger `β` sharpens the gate, cutting a forgotten record off sooner.

**Fixed at the published default 0.1** for the whole sweep. Strength is swept
through the learning rate instead, which keeps one axis moving and makes the
sweep directly comparable with the GA+GD strength sweep.

## The reference model

`π_ref = θ_finetuned` — **each seed's own** pre-unlearning checkpoint at
`ds_checkpoints/finetuned_seed{N}`, which is also the initialisation. It is
deep-copied before any optimiser touches the model, put in `eval()`, and has
`requires_grad` cleared on every parameter.

Reusing one seed's checkpoint for another is prohibited by the preregistration:
it would make the seeds dependent and the across-seed SD meaningless.

Tested: the reference receives no gradient, and its log-probabilities are
unchanged after the policy has moved.

## The variant: NPO-RT

The paper defines three:

| variant | forget term | retain term |
|---|---|---|
| NPO | `L_NPO` | none |
| **NPO-RT** (implemented) | `L_NPO` | cross-entropy on `D_retain` |
| NPO-KL | `L_NPO` | KL to the reference on `D_retain` |

```
L = L_NPO(D_forget) + λ · CE(D_retain),   λ = 1.0
```

**Why NPO-RT.** It is the variant structurally comparable to this repository's
GA+GD, which is ascent on forget plus cross-entropy descent on retain. Selected
on comparability, not on outcome — no NPO number existed when the choice was
made.

## Departures, and why each was taken

### 1. Token reduction: sum, not mean

The paper derives `r` over sequences, and the official implementation
(`licong-lin/negative-preference-optimization`, following TOFU's
`get_batch_loss`) uses the **sum** of token log-probabilities. Every other loss
in this repository is a token **mean**.

The two differ by up to `max_length = 512`, which moves β's effective scale by
about two orders of magnitude — so this is not a detail. `sum` is used, matching
the paper; `mean` is available as `unlearning.npo.log_ratio_reduction` and the
setting used is written into every output row's metadata.

It does not affect the grid: both GA+GD and NPO clip to norm 1.0, which
normalises the scale difference away.

### 2. One optimizer, not two

GA+GD uses two AdamW optimizers with independently swept ascent and descent
learning rates. NPO-RT as published is a single objective under a single
optimizer, so **sweeping its learning rate scales the retain term too**. The
published form is kept and the difference recorded rather than engineered away.

### 3. AdamW partially normalises away the mechanism — an open caveat

NPO's collapse-avoidance is a gradient **magnitude** gate, `2σ(βr)`. Adam's
second-moment normalisation rescales per-parameter update magnitudes, so it
undoes part of that gate's effect on the step size.

The official implementation also uses AdamW, so this is faithful to the
published method — but it means **an NPO collapse observed here would not refute
the paper's mechanism, only its behaviour under this optimiser.** This is stated
in the preregistration's risk list and must be stated in the paper. It is not a
bug to be fixed by switching optimisers, which would make the run something
other than NPO.

### 4. Forget and retain backwarded separately

Gradients accumulate into `.grad` either way, so the update is identical, but
summing the losses first keeps both autograd graphs alive at once — the same
24 GB constraint that shapes SCRUB's loop.

## Numerical notes

- `F.logsigmoid`, never `log(sigmoid(·))`. The latter underflows to `-inf` once
  `β·r` drops below about −100, which a sequence-summed log-ratio reaches
  routinely. Tested at `|r|` up to 5000.
- `F.cross_entropy(reduction="none")` rather than an explicit `log_softmax`:
  fused, handles `ignore_index` natively, and does not materialise a
  `[B, L, V]` float32 tensor (≈1 GB at B=4, L=511, V=128k).
- Padding is masked out of the log-probability sum. Without it a sequence's `r`
  would depend on how much padding its batch-mates needed.

## What the smoke test may and may not do

Permitted: check numerical stability, and check that the parameters have their
intended qualitative effect (forget loss rises; the weight starts at 1 and
falls).

**Not permitted**: choosing the grid, selecting β, or moving any threshold.
Those are frozen in `preregistration_v3_fllm.yaml` before any NPO run existed.
