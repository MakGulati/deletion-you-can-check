# The FUMA-style LoRA gradient ratio

## What is implemented

    fuma_lora_ratio(z) = magnitude(∇L(z) restricted to LoRA parameters)
                       / max(magnitude(∇L(z) restricted to non-LoRA parameters), eps)

with the group magnitude reduced, by default, as the **parameter-weighted mean
absolute gradient**:

    magnitude(group) = ( Σ_{p ∈ group} ‖∇_p L‖₁ ) / ( Σ_{p ∈ group} numel(p) )

Configurable alternatives (`scores.fuma.norm`):

| value | magnitude(group) |
|---|---|
| `l1_mean_abs` *(default)* | `Σ‖∇_p L‖₁ / Σ numel(p)` — mean absolute entry |
| `l2` | `sqrt(Σ‖∇_p L‖₂² / Σ numel(p))` — root mean square entry |
| `l2_sq` | `Σ‖∇_p L‖₂² / Σ numel(p)` — mean squared entry; the same convention `grad_norm_sq` uses |

And `scores.fuma.weighting`:

| value | meaning |
|---|---|
| `parameter` *(default)* | divide by the group's total element count, so a 2048×2048 projection outweighs a 16×2048 `lora_A` by its size |
| `tensor` | average each tensor's own per-entry mean, then average across tensors, so every tensor is one vote; only defined together with `l1_mean_abs` |

**LoRA membership** is decided by the substring `lora` in the lowercased
parameter name, which is how PEFT names adapter parameters
(`…q_proj.lora_A.default.weight`) and what `get_peft_model_state_dict` keys on.
Everything else — backbone projections, embeddings, layer norms, the output head
— counts as non-LoRA.

The convention actually used is recorded in every output file under
`metadata.fuma`, alongside the discrepancy list below.

## Discrepancy: the formula was not verified against the paper

The exact published FUMA formula could not be checked from this environment. The
implementation therefore does **not** claim to reproduce it. What it does is
state its own convention explicitly, make the three plausible readings
configurable, and record which one produced each row. If the published
definition turns out to be one of the alternatives, the switch is a config key
and a re-run of the collection stage — the scoring code does not change.

Open questions this leaves, all of which change the number:

- L1, L2, or squared L2 as the per-group reduction
- parameter-weighted or tensor-weighted averaging
- whether the denominator is meant to span the *whole* backbone or only the
  modules the adapters target

## Feasibility by training mode

### QLoRA (the primary setting)

The non-LoRA backbone is **frozen and 4-bit quantised**. bitsandbytes does not
backpropagate into 4-bit weights, so ‖∇L‖ over the backbone does not exist on
the model that was actually unlearned. The denominator is simply unobtainable
there.

The resolution is `scores.fuma.denominator_source: shadow_bf16`: load an
unquantised bf16 copy of the backbone, apply the same adapters, unfreeze
everything, and take **both** halves of the ratio from that one backward pass.
Taking both from the same model matters — a ratio whose numerator came from a
4-bit model and whose denominator came from a bf16 one would straddle a
quantisation boundary and mean nothing.

Recorded discrepancies from the published setting:

1. Both magnitudes come from an **unquantised copy** of the unlearned model, not
   from the 4-bit model that was unlearned. Dequantisation is not exact, so
   these are gradients of a nearby model rather than of the scored one.
2. The backbone is **unfrozen purely to obtain gradients**. It was frozen during
   training, so these gradients describe a direction the optimiser never took.
3. The non-LoRA group **includes the embedding and output projection**, which
   dominate the parameter-weighted denominator by element count.

If the shadow model does not fit in memory, `load_shadow_model` raises and the
collector records FUMA as not-applicable with that reason. It does not fall back
to an invented definition.

### Full fine-tuning

There are no LoRA parameters, so the numerator is undefined and FUMA is **not
applicable**. Rows carry `fuma_lora_ratio: null` together with
`fuma_na_reason`, never `0.0` — zero is a legitimate ratio and a reader could
not tell a placeholder from a measurement.

`scores.updated_parameter_ratio` exists as a defensible
updated-versus-non-updated analogue, deliberately under a **different name**.
In a plain full fine-tune every parameter is updated, so it too is not
applicable unless the caller supplies a partition through
`scoring.param_scope: named_subset`. Calling any of this "FUMA" would misreport
the comparison: the published method is defined over LoRA adapters.

### Unquantised backbone (the CPU smoke configuration)

Nothing is quantised, so `denominator_source: same_model` works and no shadow
copy is needed. The backbone is still frozen by PEFT, so the collector unfreezes
it — after the scored parameter set has already been fixed, leaving the other
scores untouched — and records discrepancy (2) above.

## Not-applicable is a first-class outcome

Every path that cannot produce the ratio produces `(None, reason)` instead:

| situation | reason recorded |
|---|---|
| no LoRA parameters in scope | "full fine-tuning or a non-LoRA checkpoint; FUMA's LoRA/non-LoRA split is undefined" |
| no non-LoRA gradients | "the backbone is frozen and 4-bit quantised…; enable the bf16 shadow model" |
| `denominator_source: none` | "the non-LoRA gradient magnitude was not obtained, so the ratio is not computed" |
| `scores.fuma.enabled: false` | "disabled in configuration" |
| shadow model would not load | "the bf16 shadow model required for the FUMA denominator was not loaded" |
| a magnitude came out non-finite | "numerator or denominator is not finite" |

The offline evaluator drops rows with a `None` value for a score, counts them in
`n_dropped`, and reports the score as unavailable when no row has one. It never
substitutes a value.
