# Preregistration — BookMIA + Qwen2.5-1.5B replication

**Frozen 2026-08-09, after the base-calibration pre-flight passed and before any
BookMIA training run existed.**

Machine-readable configuration: [`configs/bookmia_qwen.yaml`](../../deletion_signals/configs/bookmia_qwen.yaml).
Criteria: inherited **unchanged** from
[`configs/preregistration_v3_fllm.yaml`](../../deletion_signals/configs/preregistration_v3_fllm.yaml).

## Purpose, and what it can and cannot establish

The ArxivMIA package's largest weakness is external validity: one corpus, one
model family, one tokenizer. This replication tests whether the **pattern of
findings** transfers — not whether the numbers do.

It **can** establish that the pattern holds on a second dataset and a second
model family. It **cannot** fix n=3, cannot make "no method passed" a general
claim about approximate unlearning, and cannot license any statement about
datasets or architectures not tested.

## No threshold moves

Every criterion is inherited by value from v3: membership equivalence region
[0.45, 0.55]; utility max perplexity ratio 1.10; RNFF ≥ 0.50 forget and ≤ 0.20
retain; deletion detectable at AUC ≥ 0.70 with CI lower > 0.50 above the
null-refit floor; primary detector `loss_ratio_vs_finetuned`; seed as the
replication unit; the same six-label classification partition.

A replication that adjusted its thresholds would not be one.

## Pre-flight result (the gate this run had to clear)

The membership axis divides by `L_base`, which assumes the base model never saw
the data. BookMIA's `label=1` marks books from pretraining corpora, so that
assumption needed testing before eight hours were spent on it.

| | AUC(member vs unseen) under `-L_base` | deviation from chance |
|---|---|---|
| ArxivMIA + Llama-3.2-1B (reference) | 0.4626 ± 0.0221 | 0.0374 |
| **BookMIA + Qwen2.5-1.5B** | **0.4597 ± 0.0287** | **0.0403** |

**PASS** against the criterion fixed in
[BASE_CALIBRATION_CRITERION.md](BASE_CALIBRATION_CRITERION.md) (`d ≤ 0.10`).
The base model marginally *dis*favours members on both pairs. The calibration is
clean and the membership axis is comparable across datasets.

## The matrix

Three seeds (42, 43, 44), each with its own fine-tuned checkpoint and its own
split. Populations match ArxivMIA: 200 forget, 800 retain, 300 scored test.

| condition | setting | rationale |
|---|---|---|
| null refit | full member set, RNG offset 1000 | false-positive floor |
| exact retraining | retain only | gold standard; supplies the RNFF denominator |
| GA+GD weak | `ascent_lr = 2.6e-5` | **transferred**, the ArxivMIA best-viable setting |
| GA+GD aggressive | `ascent_lr = 5.0e-5` | **transferred**, the ArxivMIA destructive setting |
| NPO | `lr = 1e-4, β = 0.1, sum` | **transferred**, the ArxivMIA best-forgetting setting |
| SCRUB | `forget_kl = 4, lr = 5e-5` | **transferred**, the ArxivMIA best-scoring setting |
| routine update | no deletion + maintenance | nuisance floor |
| exact deletion + routine update | deletion + identical maintenance | recoverability under maintenance |

**Every method setting is transferred, not tuned.** No search is run on BookMIA,
so there is no selection freedom to declare. If a transferred setting behaves
differently here, that is the finding.

## Two deliberate differences from the ArxivMIA configuration

**1. FUMA and Fisher-SIF are off.** Both are already declared negative results —
SIF tracked `grad_norm_sq` at ρ = 0.9999 and FUMA was the weakest detector in
every ArxivMIA table. Independently, FUMA's bf16 shadow model peaked at 11.5 GB
on Llama-1B at 205 tokens; at 1.25× the parameters and 2.5× the sequence length
it would not fit in 24 GB. Eleven detectors remain, including every one in the
central matched-access comparison.

Consequence: no Fisher curvature holdout, so retain is scored whole (800 rather
than 600). Negative counts therefore differ from ArxivMIA's, which changes the
FPR resolution — 1/800 = 0.00125 rather than 1/600 = 0.00167. TPR@0.1% FPR
remains unsupported on both.

**2. The routine update draws fresh MEMBER-pool records** (`source:
member_tail`). BookMIA supplies 4935 members against a draw of 1000, so records
the split never touched are available and are from the **same population as the
training data**.

This removes the confound ArxivMIA forced, where the only disjoint records were
non-members and the update therefore also shifted the model toward the
non-member distribution. **The BookMIA routine-update control is the clean
version**; the ArxivMIA one is reported with its confound declared, and the two
are compared with that difference stated.

Disjointness is enforced by text fingerprint at run time against forget, retain
and the scored test prefix, and the run refuses to train if it does not hold.

## Declared limitation: universal truncation

**Every BookMIA snippet exceeds `max_length`.** Mean length is 710 tokens,
median 664, and **100% of scored samples are truncated at 512** (measured:
1300/1300 rows, mean 511 scored tokens).

This is reported as a limitation of the replication, not hidden:

- it is a property of the corpus and the length budget, not of any method;
- it applies **identically** to every method, control and detector, so all
  within-dataset comparisons remain matched and the detector ranking is
  unaffected by it;
- it does mean BookMIA results describe the **first 512 tokens** of each
  snippet, and a method whose behaviour depends on later content would not be
  measured here;
- ArxivMIA, by contrast, truncates almost nothing (mean 205 tokens), so
  cross-dataset differences confound corpus, model family **and** truncation
  regime. No cross-dataset difference is attributed to any one of the three.

## What a successful replication would look like

Stated in advance so it cannot be decided afterwards. The pattern replicates if:

1. exact retraining passes the conjunction on every seed, and null refit
   produces RNFF ≈ 0 with every detector at chance;
2. no approximate method passes the conjunction on any seed;
3. aggressive GA+GD reaches the membership region while failing utility — the
   false-certification result;
4. under matched access, `loss_ratio_vs_finetuned` is not beaten by
   `raw_cosine`, `alignment_cos_sq` or `grad_ratio` by more than one across-seed
   SD;
5. precision at 1:100 falls substantially below ROC-AUC for every detector.

A failure of any of these on BookMIA is reported as a **limit on the
generalisation**, not explained away.

## Output

`ds_bookmia/`, mirroring `fllm_audit/`. Nothing writes to
`fllm_audit/`, `ds_confirm*/`, `ds_sweep/` or `ds_results/`.
