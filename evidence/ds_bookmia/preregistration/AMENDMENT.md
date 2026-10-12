# Amendment 1 — 2026-08-09 22:00, before any BookMIA training run completed

## Micro-batch reduced, effective batch unchanged

The first launch OOMed on the fine-tune. Qwen2.5's vocabulary is 151936 tokens
against Llama-3.2's 128256, so the logits tensor at batch 8 x 512 is ~2.5 GB in
float32 and the backward exceeded 24 GB.

`training.batch_size` 8 -> 4 and `gradient_accumulation_steps` 4 -> 8, and the
same in `routine_update`. **Effective batch size stays 32**, identical to
ArxivMIA, so the optimiser sees the same number of examples per step and the
same number of steps. Only the memory layout changed; this is not a protocol
change and no criterion, method setting or threshold moved.

Cause of the miss: the structural pre-check ran at batch size 1, which does not
exercise the memory the training loop actually needs. Memory was subsequently
probed at the real batch sizes for every training path:

| path | batch | peak |
|---|---:|---:|
| fine-tune fwd+bwd | 4 | 10.13 GiB |
| GA+GD ascent fwd+bwd | 4 | 10.13 GiB |
| NPO fwd+bwd + frozen reference | 4 | 11.24 GiB |
| SCRUB fwd+bwd + frozen teacher | 4 | 20.49 GiB |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the runner.

## SCRUB is run at its transferred batch size despite thin headroom

SCRUB peaks at 20.49 GiB of 23.55 available. Reducing its batch size would make
the transfer from ArxivMIA inexact -- batch size affects per-step gradient noise
and is part of the method setting -- so it is left at 4. **If SCRUB OOMs it will
be reported as "did not fit on this hardware at the transferred setting", not
silently retuned to fit.** The runner records a stage failure and continues, so
one method failing does not cost the rest of the matrix.
