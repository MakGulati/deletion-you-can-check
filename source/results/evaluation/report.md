# Deletion-set inference: head-to-head evaluation

Generated 2026-08-08T07:44:43.860893+00:00
Inputs: 6 score file(s)
Primary task: **forget_vs_retain**

All AUCs use the preregistered orientation for each (score, task) pair. `max(AUC, 1-AUC)` is not used. A score below 0.50 is reported below 0.50.

## Per run, task `forget_vs_retain`

| run                | score                | dir  |   AUC    |       95% CI       |    AP    |  TPR@1%  |
|--------------------|----------------------|------|----------|--------------------|----------|----------|
| ga_gd/seed42       | alignment_cos_sq     | up   |  0.8525  | [0.8255, 0.8800] |  0.6323  |  0.1300  |
| ga_gd/seed42       | combined_log_score   | up   |  0.8667  | [0.8403, 0.8918] |  0.6800  |  0.2250  |
| ga_gd/seed42       | fuma_lora_ratio      | up   |  0.5146  | [0.4661, 0.5607] |  0.2636  |  0.0050  |
| ga_gd/seed42       | grad_norm_sq         | up   |  0.5195  | [0.4731, 0.5631] |  0.2773  |  0.0150  |
| ga_gd/seed42       | grad_ratio           | up   |  0.8511  | [0.8208, 0.8780] |  0.6720  |  0.2550  |
| ga_gd/seed42       | log_grad_ratio       | up   |  0.8511  | [0.8208, 0.8780] |  0.6720  |  0.2550  |
| ga_gd/seed42       | loss_delta           | up   |  0.6836  | [0.6388, 0.7255] |  0.4450  |  0.0650  |
| ga_gd/seed42       | loss_ratio           | up   |  0.6820  | [0.6389, 0.7242] |  0.4252  |  0.0600  |
| ga_gd/seed42       | loss_unlearned       | up   |  0.5554  | [0.5094, 0.6015] |  0.2974  |  0.0200  |
| ga_gd/seed42       | raw_cosine           | up   |  0.9876  | [0.9814, 0.9925] |  0.9681  |  0.8100  |
| ga_gd/seed42       | sif                  | up   |  0.5192  | [0.4727, 0.5628] |  0.2768  |  0.0150  |
| ga_gd/seed43       | alignment_cos_sq     | up   |  0.8566  | [0.8253, 0.8824] |  0.6885  |  0.2500  |
| ga_gd/seed43       | combined_log_score   | up   |  0.8709  | [0.8404, 0.8961] |  0.6842  |  0.1400  |
| ga_gd/seed43       | fuma_lora_ratio      | up   |  0.5135  | [0.4644, 0.5613] |  0.2681  |  0.0100  |
| ga_gd/seed43       | grad_norm_sq         | up   |  0.5687  | [0.5203, 0.6098] |  0.2997  |  0.0200  |
| ga_gd/seed43       | grad_ratio           | up   |  0.8569  | [0.8294, 0.8825] |  0.6304  |  0.1150  |
| ga_gd/seed43       | log_grad_ratio       | up   |  0.8569  | [0.8294, 0.8825] |  0.6304  |  0.1150  |
| ga_gd/seed43       | loss_delta           | up   |  0.7068  | [0.6676, 0.7485] |  0.4697  |  0.0850  |
| ga_gd/seed43       | loss_ratio           | up   |  0.6884  | [0.6475, 0.7269] |  0.4585  |  0.0700  |
| ga_gd/seed43       | loss_unlearned       | up   |  0.5654  | [0.5209, 0.6084] |  0.2967  |  0.0100  |
| ga_gd/seed43       | raw_cosine           | up   |  0.9820  | [0.9725, 0.9892] |  0.9625  |  0.8000  |
| ga_gd/seed43       | sif                  | up   |  0.5684  | [0.5198, 0.6094] |  0.3000  |  0.0200  |
| ga_gd/seed44       | alignment_cos_sq     | up   |  0.8153  | [0.7812, 0.8491] |  0.5836  |  0.1050  |
| ga_gd/seed44       | combined_log_score   | up   |  0.8345  | [0.8007, 0.8671] |  0.6226  |  0.1500  |
| ga_gd/seed44       | fuma_lora_ratio      | up   |  0.5548  | [0.5056, 0.5994] |  0.3001  |  0.0100  |
| ga_gd/seed44       | grad_norm_sq         | up   |  0.5372  | [0.4911, 0.5814] |  0.2834  |  0.0200  |
| ga_gd/seed44       | grad_ratio           | up   |  0.8452  | [0.8115, 0.8736] |  0.6430  |  0.1200  |
| ga_gd/seed44       | log_grad_ratio       | up   |  0.8452  | [0.8115, 0.8736] |  0.6430  |  0.1200  |
| ga_gd/seed44       | loss_delta           | up   |  0.6953  | [0.6552, 0.7366] |  0.4327  |  0.0500  |
| ga_gd/seed44       | loss_ratio           | up   |  0.6562  | [0.6123, 0.7001] |  0.3945  |  0.0600  |
| ga_gd/seed44       | loss_unlearned       | up   |  0.5228  | [0.4724, 0.5677] |  0.2620  |  0.0100  |
| ga_gd/seed44       | raw_cosine           | up   |  0.9761  | [0.9658, 0.9849] |  0.9475  |  0.7550  |
| ga_gd/seed44       | sif                  | up   |  0.5357  | [0.4902, 0.5798] |  0.2830  |  0.0200  |
| scrub/seed42       | alignment_cos_sq     | up   |  0.5872  | [0.5417, 0.6327] |  0.3186  |  0.0150  |
| scrub/seed42       | combined_log_score   | up   |  0.6201  | [0.5740, 0.6629] |  0.3410  |  0.0050  |
| scrub/seed42       | fuma_lora_ratio      | up   |  0.5273  | [0.4774, 0.5700] |  0.2674  |  0.0050  |
| scrub/seed42       | grad_norm_sq         | up   |  0.4944  | [0.4459, 0.5409] |  0.2514  |  0.0050  |
| scrub/seed42       | grad_ratio           | up   |  0.5897  | [0.5416, 0.6358] |  0.3343  |  0.0250  |
| scrub/seed42       | log_grad_ratio       | up   |  0.5897  | [0.5416, 0.6358] |  0.3343  |  0.0250  |
| scrub/seed42       | loss_delta           | up   |  0.5599  | [0.5128, 0.6045] |  0.2874  |  0.0000  |
| scrub/seed42       | loss_ratio           | up   |  0.5622  | [0.5157, 0.6093] |  0.2843  |  0.0000  |
| scrub/seed42       | loss_unlearned       | up   |  0.5216  | [0.4756, 0.5682] |  0.2744  |  0.0150  |
| scrub/seed42       | raw_cosine           | up   |  0.5872  | [0.5417, 0.6327] |  0.3186  |  0.0150  |
| scrub/seed42       | sif                  | up   |  0.4908  | [0.4447, 0.5369] |  0.2493  |  0.0050  |
| scrub/seed43       | alignment_cos_sq     | up   |  0.5717  | [0.5238, 0.6176] |  0.3145  |  0.0150  |
| scrub/seed43       | combined_log_score   | up   |  0.6280  | [0.5792, 0.6724] |  0.3930  |  0.0800  |
| scrub/seed43       | fuma_lora_ratio      | up   |  0.5026  | [0.4544, 0.5493] |  0.2593  |  0.0300  |
| scrub/seed43       | grad_norm_sq         | up   |  0.5298  | [0.4807, 0.5745] |  0.2812  |  0.0200  |
| scrub/seed43       | grad_ratio           | up   |  0.5723  | [0.5309, 0.6168] |  0.3197  |  0.0400  |
| scrub/seed43       | log_grad_ratio       | up   |  0.5723  | [0.5309, 0.6168] |  0.3197  |  0.0400  |
| scrub/seed43       | loss_delta           | up   |  0.5753  | [0.5295, 0.6234] |  0.3274  |  0.0400  |
| scrub/seed43       | loss_ratio           | up   |  0.5763  | [0.5322, 0.6225] |  0.3279  |  0.0400  |
| scrub/seed43       | loss_unlearned       | up   |  0.5298  | [0.4851, 0.5731] |  0.2667  |  0.0100  |
| scrub/seed43       | raw_cosine           | up   |  0.5717  | [0.5238, 0.6176] |  0.3145  |  0.0150  |
| scrub/seed43       | sif                  | up   |  0.5275  | [0.4790, 0.5701] |  0.2786  |  0.0200  |
| scrub/seed44       | alignment_cos_sq     | up   |  0.6000  | [0.5516, 0.6456] |  0.3487  |  0.0100  |
| scrub/seed44       | combined_log_score   | up   |  0.5958  | [0.5527, 0.6422] |  0.3560  |  0.0550  |
| scrub/seed44       | fuma_lora_ratio      | up   |  0.5477  | [0.5010, 0.5939] |  0.2958  |  0.0000  |
| scrub/seed44       | grad_norm_sq         | up   |  0.5145  | [0.4687, 0.5581] |  0.2620  |  0.0000  |
| scrub/seed44       | grad_ratio           | up   |  0.5870  | [0.5430, 0.6346] |  0.3326  |  0.0200  |
| scrub/seed44       | log_grad_ratio       | up   |  0.5870  | [0.5430, 0.6346] |  0.3326  |  0.0200  |
| scrub/seed44       | loss_delta           | up   |  0.5594  | [0.5142, 0.6037] |  0.2937  |  0.0150  |
| scrub/seed44       | loss_ratio           | up   |  0.5409  | [0.4936, 0.5875] |  0.2845  |  0.0200  |
| scrub/seed44       | loss_unlearned       | up   |  0.4901  | [0.4420, 0.5356] |  0.2411  |  0.0000  |
| scrub/seed44       | raw_cosine           | up   |  0.6000  | [0.5516, 0.6456] |  0.3487  |  0.0100  |
| scrub/seed44       | sif                  | up   |  0.5061  | [0.4607, 0.5493] |  0.2587  |  0.0000  |

## Across seeds (mean +/- SD of the per-seed AUC)

### ga_gd

| task               | score                |   mean   |    SD    | n seeds  |
|--------------------|----------------------|----------|----------|----------|
| forget_vs_rest     | alignment_cos_sq     |  0.7951  |  0.0581  |    3     |
| forget_vs_rest     | combined_log_score   |  0.7154  |  0.0636  |    3     |
| forget_vs_rest     | fuma_lora_ratio      |  0.4171  |  0.0093  |    3     |
| forget_vs_rest     | grad_norm_sq         |  0.3763  |  0.0111  |    3     |
| forget_vs_rest     | grad_ratio           |  0.6605  |  0.0331  |    3     |
| forget_vs_rest     | log_grad_ratio       |  0.6605  |  0.0331  |    3     |
| forget_vs_rest     | loss_delta           |  0.2609  |  0.0044  |    3     |
| forget_vs_rest     | loss_ratio           |  0.2537  |  0.0064  |    3     |
| forget_vs_rest     | loss_unlearned       |  0.3576  |  0.0148  |    3     |
| forget_vs_rest     | raw_cosine           |  0.8479  |  0.0514  |    3     |
| forget_vs_rest     | sif                  |  0.3777  |  0.0110  |    3     |
| forget_vs_retain   | alignment_cos_sq     |  0.8414  |  0.0227  |    3     |
| forget_vs_retain   | combined_log_score   |  0.8574  |  0.0199  |    3     |
| forget_vs_retain   | fuma_lora_ratio      |  0.5276  |  0.0235  |    3     |
| forget_vs_retain   | grad_norm_sq         |  0.5418  |  0.0249  |    3     |
| forget_vs_retain   | grad_ratio           |  0.8511  |  0.0059  |    3     |
| forget_vs_retain   | log_grad_ratio       |  0.8511  |  0.0059  |    3     |
| forget_vs_retain   | loss_delta           |  0.6952  |  0.0116  |    3     |
| forget_vs_retain   | loss_ratio           |  0.6755  |  0.0170  |    3     |
| forget_vs_retain   | loss_unlearned       |  0.5479  |  0.0223  |    3     |
| forget_vs_retain   | raw_cosine           |  0.9819  |  0.0058  |    3     |
| forget_vs_retain   | sif                  |  0.5411  |  0.0250  |    3     |
| forget_vs_unseen   | alignment_cos_sq     |  0.7673  |  0.0795  |    3     |
| forget_vs_unseen   | combined_log_score   |  0.6302  |  0.0901  |    3     |
| forget_vs_unseen   | fuma_lora_ratio      |  0.3508  |  0.0043  |    3     |
| forget_vs_unseen   | grad_norm_sq         |  0.2770  |  0.0051  |    3     |
| forget_vs_unseen   | grad_ratio           |  0.5461  |  0.0496  |    3     |
| forget_vs_unseen   | log_grad_ratio       |  0.5461  |  0.0496  |    3     |
| forget_vs_unseen   | loss_delta           |  0.0003  |  0.0001  |    3     |
| forget_vs_unseen   | loss_ratio           |  0.0006  |  0.0002  |    3     |
| forget_vs_unseen   | loss_unlearned       |  0.2435  |  0.0112  |    3     |
| forget_vs_unseen   | raw_cosine           |  0.7675  |  0.0796  |    3     |
| forget_vs_unseen   | sif                  |  0.2797  |  0.0045  |    3     |
| member_vs_unseen   | alignment_cos_sq     |  0.4433  |  0.0491  |    3     |
| member_vs_unseen   | combined_log_score   |  0.3340  |  0.0494  |    3     |
| member_vs_unseen   | fuma_lora_ratio      |  0.3318  |  0.0150  |    3     |
| member_vs_unseen   | grad_norm_sq         |  0.7501  |  0.0081  |    3     |
| member_vs_unseen   | grad_ratio           |  0.7240  |  0.0295  |    3     |
| member_vs_unseen   | log_grad_ratio       |  0.7240  |  0.0295  |    3     |
| member_vs_unseen   | loss_delta           |  0.9999  |  0.0000  |    3     |
| member_vs_unseen   | loss_ratio           |  0.9998  |  0.0000  |    3     |
| member_vs_unseen   | loss_unlearned       |  0.7813  |  0.0046  |    3     |
| member_vs_unseen   | raw_cosine           |  0.2343  |  0.0282  |    3     |
| member_vs_unseen   | sif                  |  0.7474  |  0.0078  |    3     |
| retain_vs_unseen   | alignment_cos_sq     |  0.3353  |  0.0394  |    3     |
| retain_vs_unseen   | combined_log_score   |  0.2352  |  0.0359  |    3     |
| retain_vs_unseen   | fuma_lora_ratio      |  0.3255  |  0.0196  |    3     |
| retain_vs_unseen   | grad_norm_sq         |  0.7592  |  0.0104  |    3     |
| retain_vs_unseen   | grad_ratio           |  0.8141  |  0.0229  |    3     |
| retain_vs_unseen   | log_grad_ratio       |  0.8141  |  0.0229  |    3     |
| retain_vs_unseen   | loss_delta           |  1.0000  |  0.0000  |    3     |
| retain_vs_unseen   | loss_ratio           |  0.9999  |  0.0000  |    3     |
| retain_vs_unseen   | loss_unlearned       |  0.7896  |  0.0080  |    3     |
| retain_vs_unseen   | raw_cosine           |  0.0566  |  0.0122  |    3     |
| retain_vs_unseen   | sif                  |  0.7564  |  0.0102  |    3     |

### scrub

| task               | score                |   mean   |    SD    | n seeds  |
|--------------------|----------------------|----------|----------|----------|
| forget_vs_rest     | alignment_cos_sq     |  0.7339  |  0.0219  |    3     |
| forget_vs_rest     | combined_log_score   |  0.7112  |  0.0150  |    3     |
| forget_vs_rest     | fuma_lora_ratio      |  0.5098  |  0.0179  |    3     |
| forget_vs_rest     | grad_norm_sq         |  0.4293  |  0.0149  |    3     |
| forget_vs_rest     | grad_ratio           |  0.7743  |  0.0112  |    3     |
| forget_vs_rest     | log_grad_ratio       |  0.7743  |  0.0112  |    3     |
| forget_vs_rest     | loss_delta           |  0.2125  |  0.0038  |    3     |
| forget_vs_rest     | loss_ratio           |  0.2104  |  0.0068  |    3     |
| forget_vs_rest     | loss_unlearned       |  0.3626  |  0.0146  |    3     |
| forget_vs_rest     | raw_cosine           |  0.7339  |  0.0219  |    3     |
| forget_vs_rest     | sif                  |  0.3995  |  0.0108  |    3     |
| forget_vs_retain   | alignment_cos_sq     |  0.5863  |  0.0141  |    3     |
| forget_vs_retain   | combined_log_score   |  0.6146  |  0.0168  |    3     |
| forget_vs_retain   | fuma_lora_ratio      |  0.5259  |  0.0226  |    3     |
| forget_vs_retain   | grad_norm_sq         |  0.5129  |  0.0178  |    3     |
| forget_vs_retain   | grad_ratio           |  0.5830  |  0.0094  |    3     |
| forget_vs_retain   | log_grad_ratio       |  0.5830  |  0.0094  |    3     |
| forget_vs_retain   | loss_delta           |  0.5648  |  0.0091  |    3     |
| forget_vs_retain   | loss_ratio           |  0.5598  |  0.0178  |    3     |
| forget_vs_retain   | loss_unlearned       |  0.5139  |  0.0209  |    3     |
| forget_vs_retain   | raw_cosine           |  0.5863  |  0.0141  |    3     |
| forget_vs_retain   | sif                  |  0.5081  |  0.0184  |    3     |
| forget_vs_unseen   | alignment_cos_sq     |  0.8225  |  0.0313  |    3     |
| forget_vs_unseen   | combined_log_score   |  0.7692  |  0.0177  |    3     |
| forget_vs_unseen   | fuma_lora_ratio      |  0.5002  |  0.0151  |    3     |
| forget_vs_unseen   | grad_norm_sq         |  0.3791  |  0.0187  |    3     |
| forget_vs_unseen   | grad_ratio           |  0.8892  |  0.0189  |    3     |
| forget_vs_unseen   | log_grad_ratio       |  0.8892  |  0.0189  |    3     |
| forget_vs_unseen   | loss_delta           |  0.0010  |  0.0007  |    3     |
| forget_vs_unseen   | loss_ratio           |  0.0008  |  0.0004  |    3     |
| forget_vs_unseen   | loss_unlearned       |  0.2718  |  0.0117  |    3     |
| forget_vs_unseen   | raw_cosine           |  0.8225  |  0.0313  |    3     |
| forget_vs_unseen   | sif                  |  0.3343  |  0.0134  |    3     |
| member_vs_unseen   | alignment_cos_sq     |  0.7808  |  0.0407  |    3     |
| member_vs_unseen   | combined_log_score   |  0.7034  |  0.0209  |    3     |
| member_vs_unseen   | fuma_lora_ratio      |  0.4805  |  0.0028  |    3     |
| member_vs_unseen   | grad_norm_sq         |  0.6310  |  0.0179  |    3     |
| member_vs_unseen   | grad_ratio           |  0.1407  |  0.0189  |    3     |
| member_vs_unseen   | log_grad_ratio       |  0.1407  |  0.0189  |    3     |
| member_vs_unseen   | loss_delta           |  0.9992  |  0.0002  |    3     |
| member_vs_unseen   | loss_ratio           |  0.9993  |  0.0002  |    3     |
| member_vs_unseen   | loss_unlearned       |  0.7344  |  0.0042  |    3     |
| member_vs_unseen   | raw_cosine           |  0.7808  |  0.0407  |    3     |
| member_vs_unseen   | sif                  |  0.6724  |  0.0142  |    3     |
| retain_vs_unseen   | alignment_cos_sq     |  0.7668  |  0.0439  |    3     |
| retain_vs_unseen   | combined_log_score   |  0.6815  |  0.0228  |    3     |
| retain_vs_unseen   | fuma_lora_ratio      |  0.4739  |  0.0081  |    3     |
| retain_vs_unseen   | grad_norm_sq         |  0.6344  |  0.0188  |    3     |
| retain_vs_unseen   | grad_ratio           |  0.1506  |  0.0189  |    3     |
| retain_vs_unseen   | log_grad_ratio       |  0.1506  |  0.0189  |    3     |
| retain_vs_unseen   | loss_delta           |  0.9993  |  0.0002  |    3     |
| retain_vs_unseen   | loss_ratio           |  0.9994  |  0.0003  |    3     |
| retain_vs_unseen   | loss_unlearned       |  0.7365  |  0.0071  |    3     |
| retain_vs_unseen   | raw_cosine           |  0.7668  |  0.0439  |    3     |
| retain_vs_unseen   | sif                  |  0.6746  |  0.0155  |    3     |

## Paired across-seed comparisons (unit = seed)

Positive `mean diff` means the reference score has the higher AUC.

### ga_gd, task `forget_vs_retain`

| reference            | vs                   | mean diff  |    SD    |  n   | test             |    p     |
|----------------------|----------------------|------------|----------|------|------------------|----------|
| alignment_cos_sq     | grad_norm_sq         |   0.2996   |  0.0292  |  3   | paired t-test    |  0.0032  |
| alignment_cos_sq     | grad_ratio           |  -0.0096   |  0.0175  |  3   | paired t-test    |  0.4424  |
| alignment_cos_sq     | log_grad_ratio       |  -0.0096   |  0.0175  |  3   | paired t-test    |  0.4424  |
| alignment_cos_sq     | fuma_lora_ratio      |   0.3138   |  0.0462  |  3   | paired t-test    |  0.0071  |
| alignment_cos_sq     | loss_ratio           |   0.1659   |  0.0060  |  3   | paired t-test    |  0.0004  |
| alignment_cos_sq     | loss_delta           |   0.1462   |  0.0246  |  3   | paired t-test    |  0.0093  |
| alignment_cos_sq     | loss_unlearned       |   0.2936   |  0.0031  |  3   | paired t-test    |  0.0000  |
| alignment_cos_sq     | sif                  |   0.3003   |  0.0288  |  3   | paired t-test    |  0.0031  |
| alignment_cos_sq     | combined_log_score   |  -0.0159   |  0.0028  |  3   | paired t-test    |  0.0104  |

> n=3 seeds. Wilcoxon signed-rank cannot reach p<0.05 below 6 pairs (min two-sided p at n=3 is 0.25), so a paired t-test is used instead; its normality assumption is unverifiable at this n. Read the mean difference and its across-seed SD as the primary evidence.

### scrub, task `forget_vs_retain`

| reference            | vs                   | mean diff  |    SD    |  n   | test             |    p     |
|----------------------|----------------------|------------|----------|------|------------------|----------|
| alignment_cos_sq     | grad_norm_sq         |   0.0734   |  0.0275  |  3   | paired t-test    |  0.0437  |
| alignment_cos_sq     | grad_ratio           |   0.0033   |  0.0084  |  3   | paired t-test    |  0.5619  |
| alignment_cos_sq     | log_grad_ratio       |   0.0033   |  0.0084  |  3   | paired t-test    |  0.5619  |
| alignment_cos_sq     | fuma_lora_ratio      |   0.0604   |  0.0085  |  3   | paired t-test    |  0.0065  |
| alignment_cos_sq     | loss_ratio           |   0.0265   |  0.0319  |  3   | paired t-test    |  0.2860  |
| alignment_cos_sq     | loss_delta           |   0.0215   |  0.0227  |  3   | paired t-test    |  0.2423  |
| alignment_cos_sq     | loss_unlearned       |   0.0724   |  0.0345  |  3   | paired t-test    |  0.0679  |
| alignment_cos_sq     | sif                  |   0.0782   |  0.0294  |  3   | paired t-test    |  0.0440  |
| alignment_cos_sq     | combined_log_score   |  -0.0283   |  0.0305  |  3   | paired t-test    |  0.2489  |

> n=3 seeds. Wilcoxon signed-rank cannot reach p<0.05 below 6 pairs (min two-sided p at n=3 is 0.25), so a paired t-test is used instead; its normality assumption is unverifiable at this n. Read the mean difference and its across-seed SD as the primary evidence.

## Diagnostic Spearman correlations

| run                | pair                                 |   rho    |   n    |
|--------------------|--------------------------------------|----------|--------|
| ga_gd/seed42       | alignment_cos_sq~grad_norm_sq        | -0.1341  |  1800  |
| ga_gd/seed42       | fuma_lora_ratio~alignment_cos_sq     |  0.1520  |  1800  |
| ga_gd/seed42       | grad_ratio~grad_norm_sq              |  0.3774  |  1800  |
| ga_gd/seed42       | loss_ratio~loss_unlearned            |  0.5103  |  1800  |
| ga_gd/seed42       | sif~grad_norm_sq                     |  0.9999  |  1800  |
| ga_gd/seed43       | alignment_cos_sq~grad_norm_sq        | -0.0543  |  1800  |
| ga_gd/seed43       | fuma_lora_ratio~alignment_cos_sq     |  0.0917  |  1800  |
| ga_gd/seed43       | grad_ratio~grad_norm_sq              |  0.4467  |  1800  |
| ga_gd/seed43       | loss_ratio~loss_unlearned            |  0.5153  |  1800  |
| ga_gd/seed43       | sif~grad_norm_sq                     |  0.9998  |  1800  |
| ga_gd/seed44       | alignment_cos_sq~grad_norm_sq        | -0.1079  |  1800  |
| ga_gd/seed44       | fuma_lora_ratio~alignment_cos_sq     |  0.2294  |  1800  |
| ga_gd/seed44       | grad_ratio~grad_norm_sq              |  0.3802  |  1800  |
| ga_gd/seed44       | loss_ratio~loss_unlearned            |  0.5223  |  1800  |
| ga_gd/seed44       | sif~grad_norm_sq                     |  0.9998  |  1800  |
| scrub/seed42       | alignment_cos_sq~grad_norm_sq        | -0.6595  |  1800  |
| scrub/seed42       | fuma_lora_ratio~alignment_cos_sq     |  0.5998  |  1800  |
| scrub/seed42       | grad_ratio~grad_norm_sq              | -0.4183  |  1800  |
| scrub/seed42       | loss_ratio~loss_unlearned            |  0.3768  |  1800  |
| scrub/seed42       | sif~grad_norm_sq                     |  0.9888  |  1800  |
| scrub/seed43       | alignment_cos_sq~grad_norm_sq        | -0.6003  |  1800  |
| scrub/seed43       | fuma_lora_ratio~alignment_cos_sq     |  0.5447  |  1800  |
| scrub/seed43       | grad_ratio~grad_norm_sq              | -0.3680  |  1800  |
| scrub/seed43       | loss_ratio~loss_unlearned            |  0.3611  |  1800  |
| scrub/seed43       | sif~grad_norm_sq                     |  0.9885  |  1800  |
| scrub/seed44       | alignment_cos_sq~grad_norm_sq        | -0.4748  |  1800  |
| scrub/seed44       | fuma_lora_ratio~alignment_cos_sq     |  0.4452  |  1800  |
| scrub/seed44       | grad_ratio~grad_norm_sq              | -0.2832  |  1800  |
| scrub/seed44       | loss_ratio~loss_unlearned            |  0.2954  |  1800  |
| scrub/seed44       | sif~grad_norm_sq                     |  0.9881  |  1800  |

## Suppressed operating points

- FPR 0.001: unresolvable: 600 negatives give a minimum non-zero FPR of 0.0017 > 0.001; any value here would be the TPR at zero false positives, set by a single negative sample
