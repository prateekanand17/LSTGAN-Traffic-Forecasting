# Summary of run `rules_baseline_30pr`

Records: 480. Unit: scenario (runs averaged first). CI: 95% bootstrap of the mean.

| variant | metric | mean | 95% CI | n |
|---|---|---|---|---|
| B_rules | wzpa30 | 0.816 | [0.789, 0.843] | 95 |
| B_rules | wzpa60 | 0.742 | [0.708, 0.777] | 95 |
| B_rules | zpa30_exact | 0.635 | [0.580, 0.687] | 95 |
| B_rules_h | wzpa30 | 0.825 | [0.791, 0.855] | 95 |
| B_rules_h | wzpa60 | 0.866 | [0.839, 0.892] | 95 |
| B_rules_h | zpa30_exact | 0.671 | [0.617, 0.721] | 95 |
| B_rules_oracle | wzpa30 | 0.895 | [0.872, 0.916] | 96 |
| B_rules_oracle | wzpa60 | 0.754 | [0.718, 0.788] | 96 |
| B_rules_oracle | zpa30_exact | 0.790 | [0.746, 0.831] | 96 |
| B_rules_reactive | wzpa30 | 0.707 | [0.666, 0.748] | 95 |
| B_rules_reactive | wzpa60 | 0.614 | [0.567, 0.661] | 95 |
| B_rules_reactive | zpa30_exact | 0.500 | [0.435, 0.566] | 95 |

## Paired differences vs B_rules (variant − ref)

| variant | metric | n | mean diff | 95% CI | Wilcoxon p | Holm p | r_rb |
|---|---|---|---|---|---|---|---|
| B_rules_h | wzpa30 | 95 | +0.009 | [-0.011, 0.029] | 0.206 | 0.411 | +0.26 |
| B_rules_h | wzpa60 | 95 | +0.124 | [0.084, 0.164] | 7.51e-08 | 6.01e-07 | +0.73 |
| B_rules_h | zpa30_exact | 95 | +0.036 | [0.008, 0.069] | 0.0313 | 0.094 | +0.47 |
| B_rules_oracle | wzpa30 | 94 | +0.076 | [0.048, 0.105] | 6.88e-07 | 3.91e-06 | +0.66 |
| B_rules_oracle | wzpa60 | 94 | +0.012 | [-0.018, 0.042] | 0.468 | 0.468 | +0.09 |
| B_rules_oracle | zpa30_exact | 94 | +0.150 | [0.097, 0.207] | 6.51e-07 | 3.91e-06 | +0.67 |
| B_rules_reactive | wzpa30 | 92 | -0.108 | [-0.146, -0.071] | 8.06e-08 | 6.01e-07 | -0.71 |
| B_rules_reactive | wzpa60 | 92 | -0.129 | [-0.167, -0.094] | 5.03e-09 | 4.53e-08 | -0.78 |
| B_rules_reactive | zpa30_exact | 92 | -0.135 | [-0.198, -0.070] | 3.99e-05 | 0.000159 | -0.55 |
