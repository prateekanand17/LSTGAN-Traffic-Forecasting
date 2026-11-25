# Rule-agent cells (W-C): runs rule_cells_s42, rule_cells_s123, rule_cells_s456

Forecaster seeds: [42, 123, 456]. Unit: scenario (seeds averaged). Conflict metrics on contested scenario-seeds (>= 1 round-0 conflict); WZPA on all scenarios with zones. CI: 95% bootstrap.

## 1. Cell means

| cell | final conflicts | conflict-free | CRR | rounds | messages | link density | fallback | WZPA@30 | WZPA@60 | n (contested) |
|---|---|---|---|---|---|---|---|---|---|---|
| B_rules | 6.689 | 0.000 | 0.000 | 0.000 | 0.000 | 0.524 | 0.000 | 0.810 | 0.731 | 74 |
| B_rules_h | 6.689 | 0.000 | 0.000 | 0.000 | 0.000 | 0.524 | 0.000 | 0.817 | 0.852 | 74 |
| R_none | 6.689 | 0.000 | 0.000 | 0.000 | 0.000 | 0.524 | 0.000 | 0.817 | 0.852 | 74 |
| R_central | 0.000 | 1.000 | 0.994 | 0.000 | 0.000 | 0.524 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg | 2.101 | 0.164 | 0.663 | 1.218 | 83.225 | 0.524 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_fb | 0.000 | 1.000 | 0.994 | 1.218 | 83.225 | 0.524 | 0.836 | 0.817 | 0.852 | 74 |
| R_neg_K1 | 2.606 | 0.146 | 0.666 | 0.809 | 47.270 | 0.524 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_K2 | 2.142 | 0.160 | 0.663 | 1.178 | 79.171 | 0.524 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_consensus | 2.624 | 0.137 | 0.603 | 1.146 | 65.207 | 0.433 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_consensus3 | 5.709 | 0.034 | 0.168 | 0.340 | 3.459 | 0.059 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_learned_dense | 0.005 | 0.995 | 0.994 | 1.689 | 169.468 | 1.000 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_distance | 4.293 | 0.027 | 0.405 | 0.804 | 10.455 | 0.110 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_knn | 4.797 | 0.014 | 0.294 | 0.682 | 4.914 | 0.065 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_knn40 | 3.394 | 0.101 | 0.509 | 0.912 | 22.743 | 0.234 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_knn60 | 2.806 | 0.144 | 0.598 | 1.005 | 34.302 | 0.327 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_knn80 | 1.968 | 0.171 | 0.692 | 1.216 | 55.009 | 0.422 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_knn100 | 1.586 | 0.207 | 0.749 | 1.288 | 68.423 | 0.505 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_full | 0.005 | 0.995 | 0.994 | 1.689 | 169.468 | 1.000 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_random_mean | 2.495 | 0.126 | 0.593 | 1.113 | 70.748 | 0.464 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_random_consensus_mean | 3.389 | 0.100 | 0.503 | 0.949 | 42.941 | 0.330 | 0.000 | 0.817 | 0.852 | 74 |
| R_neg_none | 6.689 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.817 | 0.852 | 74 |

## 2. Planned comparisons (cell − reference; Holm across the table)

| cell | reference | metric | n | n non-zero | mean diff | 95% CI | Wilcoxon p | Holm p | r_rb |
|---|---|---|---|---|---|---|---|---|---|
| R_neg_random_mean | R_neg | n_conflicts_final | 74 | 63 | +0.394 | [0.241, 0.552] | 2.05e-06 | 2.67e-05 | +0.69 |
| R_neg_random_mean | R_neg | conflict_free | 74 | 34 | -0.038 | [-0.085, 0.003] | 0.296 | 1 | -0.21 |
| R_neg_knn100 | R_neg | n_conflicts_final | 74 | 56 | -0.516 | [-0.833, -0.218] | 0.000877 | 0.00789 | -0.51 |
| R_neg_knn100 | R_neg | conflict_free | 74 | 33 | +0.043 | [-0.036, 0.119] | 0.175 | 0.874 | +0.27 |
| R_neg_distance | R_neg | n_conflicts_final | 74 | 66 | +2.191 | [1.606, 2.827] | 1.27e-09 | 2.04e-08 | +0.86 |
| R_neg_distance | R_neg | conflict_free | 74 | 21 | -0.137 | [-0.205, -0.074] | 0.000205 | 0.00205 | -0.91 |
| R_neg_knn | R_neg | n_conflicts_final | 74 | 66 | +2.696 | [2.079, 3.351] | 3.45e-11 | 5.86e-10 | +0.94 |
| R_neg_knn | R_neg | conflict_free | 74 | 20 | -0.151 | [-0.216, -0.090] | 7.37e-05 | 0.000811 | -1.00 |
| R_neg_full | R_neg | n_conflicts_final | 74 | 71 | -2.097 | [-2.516, -1.718] | 2.29e-13 | 4.8e-12 | -1.00 |
| R_neg_full | R_neg | conflict_free | 74 | 71 | +0.831 | [0.764, 0.892] | 1.56e-14 | 3.44e-13 | +1.00 |
| R_neg_consensus | R_neg | n_conflicts_final | 74 | 62 | +0.523 | [0.273, 0.800] | 0.000985 | 0.00789 | +0.48 |
| R_neg_consensus | R_neg | conflict_free | 74 | 22 | -0.027 | [-0.097, 0.038] | 0.542 | 1 | -0.15 |
| R_neg_consensus | R_neg_random_consensus_mean | n_conflicts_final | 74 | 67 | -0.765 | [-1.044, -0.495] | 1.6e-07 | 2.23e-06 | -0.74 |
| R_neg_consensus | R_neg_random_consensus_mean | conflict_free | 74 | 25 | +0.037 | [-0.013, 0.089] | 0.3 | 1 | +0.24 |
| R_none | R_neg | n_conflicts_final | 74 | 68 | +4.588 | [3.680, 5.554] | 7.51e-13 | 1.35e-11 | +1.00 |
| R_none | R_neg | conflict_free | 74 | 22 | -0.164 | [-0.232, -0.104] | 3.22e-05 | 0.000386 | -1.00 |
| R_central | R_neg | n_conflicts_final | 74 | 71 | -2.101 | [-2.523, -1.718] | 2.29e-13 | 4.8e-12 | -1.00 |
| R_central | R_neg | conflict_free | 74 | 71 | +0.836 | [0.768, 0.896] | 1.3e-14 | 3.13e-13 | +1.00 |
| R_neg_fb | R_neg | n_conflicts_final | 74 | 71 | -2.101 | [-2.523, -1.718] | 2.29e-13 | 4.8e-12 | -1.00 |
| R_neg_fb | R_neg | conflict_free | 74 | 71 | +0.836 | [0.768, 0.896] | 1.3e-14 | 3.13e-13 | +1.00 |
| R_neg_K1 | R_neg | n_conflicts_final | 74 | 42 | +0.505 | [0.369, 0.649] | 1.62e-08 | 2.43e-07 | +1.00 |
| R_neg_K1 | R_neg | conflict_free | 74 | 4 | -0.018 | [-0.036, -0.005] | 0.0455 | 0.273 | -1.00 |
| R_neg_K2 | R_neg | n_conflicts_final | 74 | 6 | +0.041 | [0.009, 0.081] | 0.0277 | 0.194 | +1.00 |
| R_neg_K2 | R_neg | conflict_free | 74 | 1 | -0.005 | [-0.014, 0.000] | 0.317 | 1 | -1.00 |

## 3. Final conflicts per forecaster seed (contested scenarios)

| seed | R_none | R_neg | R_neg_random_mean | R_neg_knn100 | R_neg_distance | R_neg_none | R_neg_full | R_neg_consensus | R_central |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 7.358 (n=67) | 2.791 (n=67) | 3.304 (n=67) | 1.687 (n=67) | 4.687 (n=67) | 7.358 (n=67) | 0.000 (n=67) | 2.821 (n=67) | 0.000 (n=67) |
| 123 | 6.957 (n=70) | 2.043 (n=70) | 2.614 (n=70) | 1.571 (n=70) | 4.343 (n=70) | 6.957 (n=70) | 0.000 (n=70) | 2.700 (n=70) | 0.000 (n=70) |
| 456 | 7.058 (n=69) | 1.797 (n=69) | 1.971 (n=69) | 1.696 (n=69) | 4.594 (n=69) | 7.058 (n=69) | 0.014 (n=69) | 2.768 (n=69) | 0.000 (n=69) |

## 4. Final conflicts by regime (contested scenarios)

| cell | HIGH | MODERATE | CASCADE | NORMAL |
|---|---|---|---|---|
| R_none | 9.988 (n=27) | 3.529 (n=17) | 5.667 (n=29) | 1.000 (n=1) |
| R_neg | 3.049 (n=27) | 1.206 (n=17) | 1.805 (n=29) | 0.333 (n=1) |
| R_neg_random_mean | 3.798 (n=27) | 1.335 (n=17) | 2.043 (n=29) | 0.200 (n=1) |
| R_neg_knn100 | 2.247 (n=27) | 1.216 (n=17) | 1.241 (n=29) | 0.000 (n=1) |
| R_neg_distance | 6.877 (n=27) | 2.020 (n=17) | 3.333 (n=29) | 1.000 (n=1) |
| R_neg_none | 9.988 (n=27) | 3.529 (n=17) | 5.667 (n=29) | 1.000 (n=1) |
| R_central | 0.000 (n=27) | 0.000 (n=17) | 0.000 (n=29) | 0.000 (n=1) |
