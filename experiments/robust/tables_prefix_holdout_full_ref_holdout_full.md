
### test F1 (at_selected), mean ± sample SD
| config | protocol | steps | n | full | no_o2 | o2_copy | no_opt |
|---|---|---|---|---|---|---|---|
| holdout_full | holdout | 390 | 3 | 0.539 ± 0.047 | 0.148 ± 0.017 | 0.371 ± 0.062 | 0.435 ± 0.010 |
| holdout_full_s1000 | holdout | 1000 | 3 | 0.563 ± 0.016 | 0.124 ± 0.015 | 0.345 ± 0.080 | 0.433 ± 0.004 |
| holdout_full_s2500 | holdout | 2500 | 3 | 0.554 ± 0.030 | 0.168 ± 0.064 | 0.328 ± 0.043 | 0.331 ± 0.198 |

### test F1 (at_0.5), mean ± sample SD
| config | protocol | steps | n | full | no_o2 | o2_copy | no_opt |
|---|---|---|---|---|---|---|---|
| holdout_full | holdout | 390 | 3 | 0.472 ± 0.081 | 0.107 ± 0.007 | 0.404 ± 0.041 | 0.405 ± 0.079 |
| holdout_full_s1000 | holdout | 1000 | 3 | 0.486 ± 0.030 | 0.099 ± 0.002 | 0.312 ± 0.047 | 0.292 ± 0.093 |
| holdout_full_s2500 | holdout | 2500 | 3 | 0.495 ± 0.070 | 0.109 ± 0.009 | 0.251 ± 0.115 | 0.218 ± 0.180 |

### paired vs holdout_full (F1 at validation-selected threshold)
| config | condition | n | Δ mean | paired t p | Holm p (t, over conditions) | Wilcoxon p |
|---|---|---|---|---|---|---|
| holdout_full_s1000 | full | 3 | +0.024 | 0.4039 | 1.0000 | 0.5000 |
| holdout_full_s1000 | no_o2 | 3 | -0.024 | 0.0021 | 0.0083 | 0.2500 |
| holdout_full_s1000 | o2_copy | 3 | -0.026 | 0.6541 | 1.0000 | 0.7500 |
| holdout_full_s1000 | no_opt | 3 | -0.003 | 0.7431 | 1.0000 | 1.0000 |
| holdout_full_s2500 | full | 3 | +0.015 | 0.6675 | 1.0000 | 1.0000 |
| holdout_full_s2500 | no_o2 | 3 | +0.020 | 0.5638 | 1.0000 | 0.7500 |
| holdout_full_s2500 | o2_copy | 3 | -0.043 | 0.1522 | 0.6090 | 0.2500 |
| holdout_full_s2500 | no_opt | 3 | -0.104 | 0.4478 | 1.0000 | 0.7500 |
