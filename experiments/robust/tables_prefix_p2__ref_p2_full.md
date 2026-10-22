
### test F1 (at_selected), mean ± sample SD
| config | protocol | steps | n | full | no_o2 | o2_copy | no_opt |
|---|---|---|---|---|---|---|---|
| p2_dropout | holdout | 2500 | 10 | 0.553 ± 0.030 | 0.431 ± 0.034 | 0.438 ± 0.038 | 0.440 ± 0.027 |
| p2_full | holdout | 2500 | 10 | 0.556 ± 0.029 | 0.190 ± 0.068 | 0.314 ± 0.034 | 0.397 ± 0.106 |

### test F1 (at_0.5), mean ± sample SD
| config | protocol | steps | n | full | no_o2 | o2_copy | no_opt |
|---|---|---|---|---|---|---|---|
| p2_dropout | holdout | 2500 | 10 | 0.427 ± 0.062 | 0.375 ± 0.043 | 0.438 ± 0.046 | 0.378 ± 0.051 |
| p2_full | holdout | 2500 | 10 | 0.513 ± 0.055 | 0.129 ± 0.030 | 0.236 ± 0.083 | 0.180 ± 0.100 |

### paired vs p2_full (F1 at validation-selected threshold)
| config | condition | n | Δ mean | paired t p | Holm p (t, over conditions) | Wilcoxon p |
|---|---|---|---|---|---|---|
| p2_dropout | full | 10 | -0.003 | 0.8449 | 0.8449 | 0.7695 |
| p2_dropout | no_o2 | 10 | +0.241 | 1.4e-05 | 5.8e-05 | 0.0020 |
| p2_dropout | o2_copy | 10 | +0.124 | 3.2e-05 | 9.5e-05 | 0.0020 |
| p2_dropout | no_opt | 10 | +0.043 | 0.2482 | 0.4964 | 0.3223 |
