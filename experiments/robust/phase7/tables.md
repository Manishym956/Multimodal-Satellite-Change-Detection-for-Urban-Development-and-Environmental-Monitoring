### PRIMARY (RQ3): SAR-only U-Net, test F1 on 9 cities, frozen threshold, Holm over 2

| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |
|---|---|---|---|---|---|---|---|
| VV+VH - VV | -0.044 | [-0.065, -0.023] | -1.49 | 1/10 | 0.0011 | 0.0039 | VV+VH lower |
| VH - VV | -0.070 | [-0.091, -0.048] | -2.32 | 0/10 | 8.8e-05 | 0.0039 | VH lower |

### Secondary: DS-UNet, test F1 on 9 cities (own Holm family)

| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |
|---|---|---|---|---|---|---|---|
| VV+VH - VV | +0.010 | [-0.016, +0.036] | +0.27 | 7/10 | 0.4125 | 0.3750 | no detectable difference |
| VH - VV | -0.017 | [-0.039, +0.004] | -0.59 | 3/10 | 0.1925 | 0.2109 | no detectable difference |

### Secondary: SAR-only U-Net, false-alarm rate by WorldCover class (own Holm family)

| contrast | mean Δ | 95% CI | d_z | Δ > 0 | Holm p (t) | Holm p (Wilcoxon) | label |
|---|---|---|---|---|---|---|---|
| VV+VH - VV, tree cover | -0.004 | [-0.006, -0.001] | -1.04 | 2/10 | 0.0380 | 0.0586 | VV+VH lower |
| VV+VH - VV, shrubland | -0.006 | [-0.013, +0.002] | -0.54 | 3/10 | 0.2391 | 0.2617 | no detectable difference |
| VV+VH - VV, grassland | +0.001 | [-0.007, +0.010] | +0.12 | 7/10 | 0.7155 | 0.7695 | no detectable difference |
| VV+VH - VV, cropland | -0.012 | [-0.025, +0.001] | -0.64 | 4/10 | 0.2182 | 0.2520 | no detectable difference |
| VV+VH - VV, built-up | -0.022 | [-0.035, -0.010] | -1.30 | 2/10 | 0.0162 | 0.0586 | VV+VH lower |
| VV+VH - VV, bare | -0.042 | [-0.068, -0.016] | -1.16 | 1/10 | 0.0257 | 0.0586 | VV+VH lower |
| VV+VH - VV, water | -0.007 | [-0.010, -0.004] | -1.54 | 0/10 | 0.0061 | 0.0137 | VV+VH lower |

### Test metrics, full input, 9 cities, mean ± SD over 10 seeds

| arm | F1 | precision | recall | IoU | false-alarm rate |
|---|---|---|---|---|---|
| SAR-only U-Net, VV | 0.452 ± 0.011 | 0.367 ± 0.017 | 0.590 ± 0.029 | 0.292 ± 0.009 | 0.049 ± 0.005 |
| SAR-only U-Net, VH | 0.383 ± 0.029 | 0.346 ± 0.048 | 0.442 ± 0.069 | 0.237 ± 0.022 | 0.042 ± 0.016 |
| SAR-only U-Net, VV+VH | 0.408 ± 0.031 | 0.405 ± 0.040 | 0.424 ± 0.081 | 0.257 ± 0.024 | 0.031 ± 0.010 |
| DS-UNet, VV | 0.501 ± 0.024 | 0.523 ± 0.067 | 0.500 ± 0.089 | 0.335 ± 0.021 | 0.024 ± 0.013 |
| DS-UNet, VH | 0.484 ± 0.043 | 0.511 ± 0.030 | 0.462 ± 0.061 | 0.320 ± 0.036 | 0.021 ± 0.002 |
| DS-UNet, VV+VH | 0.511 ± 0.019 | 0.571 ± 0.029 | 0.465 ± 0.036 | 0.344 ± 0.017 | 0.017 ± 0.003 |

### Test metrics, full input, 10 cities, mean ± SD over 10 seeds

| arm | F1 | precision | recall | IoU | false-alarm rate |
|---|---|---|---|---|---|
| SAR-only U-Net, VV | 0.418 ± 0.043 | 0.312 ± 0.051 | 0.648 ± 0.027 | 0.265 ± 0.035 | 0.081 ± 0.020 |
| SAR-only U-Net, VH | 0.301 ± 0.048 | 0.293 ± 0.093 | 0.353 ± 0.092 | 0.178 ± 0.032 | 0.061 ± 0.058 |
| SAR-only U-Net, VV+VH | 0.224 ± 0.032 | 0.146 ± 0.045 | 0.562 ± 0.095 | 0.127 ± 0.021 | 0.194 ± 0.051 |
| DS-UNet, VV | 0.556 ± 0.029 | 0.568 ± 0.071 | 0.562 ± 0.090 | 0.385 ± 0.028 | 0.025 ± 0.013 |
| DS-UNet, VH | 0.520 ± 0.041 | 0.542 ± 0.036 | 0.504 ± 0.059 | 0.352 ± 0.036 | 0.023 ± 0.005 |
| DS-UNet, VV+VH | 0.485 ± 0.132 | 0.496 ± 0.185 | 0.533 ± 0.057 | 0.329 ± 0.104 | 0.053 ± 0.070 |

### False-alarm rate and recall by WorldCover class (9 cities, full input)

| arm | tree cover | shrubland | grassland | cropland | built-up | bare | water |
|---|---|---|---|---|---|---|---|
| SAR-only U-Net, VV | FAR 0.009, R 0.372 | FAR 0.016, R 0.151 | FAR 0.026, R 0.408 | FAR 0.043, R 0.357 | FAR 0.059, R 0.650 | FAR 0.097, R 0.610 | FAR 0.009, R 0.163 |
| SAR-only U-Net, VH | FAR 0.005, R 0.216 | FAR 0.034, R 0.087 | FAR 0.026, R 0.244 | FAR 0.046, R 0.228 | FAR 0.043, R 0.494 | FAR 0.095, R 0.475 | FAR 0.003, R 0.145 |
| SAR-only U-Net, VV+VH | FAR 0.005, R 0.208 | FAR 0.010, R 0.094 | FAR 0.027, R 0.242 | FAR 0.031, R 0.232 | FAR 0.036, R 0.485 | FAR 0.055, R 0.430 | FAR 0.002, R 0.093 |
| DS-UNet, VV | FAR 0.002, R 0.260 | FAR 0.020, R 0.216 | FAR 0.026, R 0.359 | FAR 0.019, R 0.255 | FAR 0.030, R 0.556 | FAR 0.041, R 0.512 | FAR 0.007, R 0.210 |
| DS-UNet, VH | FAR 0.002, R 0.253 | FAR 0.024, R 0.190 | FAR 0.026, R 0.328 | FAR 0.021, R 0.269 | FAR 0.025, R 0.512 | FAR 0.036, R 0.472 | FAR 0.006, R 0.195 |
| DS-UNet, VV+VH | FAR 0.001, R 0.239 | FAR 0.018, R 0.198 | FAR 0.024, R 0.368 | FAR 0.016, R 0.254 | FAR 0.020, R 0.525 | FAR 0.028, R 0.451 | FAR 0.006, R 0.233 |

Pixel share by class: tree cover 0.204, shrubland 0.014, grassland 0.084, cropland 0.117, built-up 0.373, bare 0.182, water 0.018, herbaceous wetland 0.007

