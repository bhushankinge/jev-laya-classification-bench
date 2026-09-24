paired single-class quote-gold rows: 741

## primary accuracy, ECE, hardware->software confusions (paired rows)

| source | n | accuracy | ECE | median conf | HW called Software |
|---|---|---|---|---|---|
| jev | 741 | 0.919 | 0.0487 | 1.00 | 7/568 |
| qwen | 741 | 0.896 | n/a | 0.90 | 20/568 |
| A-S2 | 741 | 0.780 | 0.3222 | 0.45 | 103/568 |
| A-S2-head-bf16 | 741 | 0.780 | 0.3222 | 0.45 | 103/568 |
| A-S2-head-fp16 | 741 | 0.779 | 0.3224 | 0.44 | 104/568 |
| A-S2-head-typed | 741 | 0.619 | 0.4216 | 0.16 | 191/568 |
| A-S2-head-multi | 741 | 0.675 | 0.253 | 0.40 | 30/568 |

## primary accuracy by S2 state length (characters)

| S2 length | n | jev | qwen | A-S2 | A-S2-head-bf16 | A-S2-head-fp16 | A-S2-head-typed | A-S2-head-multi |
|---|---|---|---|---|---|---|---|---|
| 0 to 600 | 518 | 0.921 | 0.894 | 0.817 | 0.817 | 0.815 | 0.633 | 0.701 |
| 600 to 1,200 | 116 | 0.905 | 0.905 | 0.655 | 0.655 | 0.655 | 0.578 | 0.612 |
| 1,200 to 2,000 | 59 | 0.898 | 0.864 | 0.729 | 0.729 | 0.729 | 0.610 | 0.610 |
| 2,000 to 3,000 | 48 | 0.958 | 0.938 | 0.750 | 0.750 | 0.750 | 0.583 | 0.625 |

## primary accuracy by vehicle

| vehicle | n | jev | qwen | A-S2 | A-S2-head-bf16 | A-S2-head-fp16 | A-S2-head-typed | A-S2-head-multi |
|---|---|---|---|---|---|---|---|---|
| GSA 2GIT | 95 | 0.884 | 0.863 | 0.874 | 0.874 | 0.874 | 0.611 | 0.695 |
| GSA MAS | 47 | 0.915 | 0.872 | 0.617 | 0.617 | 0.617 | 0.511 | 0.660 |
| SEWP | 599 | 0.925 | 0.903 | 0.778 | 0.778 | 0.776 | 0.629 | 0.673 |

## share of rows with the has_* noul >= 0.5, and mean noul (all labeled rows)

| subclass | jev | qwen | A-S2 | A-S2-head-bf16 | A-S2-head-fp16 | A-S2-head-typed | A-S2-head-multi |
|---|---|---|---|---|---|---|---|
| General hardware | 0.555 | 0.469 | 0.479 (mean 0.50) | 0.669 (mean 0.60) | 0.669 (mean 0.60) | 0.661 (mean 0.52) | 0.851 (mean 0.82) |
| Consumables/Supplies | 0.036 | 0.039 | 0.133 (mean 0.30) | 0.203 (mean 0.35) | 0.202 (mean 0.35) | 0.058 (mean 0.32) | 0.667 (mean 0.64) |
| Perpetual license | 0.039 | 0.108 | 0.402 (mean 0.46) | 0.449 (mean 0.50) | 0.452 (mean 0.50) | 0.165 (mean 0.39) | 0.490 (mean 0.48) |
| Subscription/SaaS | 0.349 | 0.227 | 0.601 (mean 0.58) | 0.639 (mean 0.60) | 0.640 (mean 0.60) | 0.521 (mean 0.50) | 0.409 (mean 0.42) |
| Cloud/Hosting | 0.010 | 0.015 | 0.174 (mean 0.30) | 0.209 (mean 0.33) | 0.208 (mean 0.33) | 0.182 (mean 0.38) | 0.625 (mean 0.61) |
| Support/Maintenance contract | 0.334 | 0.140 | 0.693 (mean 0.63) | 0.727 (mean 0.65) | 0.727 (mean 0.65) | 0.589 (mean 0.52) | 0.683 (mean 0.66) |
| Warranty | 0.030 | 0.015 | 0.451 (mean 0.50) | 0.502 (mean 0.53) | 0.503 (mean 0.53) | 0.366 (mean 0.45) | 0.385 (mean 0.39) |
| Professional/Consulting | 0.128 | 0.048 | 0.504 (mean 0.52) | 0.580 (mean 0.55) | 0.583 (mean 0.55) | 0.477 (mean 0.49) | 0.292 (mean 0.33) |
| Training | 0.014 | 0.008 | 0.038 (mean 0.17) | 0.051 (mean 0.18) | 0.052 (mean 0.18) | 0.096 (mean 0.32) | 0.237 (mean 0.25) |
| Installation/Integration | 0.011 | 0.020 | 0.150 (mean 0.29) | 0.176 (mean 0.32) | 0.181 (mean 0.32) | 0.225 (mean 0.41) | 0.646 (mean 0.63) |
| Furniture/Facilities | 0.033 | 0.038 | 0.096 (mean 0.22) | 0.120 (mean 0.24) | 0.120 (mean 0.24) | 0.049 (mean 0.30) | 0.191 (mean 0.23) |
| Other | 0.408 | 0.038 | 0.215 (mean 0.36) | 0.261 (mean 0.38) | 0.264 (mean 0.39) | 0.067 (mean 0.35) | 0.408 (mean 0.41) |

## flags (share of rows >= 0.5), lifecycle, components per row, latency (all labeled rows)

| source | n | rfi_market_research | text_insufficient | brand_name_only | renewal | replacement/refresh | unknown | components/row | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| jev | 12000 | 0.13 | 0.15 | 0.54 | 2311 | 714 | 6447 | 1.95 | nan |
| qwen | 11931 | 0.13 | 0.13 | 0.39 | 2104 | 0 | 0 | 1.16 | nan |
| A-S2 | 12000 | 0.63 | 0.71 | 0.49 | 4085 | 2534 | 867 | 4.05 | 299 |
| A-S2-head-bf16 | 927 | 0.69 | 0.76 | 0.56 | 172 | 266 | 63 | 4.67 | 289 |
| A-S2-head-fp16 | 927 | 0.69 | 0.76 | 0.56 | 175 | 267 | 64 | 4.69 | 292 |
| A-S2-head-typed | 927 | 0.39 | 0.43 | 0.27 | 167 | 74 | 250 | 3.62 | 286 |
| A-S2-head-multi | 927 | 0.69 | 0.57 | 0.48 | 106 | 216 | 185 | 6.00 | 118 |
