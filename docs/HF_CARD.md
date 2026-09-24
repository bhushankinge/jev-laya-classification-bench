---
license: apache-2.0
language:
  - en
pretty_name: Jev / Laya / Qwen classification bench results
task_categories:
  - text-classification
tags:
  - benchmark
  - calibration
  - typed-decisions
  - llm-evaluation
  - laya
  - qwen
size_categories:
  - n<1K
---

# Jev / Laya / Qwen classification bench results

Aggregate results of a benchmark in which **Jev** (TypeSafe System One API, `jev-1.13.0`), **Laya** ([`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), 421M) and **Qwen3.5-35B-A3B** classified the same 12,000 U.S. federal IT solicitations, graded against what a reseller actually quoted.

Code, method and full report: **https://github.com/bhushankinge/jev-laya-classification-bench**

![Headline results](https://raw.githubusercontent.com/bhushankinge/jev-laya-classification-bench/main/docs/figures/hero-light.png)

| Source | Primary-class accuracy (n=741) | ECE | Auto-accepted at 95% precision | Fulfillment-mode accuracy (n=634) |
|---|---|---|---|---|
| Jev A-S2 | 91.9% | 0.049 | 86.5% (cutoff 0.94) | 65.0% |
| Qwen3.5-35B-A3B | 89.6% | n/a | not reachable | 71.0% |
| Laya 421M A-S2 | 78.0% | 0.322 | not reachable | 45.7% |

![Precision vs coverage](https://raw.githubusercontent.com/bhushankinge/jev-laya-classification-bench/main/docs/figures/precision_coverage-light.png)

## What is in this dataset

Only aggregate metrics: one `metrics.json` and `env.json` per run (`e1-*` for the nine Jev prompt variants and Laya, `e2-full` for the final three-source comparison), plus `e5-cost.md`. The solicitation text, gold labels and per-row predictions carry a company's opportunity and quote ids and are not published.

## Laya with shipped defaults

Laya ran single-row with default config. Presence questions over-fire, accuracy drops before the 512-token window fills, and the sequence packer truncates class definitions to 25 tokens. Six testable hypotheses are listed in the [README](https://github.com/bhushankinge/jev-laya-classification-bench#laya-reproducible-issues-and-hypotheses). Replications are welcome.

## Citation

```bibtex
@software{kinge2026jevlaya,
  author  = {Kinge, Bhushan},
  title   = {jev-laya-classification-bench: Typed-decision models versus an LLM on 12,000 federal IT solicitations},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/bhushankinge/jev-laya-classification-bench},
  license = {Apache-2.0}
}
```
