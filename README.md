# jev-laya-classification-bench

Typed-decision models versus an LLM on a real classification job: **Jev** (TypeSafe System One API), **Laya** (its open-weights sibling, `convaiinnovations/laya`, run locally) and **Qwen3.5-35B-A3B** classify 12,000 U.S. federal IT solicitations, graded against what a reseller actually quoted.

Full write-up: [docs/reports/2026-09-24-classification-experiments-report.md](docs/reports/2026-09-24-classification-experiments-report.md). Design: [docs/specs](docs/specs/). Runbook: [pipeline/RUN.md](pipeline/RUN.md).

## Headline (741 opportunities with unambiguous quote gold, same rows for every source)

| Source | Primary-class accuracy | ECE | Cutoff for 95% precision (Wilson lower bound) | Auto-accept coverage at cutoff | Fulfillment-mode accuracy |
|---|---|---|---|---|---|
| Jev `jev-1.13.0`, variant A-S2 | **0.919** | 0.049 | 0.94 | **86.5%** | 0.650 |
| Qwen3.5-35B-A3B-FP8, structured JSON | 0.896 | n/a | not reachable | none | **0.710** |
| Laya 421M, variant A-S2, eager FP16 | 0.780 | 0.322 | not reachable | none | 0.457 |

- Nine Jev question and state-text variants land within four rows of each other; the cheapest won.
- Jev labeled all 12,000 rows for $0.78 at list price, median latency 185 ms.
- Laya, with shipped defaults, over-fires presence questions, mis-calibrates, and degrades with input length well before its 512-token window fills. The report's Section 9.4 traces part of that to the sequence packer's 192-token head budget, which silently truncates option definitions. Six testable hypotheses follow.
- Fulfillment mode (distributor SKU vs OEM-configured build) is the weak spot for every model.

## What is here

| Path | Contents |
|---|---|
| `pipeline/` | schema, question bundles and state builders, Jev / Laya / Qwen labelers (resumable JSONL), mapping onto one label schema, evaluator (paired scoring, Wilson-bounded cutoffs, ECE, gate simulation), review-queue client, HTML report |
| `tests/` | 49 tests, `python3 -m pytest -q` |
| `results/e1-*/`, `results/e2-full/` | every metric in this study as `metrics.json` plus `env.json` |
| `results/e5-cost.md` | cost and latency per source |
| `qwen_discovery/` | the Qwen discovery labeler and its aggregate findings that fixed the taxonomy |
| `docs/specs/` | the approved research design |

Not in this repository: the solicitation sample, the gold files and the label files. They carry the reseller's opportunity and quote ids. The evaluator runs on them locally; the aggregate results are committed.

## Reproduce with your own data

The pipeline expects a JSONL sample with `id, vehicle, title, description, rfq_type, lines, attachment_text` per row, a composition gold file `{id, vehicle, classes}` and a fulfillment gold file `{id, vehicle, fulfillment_mode}`. Secrets and endpoints come from environment variables; copy `.env.example` to `~/.config/laya-jev-bench.env`.

```bash
python3 -m pipeline.label_jev --variant A --state S2 --verify          # 3 rows, prints raw Jev answers
python3 -m pipeline.label_jev --variant A --state S2 --ids-from gold/composition-<date>.jsonl --vehicle-balanced
$LAYA_PYTHON -m pipeline.label_laya --variant A --state S2 --ids-from gold/composition-<date>.jsonl --vehicle-balanced
python3 -m pipeline.label_qwen
python3 -m pipeline.evaluate --run e2 --jev A-S2 --laya A-S2 --qwen v2 \
  --composition gold/composition-<date>.jsonl --fulfillment gold/fulfillment-<date>.jsonl
python3 -m pipeline.report --run e2
```

`pipeline/RUN.md` has the full E0 to E7 sequence and the pre-registered variant-selection rule.

## Status

E0 to E2 and E5 complete. E3 and E4 (human adjudication of disagreements, fulfillment validation) wait on hosting the review page. Companion repository: the Laya CUDA capacity bench (decisions per second under p99 SLOs on four NVIDIA GPUs), published separately.

## License

Apache-2.0. Copyright 2026 Bhushan Kinge.
