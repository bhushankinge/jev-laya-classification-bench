# Experiment runbook

All commands from `<repo>`. Dates in file names are the day the file was built; never overwrite a gold file.

## E0 Gold
python3 -m pipeline.gold fingerprints          # add HPE/Dell keys to CONFIGURATOR if present, re-run
python3 -m pipeline.gold composition && python3 -m pipeline.gold fulfillment

## E1 Question and state variants (2,000-row subset first)
for v in A B C; do for s in S1 S2 S3; do python3 -m pipeline.label_jev --variant $v --state $s --limit 2000; done; done
<bench-repo>/.venv/bin/python -m pipeline.label_laya --variant A --state S2 --limit 2000
Laya rows: run under `<bench-repo>/.venv/bin/python`.
python3 -m pipeline.label_qwen                     # all 12,000 rows, v2 schema (about 45 min at 48 concurrent)
for v in A B C; do for s in S1 S2 S3; do python3 -m pipeline.evaluate --run e1-$v-$s --jev $v-$s --qwen v2 \
   --composition gold/composition-<date>.jsonl --fulfillment gold/fulfillment-<date>.jsonl; done; done
Pick the (variant, state) with the highest primary-class accuracy whose cutoff_95 coverage is highest; ties go to the cheaper state.
Then label all 12,000 rows with the winner for Jev and Laya.

## E2 Cutoffs
python3 -m pipeline.evaluate --run e2-full --jev <win> --laya <win> --qwen v2 --composition ... --fulfillment ...
Read cutoff_95 (primary) and cutoff_90 (fulfillment) per source from results/e2-full/metrics.json; coverage at the cutoff is the auto-accept share.

## E3 Human gold and disagreements (needs the CRM review page live)
python3 -m pipeline.review_queue enqueue --run gold-<date> --reason gold_sample --per-vehicle 200 --qwen v2 --jev <win> --laya <win> --token-file ~/.crm-review-token --base https://<crm-host>
python3 -m pipeline.review_queue enqueue --run dis-<date> --reason disagreement --limit 300 --qwen v2 --jev <win> --token-file ... --base ...
python3 -m pipeline.review_queue enqueue --run ful-<date> --reason fulfillment --per-vehicle 70 --qwen v2 --jev <win> --token-file ... --base ...
# after reviewers finish:
python3 -m pipeline.review_queue pull --run gold-<date> ...; pull dis-<date>; pull ful-<date>
python3 -m pipeline.evaluate --run e3-human --jev <win> --laya <win> --qwen v2 --composition ... --fulfillment ... --human gold/human-<date>.jsonl

## E4 Fulfillment
Covered by fulfillment_vs_quote_gold and vs_human_gold.fulfillment_accuracy in e2-full and e3-human. Also report the proxy rule's own precision: share of fulfillment-gold rows the humans confirmed (compute from gold/human-<date>.jsonl where reason == "fulfillment").

## E5 Cost and latency
Jev: mean input_tokens × $0.042 / 1e6 per opportunity; p50/p95 of latency_ms from labels/jev/<win>.jsonl.
Qwen: total_tokens and wall time from qwen_classify run log; GPU seconds = wall time × 1 GPU.
Laya: rows/s from label_laya stderr; $/M decisions from ~/<bench-repo> results (bench report).
Record all three in results/e5-cost.md.

## E6 Regression
Any prompt or model change: re-run the labeler into a new run name, then `python3 -m pipeline.evaluate --run <new> ...` and diff metrics.json against the previous run.

## Stop rules
- Jev returns 429s: the labeler backs off; if sustained, lower MAX_RPS to 5.
- Any labeler error rate above 2%: stop, inspect the error rows, fix, resume (all labelers resume by id).
- Qwen batches only at ≤ 48 concurrent, preferably outside 08:00–18:00 Phoenix.
- Laya's 512-token context truncates silently; S3 is only partially seen by Laya.
