# Experiment runbook

All commands from `<repo>`. Dates in file names are the day the file was built; never overwrite a gold file.

## E0 Sample and gold
python3 qwen_discovery/qwen_classify.py 0      # builds qwen_discovery/sample.jsonl, labels nothing
python3 -m pipeline.gold fingerprints          # add HPE/Dell keys to CONFIGURATOR if present, re-run
python3 -m pipeline.gold composition && python3 -m pipeline.gold fulfillment
Gold builds refuse to overwrite an existing file for today: move the old one aside first.
Fulfillment gold covers the hardware lines of the latest typed quote per opportunity.

## E1 Question and state variants (1,000-row subset first)
The subset is drawn from the composition gold so every row can be scored, and balanced across vehicles.
python3 -m pipeline.label_jev --variant A --state S2 --verify        # 3 rows, prints raw answers; do this first
for v in A B C; do for s in S1 S2 S3; do python3 -m pipeline.label_jev --variant $v --state $s \
   --ids-from gold/composition-<date>.jsonl --vehicle-balanced --limit 1000; done; done
<bench-repo>/.venv/bin/python -m pipeline.label_laya --variant A --state S2 --verify
<bench-repo>/.venv/bin/python -m pipeline.label_laya --variant A --state S2 \
   --ids-from gold/composition-<date>.jsonl --vehicle-balanced --limit 1000
Laya rows: run under `<bench-repo>/.venv/bin/python`.
python3 -m pipeline.label_qwen 5                   # smoke, then:
python3 -m pipeline.label_qwen 2>&1 | tee results/e5-qwen.log   # all 12,000 rows, v2 schema (~45 min at 48 concurrent)
for v in A B C; do for s in S1 S2 S3; do python3 -m pipeline.evaluate --run e1-$v-$s --jev $v-$s --qwen v2 \
   --composition gold/composition-<date>.jsonl --fulfillment gold/fulfillment-<date>.jsonl; done; done
Selection rule (one rule, in this order, all from metrics.json):
1. highest `paired.primary_vs_quote_gold.accuracy` (the id set every source labeled);
2. tie -> higher `paired.primary_vs_quote_gold.coverage_at_cutoff` (the Wilson-bounded cutoff_95; a null
   cutoff counts as coverage 0);
3. tie -> the cheaper state (S1 < S2 < S3).
Then label all 12,000 rows with the winner for Jev and Laya (same command without --ids-from/--limit).

## E2 Cutoffs
python3 -m pipeline.evaluate --run e2-full --jev <win> --laya <win> --qwen v2 --composition ... --fulfillment ...
Read cutoff_95 (primary) and cutoff_90 (fulfillment) per source from results/e2-full/metrics.json; coverage at the cutoff is the auto-accept share.

## E3 Human gold and disagreements (needs the CRM review page live)
python3 -m pipeline.review_queue enqueue --run gold-<date> --reason gold_sample --per-vehicle 200 --qwen v2 --jev <win> --laya <win> --token-file ~/.crm-review-token --base https://<crm-host>
python3 -m pipeline.review_queue enqueue --run dis-<date> --reason disagreement --limit 300 --qwen v2 --jev <win> --token-file ... --base ...
python3 -m pipeline.review_queue enqueue --run ful-<date> --reason fulfillment --per-vehicle 70 --qwen v2 --jev <win> --token-file ... --base ...
Disagreement rows go out blind as proposals A/B; the key map is written to gold/blind-dis-<date>.jsonl.
# after reviewers finish:
python3 -m pipeline.review_queue pull --run gold-<date> ...; pull dis-<date>; pull ful-<date>
python3 -m pipeline.evaluate --run e3-human --jev <win> --laya <win> --qwen v2 --composition ... --fulfillment ... --human gold/human-<date>.jsonl

## E4 Fulfillment
Covered by fulfillment_vs_quote_gold and vs_human_gold.fulfillment_accuracy in e2-full and e3-human. Also report the proxy rule's own precision: share of fulfillment-gold rows the humans confirmed (compute from gold/human-<date>.jsonl where reason == "fulfillment").

## E5 Cost and latency
Jev: mean `usage.input_tokens` × $0.042 / 1e6 per opportunity; p50/p95 of latency_ms from labels/jev/<win>.jsonl.
Qwen: `usage.total_tokens` summed from labels/qwen/v2.jsonl and wall time from results/e5-qwen.log
(run the labeler with `2>&1 | tee results/e5-qwen.log`); GPU seconds = wall time × 1 GPU.
Laya: rows/s from label_laya stderr; $/M decisions from ~/<bench-repo> results (bench report).
Record all three in results/e5-cost.md.

## E6 Regression
Any prompt or model change: re-run the labeler into a new run name, then `python3 -m pipeline.evaluate --run <new> ...` and diff metrics.json against the previous run.

## E7 Report
python3 -m pipeline.report --run <run>             # results/<run>/report.html from metrics.json
Cutoffs print as "not achieved" when no cutoff meets the target with enough rows behind it, and
ECE as "n/a" when the source's confidence has fewer than 10 distinct values (Qwen's three buckets).

## Stop rules
- Jev returns 429s: the labeler backs off; if sustained, lower MAX_RPS to 5.
- Any labeler error rate above 2%: stop, inspect the error rows, fix, resume (all labelers resume by id).
- Qwen batches only at ≤ 48 concurrent, preferably outside 08:00–18:00 Phoenix.
- Laya's 512-token context truncates silently; S3 is only partially seen by Laya.
