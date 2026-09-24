# E5 Cost and latency (2026-09-23)

All three sources over the 12,000-row sample, state variant S2 (title + description + line items), question variant A for Jev and Laya.

| Source | Where it ran | Rows | Errors | Tokens per opportunity | Cost | Latency / throughput |
|---|---|---|---|---|---|---|
| Jev (jev-1.13.0) | TypeSafe API, 10 req/s cap, 8 concurrent | 12,000 | 0 | 1,555 input, 669 output | $0.000065 per opportunity at $0.042/M input; $0.78 for the sample | p50 185 ms, p95 273 ms |
| Qwen3.5-35B v2 | PCAI endpoint, 16 concurrent (business hours) | 11,931 | 69 permanently malformed JSON (0.6%) | 1,002 total | GPU time: 80 min wall x 1 GPU, incl. one retry pass | 2.7 opportunities/s, 2,700 tok/s |
| Laya | local RTX 2000 Ada (8 GB), batch 1 | 12,000 | 0 | n/a (512-token context, truncates silently) | see ~/<bench-repo> bench report for $/M decisions | p50 299 ms, p95 576 ms, 2.9 rows/s (~64 min for 11,073 rows) |

Notes
- Jev's per-request tokens include the ~19-question bundle; the state text itself is ~600 tokens.
- Qwen at 48 concurrent (off-hours) is ~3x faster; 16 was used to stay light on PCAI during the day.
- Laya latency is single-row, unbatched; the bench harness measures batched server throughput.
