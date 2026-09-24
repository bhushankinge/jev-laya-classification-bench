"""Laya breakdown tables (report Section 9.4) for one or more Laya label runs, on the paired quote-gold rows:
primary accuracy by S2 length bucket and by vehicle, calibration, has_* noul fire rates, flag rates,
lifecycle counts, components per row, latency.

    python3 -m pipeline.laya_breakdown --runs A-S2 A-S2-head-bf16 --jev A-S2 --qwen v2 \
        --composition gold/composition-<date>.jsonl [--per-case-dir results/e2-laya-head/per_case]
"""
import argparse, json, statistics
from collections import Counter
from . import common, bundle
from .evaluate import collapse_to_quote_vocab, ece
from .mapping import load_labels
from .schema import SUBCLASSES

BUCKETS = [(0, 600), (600, 1200), (1200, 2000), (2000, 3000), (3000, 10**9)]


def main(runs, jev, qwen, composition, per_case_dir=None):
    comp = {r["id"]: r for r in common.read_jsonl(composition)}
    gold = {i: r["classes"][0] for i, r in comp.items() if len(r["classes"]) == 1}
    sample = {r["id"]: r for r in common.read_jsonl(common.SAMPLE)}
    length = {i: len(bundle.state(r, "S2")) for i, r in sample.items()}
    labels = {r: load_labels("laya", r) for r in runs}
    refs = {}
    if jev:
        refs["jev"] = load_labels("jev", jev)
    if qwen:
        refs["qwen"] = load_labels("qwen", qwen)
    paired = set(gold) & set.intersection(*(set(l) for l in list(labels.values()) + list(refs.values())))
    print(f"paired single-class quote-gold rows: {len(paired)}")
    raw = {r: {row["id"]: row for row in common.latest_by_id(common.LABELS_DIR / "laya" / f"{r}.jsonl").values()
               if "answers" in row} for r in runs}
    if per_case_dir:                            # id-free rows: nothing in them identifies a notice or a quote
        d = common.Path(per_case_dir); d.mkdir(parents=True, exist_ok=True)
        for r in runs:
            with open(d / f"{r}.jsonl", "w") as f:
                for i in sorted(paired, key=lambda i: (sample[i]["vehicle"], length[i])):
                    a = raw[r][i]["answers"]["primary_class"]
                    f.write(json.dumps({"vehicle": sample[i]["vehicle"], "s2_chars": length[i], "gold": gold[i],
                                        "pred": a["choice"], "confidence": a["confidence"],
                                        "probabilities": a["probabilities"], "laya": raw[r][i].get("laya", "0.3.3"),
                                        "dtype": raw[r][i].get("dtype", "torch.bfloat16")}) + "\n")
    cols = list(refs) + runs
    allsrc = {**refs, **labels}

    def acc(src, ids):
        ids = [i for i in ids if i in allsrc[src]]
        return sum(collapse_to_quote_vocab(allsrc[src][i].primary_class) == gold[i] for i in ids) / len(ids) if ids else float("nan")

    print("\n## primary accuracy, ECE, hardware->software confusions (paired rows)\n")
    print("| source | n | accuracy | ECE | median conf | HW called Software |")
    print("|---|---|---|---|---|---|")
    for s in cols:
        ids = sorted(paired)
        conf = [allsrc[s][i].confidence.get("primary_class", 0.0) for i in ids]
        corr = [collapse_to_quote_vocab(allsrc[s][i].primary_class) == gold[i] for i in ids]
        hw_sw = sum(gold[i] == "Hardware" and allsrc[s][i].primary_class == "Software" for i in ids)
        e = ece(conf, corr) if len(set(conf)) >= 10 else "n/a"
        print(f"| {s} | {len(ids)} | {acc(s, ids):.3f} | {e} | {statistics.median(conf):.2f} | {hw_sw}/{sum(gold[i] == 'Hardware' for i in ids)} |")

    print("\n## primary accuracy by S2 state length (characters)\n")
    print("| S2 length | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for lo, hi in BUCKETS:
        ids = [i for i in paired if lo <= length[i] < hi]
        if ids:
            lab = f"{lo:,} to {hi:,}" if hi < 10**9 else f"{lo:,} and over"
            print(f"| {lab} | {len(ids)} | " + " | ".join(f"{acc(s, ids):.3f}" for s in cols) + " |")

    print("\n## primary accuracy by vehicle\n")
    print("| vehicle | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for v in sorted({sample[i]["vehicle"] for i in paired}):
        ids = [i for i in paired if sample[i]["vehicle"] == v]
        print(f"| {v} | {len(ids)} | " + " | ".join(f"{acc(s, ids):.3f}" for s in cols) + " |")

    print("\n## share of rows with the has_* noul >= 0.5, and mean noul (all labeled rows)\n")
    print("| subclass | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for sub in SUBCLASSES:
        cells = []
        for s in cols:
            if s in refs:
                labs = refs[s]
                cells.append(f"{sum(sub in l.components for l in labs.values()) / len(labs):.3f}")
            else:
                ps = [float(row["answers"][bundle.HAS_KEY[sub]]["noul"]) for row in raw[s].values()]
                cells.append(f"{sum(p >= 0.5 for p in ps) / len(ps):.3f} (mean {statistics.fmean(ps):.2f})")
        print(f"| {sub} | " + " | ".join(cells) + " |")

    print("\n## flags (share of rows >= 0.5), lifecycle, components per row, latency (all labeled rows)\n")
    print("| source | n | rfi_market_research | text_insufficient | brand_name_only | renewal | replacement/refresh | unknown | components/row | p50 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in cols:
        labs = allsrc[s]
        n = len(labs)
        fl = [f"{sum(l.flags[f] for l in labs.values()) / n:.2f}" for f in ("rfi_market_research", "text_insufficient", "brand_name_only")]
        lc = Counter(l.lifecycle for l in labs.values())
        comps = statistics.fmean(len(l.components) for l in labs.values())
        lat = statistics.median(r["latency_ms"] for r in raw[s].values()) if s in raw else float("nan")
        print(f"| {s} | {n} | " + " | ".join(fl) + f" | {lc['renewal']} | {lc['replacement/refresh']} | {lc['unknown']} | {comps:.2f} | {lat:.0f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="laya label runs, e.g. A-S2 A-S2-head-bf16")
    ap.add_argument("--jev"); ap.add_argument("--qwen")
    ap.add_argument("--composition", required=True)
    ap.add_argument("--per-case-dir", help="write id-free per-case rows (vehicle, length, gold, prediction, probabilities) per run")
    a = ap.parse_args()
    main(a.runs, a.jev, a.qwen, a.composition, a.per_case_dir)
