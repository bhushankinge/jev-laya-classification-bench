"""Metrics, calibration, cutoffs, agreement and cost for one run (spec Section 6)."""
import argparse, json, platform, time
from collections import Counter, defaultdict
from . import common
from .mapping import load_labels
from .schema import CLASSES, FULFILLMENT, SUBCLASS_TO_CLASS

TARGETS = {"primary_class": 0.95, "components": 0.90, "fulfillment_mode": 0.90}


def accuracy(pred, gold):
    ids = [i for i in pred if i in gold]
    return sum(pred[i] == gold[i] for i in ids) / len(ids) if ids else 0.0


def per_class_prf(pred, gold, classes):
    out = {}
    ids = [i for i in pred if i in gold]
    for c in classes:
        tp = sum(pred[i] == c and gold[i] == c for i in ids)
        fp = sum(pred[i] == c and gold[i] != c for i in ids)
        fn = sum(pred[i] != c and gold[i] == c for i in ids)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn}
    return out


def ece(confidences, correct, bins=10):
    n = len(confidences)
    if not n:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return round(total, 4)


def precision_coverage_curve(confidences, correct):
    pairs = sorted(zip(confidences, correct), reverse=True)
    n, curve, tp = len(pairs), [], 0
    for k, (c, ok) in enumerate(pairs, 1):
        tp += ok
        if k == n or pairs[k][0] != c:
            curve.append({"cutoff": c, "precision": tp / k, "coverage": k / n})
    return curve


def cutoff_for(curve, target):
    ok = [p for p in curve if p["precision"] >= target]
    return min(ok, key=lambda p: p["cutoff"])["cutoff"] if ok else None


def agreement_matrix(a, b):
    m = defaultdict(Counter)
    for i in a:
        if i in b:
            m[a[i]][b[i]] += 1
    return {k: dict(v) for k, v in m.items()}


def _gold_classes(path):
    return {r["id"]: r for r in common.read_jsonl(path)}


def SUB2CLS(components):
    return [SUBCLASS_TO_CLASS[c] for c in components]


def evaluate(run, sources: dict, composition_path, fulfillment_path, human_path=None):
    labels = {s: load_labels(s, r) for s, r in sources.items()}
    comp = _gold_classes(composition_path)
    ful = {r["id"]: r["fulfillment_mode"] for r in common.read_jsonl(fulfillment_path)}
    human = {r["id"]: r["verdict"] for r in common.read_jsonl(human_path)} if human_path else {}
    out = {"run": run, "sources": sources, "n_labeled": {s: len(l) for s, l in labels.items()}, "by_source": {}}
    for s, labs in labels.items():
        pri = {i: l.primary_class for i, l in labs.items()}
        gold_pri = {i: (r["classes"][0] if len(r["classes"]) == 1 else None) for i, r in comp.items()}
        gold_pri = {i: g for i, g in gold_pri.items() if g}      # single-class quotes give an unambiguous primary
        comp_set_ok = {i: set(SUB2CLS(labs[i].components)) == set(comp[i]["classes"]) for i in labs if i in comp}
        conf_pri = [labs[i].confidence.get("primary_class", 0.0) for i in pri if i in gold_pri]
        corr_pri = [pri[i] == gold_pri[i] for i in pri if i in gold_pri]
        curve = precision_coverage_curve(conf_pri, corr_pri)
        ful_pred = {i: l.fulfillment_mode for i, l in labs.items() if i in ful and l.fulfillment_mode != "not applicable"}
        ful_conf = [labs[i].confidence.get("fulfillment_mode", 0.0) for i in ful_pred]
        ful_corr = [ful_pred[i] == ful[i] for i in ful_pred]
        ful_curve = precision_coverage_curve(ful_conf, ful_corr)
        res = {
            "primary_vs_quote_gold": {"n": len(corr_pri), "accuracy": accuracy(pri, gold_pri),
                                      "per_class": per_class_prf(pri, gold_pri, CLASSES),
                                      "ece": ece(conf_pri, corr_pri),
                                      "cutoff_95": cutoff_for(curve, TARGETS["primary_class"]),
                                      "curve": curve[:: max(1, len(curve) // 50)]},
            "component_set_exact_vs_quote_gold": {"n": len(comp_set_ok), "accuracy": sum(comp_set_ok.values()) / len(comp_set_ok) if comp_set_ok else 0.0},
            "fulfillment_vs_quote_gold": {"n": len(ful_corr), "accuracy": sum(ful_corr) / len(ful_corr) if ful_corr else 0.0,
                                          "per_class": per_class_prf(ful_pred, ful, FULFILLMENT),
                                          "cutoff_90": cutoff_for(ful_curve, TARGETS["fulfillment_mode"])},
            "flags_rate": {f: sum(l.flags[f] for l in labs.values()) / len(labs) for f in ("rfi_market_research", "text_insufficient", "brand_name_only")} if labs else {},
        }
        if human:
            hp = {i: v["primary_class"] for i, v in human.items()}
            hs = {i: set(v["components"]) for i, v in human.items()}
            res["vs_human_gold"] = {
                "n": len([i for i in labs if i in human]),
                "primary_accuracy": accuracy(pri, hp),
                "subclass_set_exact": sum(set(labs[i].components) == hs[i] for i in labs if i in hs) / max(1, len([i for i in labs if i in hs])),
                "lifecycle_accuracy": accuracy({i: l.lifecycle for i, l in labs.items()}, {i: v["lifecycle"] for i, v in human.items()}),
                "fulfillment_accuracy": accuracy({i: l.fulfillment_mode for i, l in labs.items()}, {i: v["fulfillment_mode"] for i, v in human.items()}),
            }
        out["by_source"][s] = res
    if "jev" in labels and "qwen" in labels:
        out["jev_vs_qwen_primary"] = agreement_matrix({i: l.primary_class for i, l in labels["jev"].items()},
                                                      {i: l.primary_class for i, l in labels["qwen"].items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jev"); ap.add_argument("--laya"); ap.add_argument("--qwen")
    ap.add_argument("--composition", required=True); ap.add_argument("--fulfillment", required=True)
    ap.add_argument("--human")
    a = ap.parse_args()
    sources = {k: v for k, v in (("jev", a.jev), ("laya", a.laya), ("qwen", a.qwen)) if v}
    res = evaluate(a.run, sources, a.composition, a.fulfillment, a.human)
    d = common.RESULTS_DIR / a.run
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
    (d / "env.json").write_text(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(),
                                            "sources": sources, "gold": [a.composition, a.fulfillment, a.human]}, indent=1))
    for s, r in res["by_source"].items():
        print(s, "primary acc", round(r["primary_vs_quote_gold"]["accuracy"], 3), "cutoff95", r["primary_vs_quote_gold"]["cutoff_95"],
              "fulfillment acc", round(r["fulfillment_vs_quote_gold"]["accuracy"], 3))
