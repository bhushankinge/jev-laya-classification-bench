"""Metrics, calibration, cutoffs, gating and agreement for one run (spec Section 6)."""
import argparse, json, platform, time
from collections import Counter, defaultdict
from math import sqrt
from . import common
from .bundle import HAS_KEY
from .mapping import load_labels
from .schema import CLASSES, FULFILLMENT, SUBCLASSES, SUBCLASS_TO_CLASS

TARGETS = {"primary_class": 0.95, "components": 0.90, "fulfillment_mode": 0.90}
# Quote gold only knows the five product-type classes, so the two classes it cannot express are
# folded into their nearest quote class before scoring; they are measured against human gold only.
COLLAPSE = {"Installation & Integration": "Services", "Furniture / Facilities": "Other"}
QUOTE_CLASSES = [c for c in CLASSES if c not in COLLAPSE]
QUOTE_NOTE = ("predictions are collapsed to the quote-gold vocabulary before scoring: "
              "'Installation & Integration' -> 'Services', 'Furniture / Facilities' -> 'Other'. "
              "Those two classes are measured against human gold only.")
MIN_DISTINCT_CONF = 10
DISCRETE_NOTE = "confidence not continuous"


def accuracy(pred, gold):
    ids = [i for i in pred if i in gold]
    return sum(pred[i] == gold[i] for i in ids) / len(ids) if ids else None


def collapse_to_quote_vocab(cls: str) -> str:
    return COLLAPSE.get(cls, cls)


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
            curve.append({"cutoff": c, "precision": tp / k, "coverage": k / n, "n": k})
    return curve


def wilson_lower(k, n, z=1.96):
    """Lower end of the Wilson 95% interval for k successes in n. 52/52 -> 0.93, 10/10 -> 0.72."""
    if not n:
        return 0.0
    p = k / n
    return (p + z * z / (2 * n) - z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def cutoff_for(curve, target):
    """The deepest cutoff whose Wilson-bounded precision clears the target there and at every
    higher cutoff. A lucky bucket can no longer buy a cutoff the prefix above it does not support.

    Points too small for any bound to reach the target (a perfect 3/3 cannot show 95% precision)
    are skipped rather than treated as failures: the curve is cumulative, so the rows in them are
    still counted by every point below, where n is large enough to mean something."""
    best = None
    for p in curve:                                     # the curve runs from the highest cutoff down
        if wilson_lower(p["n"], p["n"]) < target:
            continue
        if wilson_lower(round(p["precision"] * p["n"]), p["n"]) < target:
            break
        best = p["cutoff"]
    return best


def _curve_block(confidences, correct, target, cutoff_key):
    """Curve plus calibration. Near-discrete confidence (Qwen's three buckets) cannot support a
    cutoff or an ECE, so those come back None with a note instead of a number that reads as real."""
    curve = precision_coverage_curve(confidences, correct)
    distinct = len(set(confidences))
    blk = {"n_distinct_confidences": distinct}
    if distinct < MIN_DISTINCT_CONF:
        blk.update({"ece": None, cutoff_key: None, "n_at_cutoff": None,
                    "coverage_at_cutoff": None, "note": DISCRETE_NOTE})
    else:
        c = cutoff_for(curve, target)
        pt = next((p for p in curve if p["cutoff"] == c), {})
        blk.update({"ece": ece(confidences, correct), cutoff_key: c,
                    "n_at_cutoff": pt.get("n"), "coverage_at_cutoff": pt.get("coverage")})
    blk["curve"] = curve[:: max(1, len(curve) // 50)]
    return blk


def subclass_presence_curve(labels, human_gold):
    """Subclass presence scored as 12 yes/no calls per opportunity against human gold
    (human_gold: {id: components}). All calls pool into one precision/coverage curve."""
    confidences, correct = [], []
    stat = {s: Counter() for s in SUBCLASSES}
    for i, lab in labels.items():
        if i not in human_gold:
            continue
        gold = set(human_gold[i])
        fallback = max([lab.confidence[HAS_KEY[c]] for c in lab.components if HAS_KEY[c] in lab.confidence]
                       or [lab.confidence.get("components", 0.0)])
        for s in SUBCLASSES:
            pred, truth = s in lab.components, s in gold
            confidences.append(lab.confidence.get(HAS_KEY[s], fallback))
            correct.append(pred == truth)
            stat[s]["tp" if pred and truth else "fp" if pred else "fn" if truth else "tn"] += 1
    out = _curve_block(confidences, correct, TARGETS["components"], "cutoff_90_components")
    out["n_calls"] = len(confidences)
    out["per_subclass"] = {}
    for s, c in stat.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        out["per_subclass"][s] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                                  "support": c["tp"] + c["fn"]}
    return out


def gate(rows, cutoffs, flags_always_queue=True):
    """Auto-accept policy for the Jev label, measured on rows
    {id: {"jev": Label, "qwen": Label|None, "gold_primary": str|None, "gold_fulfillment": str|None}}.
    Queue anything flagged or called "Other"; otherwise accept on confidence, else accept when Qwen
    agrees on the class, else queue. flags_always_queue=False drops the flag rule ("Other" still queues)."""
    cut = cutoffs.get("primary_class") if isinstance(cutoffs, dict) else cutoffs
    queued, accepted = Counter(), []
    for r in rows.values():
        jev = r["jev"]
        if (flags_always_queue and any(jev.flags.values())) or jev.primary_class == "Other":
            queued["flag"] += 1
        elif cut is not None and jev.confidence.get("primary_class", 0.0) >= cut:
            accepted.append(r)
        elif r.get("qwen") is not None and r["qwen"].primary_class == jev.primary_class:
            accepted.append(r)
        else:
            queued["disagreement" if r.get("qwen") is not None else "confidence"] += 1
    pri = [collapse_to_quote_vocab(r["jev"].primary_class) == r["gold_primary"] for r in accepted if r.get("gold_primary")]
    ful = [r["jev"].fulfillment_mode == r["gold_fulfillment"] for r in accepted if r.get("gold_fulfillment")]
    return {"coverage": len(accepted) / len(rows) if rows else None,
            "n_accepted": len(accepted), "n_queued": len(rows) - len(accepted),
            "precision_primary": sum(pri) / len(pri) if pri else None, "n_primary_scored": len(pri),
            "precision_fulfillment": sum(ful) / len(ful) if ful else None, "n_fulfillment_scored": len(ful),
            "queued_by_flag": queued["flag"], "queued_by_confidence": queued["confidence"],
            "queued_by_disagreement": queued["disagreement"]}


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


def _quote_metrics(labs, ids, comp, gold_pri, ful):
    """Every quote-gold metric for one source over one id set (all of its ids, or the paired set)."""
    labs = {i: l for i, l in labs.items() if i in ids}
    pri = {i: collapse_to_quote_vocab(l.primary_class) for i, l in labs.items()}
    cls_ok = {i: {collapse_to_quote_vocab(c) for c in SUB2CLS(labs[i].components)} == set(comp[i]["classes"])
              for i in labs if i in comp}
    conf_pri = [labs[i].confidence.get("primary_class", 0.0) for i in pri if i in gold_pri]
    corr_pri = [pri[i] == gold_pri[i] for i in pri if i in gold_pri]
    # a gold mode with a "not applicable" prediction is a wrong answer, not an excused one
    ful_pred = {i: l.fulfillment_mode for i, l in labs.items() if i in ful}
    ful_conf = [labs[i].confidence.get("fulfillment_mode", 0.0) for i in ful_pred]
    ful_corr = [ful_pred[i] == ful[i] for i in ful_pred]
    return {
        "n_ids": len(labs),
        "primary_vs_quote_gold": {"n": len(corr_pri), "accuracy": accuracy(pri, gold_pri),
                                  "per_class": per_class_prf(pri, gold_pri, QUOTE_CLASSES),
                                  **_curve_block(conf_pri, corr_pri, TARGETS["primary_class"], "cutoff_95")},
        "class_set_exact_vs_quote_gold": {"n": len(cls_ok),
                                          "accuracy": sum(cls_ok.values()) / len(cls_ok) if cls_ok else None},
        "fulfillment_vs_quote_gold": {"n": len(ful_corr),
                                      "accuracy": sum(ful_corr) / len(ful_corr) if ful_corr else None,
                                      "per_class": per_class_prf(ful_pred, ful, FULFILLMENT),
                                      **_curve_block(ful_conf, ful_corr, TARGETS["fulfillment_mode"], "cutoff_90")},
    }


def evaluate(run, sources: dict, composition_path, fulfillment_path, human_path=None):
    labels = {s: load_labels(s, r) for s, r in sources.items()}
    comp = _gold_classes(composition_path)
    ful = {r["id"]: r["fulfillment_mode"] for r in common.read_jsonl(fulfillment_path)}
    human = {r["id"]: r["verdict"] for r in common.read_jsonl(human_path)} if human_path else {}
    gold_pri = {i: r["classes"][0] for i, r in comp.items() if len(r["classes"]) == 1}  # single-class quote = unambiguous primary
    paired = set.intersection(*(set(l) for l in labels.values())) if labels else set()
    out = {"run": run, "sources": sources, "note": QUOTE_NOTE,
           "n_labeled": {s: len(l) for s, l in labels.items()}, "n_paired_ids": len(paired), "by_source": {}}
    for s, labs in labels.items():
        res = {"full": _quote_metrics(labs, set(labs), comp, gold_pri, ful),
               "flags_rate": {f: sum(l.flags[f] for l in labs.values()) / len(labs)
                              for f in ("rfi_market_research", "text_insufficient", "brand_name_only")} if labs else {}}
        if len(labels) > 1:
            res["paired"] = _quote_metrics(labs, paired, comp, gold_pri, ful)
        if human:
            hp = {i: v["primary_class"] for i, v in human.items()}
            hs = {i: set(v["components"]) for i, v in human.items()}
            shared = [i for i in labs if i in hs]
            res["vs_human_gold"] = {
                "n": len(shared),
                "primary_accuracy_stratified": accuracy({i: l.primary_class for i, l in labs.items()}, hp),
                "subclass_set_exact_stratified": (sum(set(labs[i].components) == hs[i] for i in shared) / len(shared)
                                                  if shared else None),
                "lifecycle_accuracy_stratified": accuracy({i: l.lifecycle for i, l in labs.items()},
                                                          {i: v["lifecycle"] for i, v in human.items()}),
                "fulfillment_accuracy_stratified": accuracy({i: l.fulfillment_mode for i, l in labs.items()},
                                                            {i: v["fulfillment_mode"] for i, v in human.items()}),
                "subclass_presence_stratified": subclass_presence_curve(labs, {i: v["components"] for i, v in human.items()}),
            }
        out["by_source"][s] = res
    if "jev" in labels and "qwen" in labels:
        out["jev_vs_qwen_primary"] = agreement_matrix({i: l.primary_class for i, l in labels["jev"].items()},
                                                      {i: l.primary_class for i, l in labels["qwen"].items()})
    if "jev" in labels:
        cut = out["by_source"]["jev"]["full"]["primary_vs_quote_gold"]["cutoff_95"]
        rows = {i: {"jev": l, "qwen": labels.get("qwen", {}).get(i),
                    "gold_primary": gold_pri.get(i), "gold_fulfillment": ful.get(i)}
                for i, l in labels["jev"].items()}
        out["gate"] = {"cutoff": cut,
                       "flags_always_queue": gate(rows, {"primary_class": cut}, True),
                       "flags_not_queued": gate(rows, {"primary_class": cut}, False)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jev"); ap.add_argument("--laya"); ap.add_argument("--qwen")
    ap.add_argument("--composition", required=True); ap.add_argument("--fulfillment", required=True)
    ap.add_argument("--human")
    a = ap.parse_args()
    sources = {k: v for k, v in (("jev", a.jev), ("laya", a.laya), ("qwen", a.qwen)) if v}
    if not sources:
        ap.error("give at least one of --jev / --laya / --qwen")
    res = evaluate(a.run, sources, a.composition, a.fulfillment, a.human)
    d = common.RESULTS_DIR / a.run
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
    (d / "env.json").write_text(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(),
                                            "sources": sources, "gold": [a.composition, a.fulfillment, a.human]}, indent=1))
    for s, r in res["by_source"].items():
        pq = r["full"]["primary_vs_quote_gold"]
        print(s, "primary acc", pq["accuracy"], "cutoff95", pq["cutoff_95"],
              "fulfillment acc", r["full"]["fulfillment_vs_quote_gold"]["accuracy"])
