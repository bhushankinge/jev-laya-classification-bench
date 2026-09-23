from pipeline.evaluate import (accuracy, per_class_prf, ece, precision_coverage_curve, cutoff_for,
                               agreement_matrix, wilson_lower, gate, collapse_to_quote_vocab,
                               subclass_presence_curve)
from pipeline.schema import Label

def test_accuracy_and_prf():
    pred = {"1": "Hardware", "2": "Software", "3": "Hardware"}
    gold = {"1": "Hardware", "2": "Hardware", "3": "Hardware", "4": "Software"}
    assert accuracy(pred, gold) == 2 / 3          # only ids in both
    assert accuracy(pred, {}) is None             # nothing to score is not 0%
    prf = per_class_prf(pred, gold, ["Hardware", "Software"])
    assert prf["Hardware"]["precision"] == 1.0 and prf["Hardware"]["recall"] == 2 / 3

def test_ece_is_zero_when_perfectly_calibrated():
    conf = [0.95] * 20
    correct = [True] * 19 + [False]
    assert ece(conf, correct, bins=10) == 0.0

def test_wilson_lower_bound():
    assert round(wilson_lower(52, 52), 2) == 0.93
    assert round(wilson_lower(10, 10), 2) == 0.72
    assert wilson_lower(0, 0) == 0.0

def test_cutoff_needs_enough_evidence_and_a_clean_prefix():
    conf = [round(1 - i / 1000, 3) for i in range(120)]      # 120 distinct confidences, descending
    curve = precision_coverage_curve(conf, [True] * 100 + [False] * 20)
    c = cutoff_for(curve, 0.90)
    pt = curve[[p["cutoff"] for p in curve].index(c)]
    nxt = curve[curve.index(pt) + 1]
    assert wilson_lower(round(pt["precision"] * pt["n"]), pt["n"]) >= 0.90     # holds here
    assert wilson_lower(round(nxt["precision"] * nxt["n"]), nxt["n"]) < 0.90   # and stops right after
    # a miss at the very top breaks the envelope, however good the rest is
    assert cutoff_for(precision_coverage_curve(conf, [False] + [True] * 119), 0.90) is None
    # three perfect rows are not evidence of 95% precision
    small = precision_coverage_curve([0.99, 0.95, 0.9, 0.8], [True, True, True, False])
    assert cutoff_for(small, 0.95) is None

def test_agreement_matrix_counts_pairs():
    a = {"1": "Hardware", "2": "Software"}
    b = {"1": "Hardware", "2": "Hardware"}
    m = agreement_matrix(a, b)
    assert m["Hardware"]["Hardware"] == 1 and m["Software"]["Hardware"] == 1

def test_collapse_to_quote_vocab():
    assert collapse_to_quote_vocab("Installation & Integration") == "Services"
    assert collapse_to_quote_vocab("Furniture / Facilities") == "Other"
    assert collapse_to_quote_vocab("Hardware") == "Hardware"


def lab(cls, conf=0.99, flag=False, comps=("General hardware",), ful="à la carte"):
    return Label(cls, list(comps), "new", "networking", ful,
                 {"rfi_market_research": flag, "text_insufficient": False, "brand_name_only": False},
                 {"primary_class": conf, "fulfillment_mode": conf}, "jev")

def test_gate_routes_each_row_and_scores_only_the_accepted():
    rows = {
        "a": {"jev": lab("Hardware"), "qwen": None, "gold_primary": "Hardware", "gold_fulfillment": "à la carte"},
        "b": {"jev": lab("Hardware", conf=0.4), "qwen": lab("Hardware"), "gold_primary": "Software", "gold_fulfillment": None},
        "c": {"jev": lab("Hardware", conf=0.4), "qwen": lab("Software"), "gold_primary": "Hardware", "gold_fulfillment": None},
        "d": {"jev": lab("Hardware", conf=0.4), "qwen": None, "gold_primary": "Hardware", "gold_fulfillment": None},
        "e": {"jev": lab("Hardware", flag=True), "qwen": lab("Hardware"), "gold_primary": "Hardware", "gold_fulfillment": None},
        "f": {"jev": lab("Other"), "qwen": lab("Other"), "gold_primary": "Other", "gold_fulfillment": None},
    }
    g = gate(rows, {"primary_class": 0.9})
    assert (g["n_accepted"], g["n_queued"]) == (2, 4)                       # a (confidence), b (qwen agrees)
    assert (g["queued_by_flag"], g["queued_by_disagreement"], g["queued_by_confidence"]) == (2, 1, 1)
    assert g["coverage"] == 2 / 6 and g["precision_primary"] == 0.5         # b is accepted and wrong
    assert g["precision_fulfillment"] == 1.0 and g["n_fulfillment_scored"] == 1
    loose = gate(rows, {"primary_class": 0.9}, flags_always_queue=False)
    assert loose["n_accepted"] == 3 and loose["queued_by_flag"] == 1        # e accepted, "Other" still queues

def test_gate_without_a_cutoff_falls_back_to_qwen_agreement():
    rows = {"a": {"jev": lab("Hardware"), "qwen": lab("Hardware"), "gold_primary": None, "gold_fulfillment": None},
            "b": {"jev": lab("Hardware"), "qwen": None, "gold_primary": None, "gold_fulfillment": None}}
    g = gate(rows, {"primary_class": None})
    assert g["n_accepted"] == 1 and g["queued_by_confidence"] == 1 and g["precision_primary"] is None

def test_subclass_presence_curve_scores_every_subclass():
    labels = {"a": lab("Hardware", comps=["General hardware", "Warranty"]),
              "b": lab("Hardware", comps=["General hardware"])}
    res = subclass_presence_curve(labels, {"a": ["General hardware"], "b": ["General hardware"]})
    assert res["n_calls"] == 24                                     # 12 subclasses x 2 opportunities
    assert res["per_subclass"]["General hardware"]["recall"] == 1.0
    assert res["per_subclass"]["Warranty"]["precision"] == 0.0      # one false positive, no support
    assert res["n_distinct_confidences"] == 1 and res["cutoff_90_components"] is None
