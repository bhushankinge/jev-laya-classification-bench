from pipeline.evaluate import accuracy, per_class_prf, ece, precision_coverage_curve, cutoff_for, agreement_matrix

def test_accuracy_and_prf():
    pred = {"1": "Hardware", "2": "Software", "3": "Hardware"}
    gold = {"1": "Hardware", "2": "Hardware", "3": "Hardware", "4": "Software"}
    assert accuracy(pred, gold) == 2 / 3          # only ids in both
    prf = per_class_prf(pred, gold, ["Hardware", "Software"])
    assert prf["Hardware"]["precision"] == 1.0 and prf["Hardware"]["recall"] == 2 / 3

def test_ece_is_zero_when_perfectly_calibrated():
    conf = [0.95] * 20
    correct = [True] * 19 + [False]
    assert ece(conf, correct, bins=10) == 0.0

def test_cutoff_for_target_precision():
    conf = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6]
    correct = [True, True, True, False, True, False]
    curve = precision_coverage_curve(conf, correct)
    assert cutoff_for(curve, 0.95) == 0.9
    assert round(next(p["coverage"] for p in curve if p["cutoff"] == 0.9), 2) == 0.5
    assert cutoff_for(curve, 1.01) is None

def test_agreement_matrix_counts_pairs():
    a = {"1": "Hardware", "2": "Software"}
    b = {"1": "Hardware", "2": "Hardware"}
    m = agreement_matrix(a, b)
    assert m["Hardware"]["Hardware"] == 1 and m["Software"]["Hardware"] == 1
