from pipeline.review_queue import build_items, stratified_ids, blind_map, disagreement_ids
from pipeline.schema import Label

def lab(sub, veh_conf=0.9):
    return Label("Hardware", [sub], "new", "networking", "à la carte",
                 {"rfi_market_research": False, "text_insufficient": False, "brand_name_only": False},
                 {"primary_class": veh_conf}, "qwen")

def test_build_items_shape():
    sample = {"o1": {"id": "o1", "vehicle": "SEWP", "title": "t", "description": "d", "lines": None, "attachment_text": "a"}}
    items = build_items(["o1"], sample, {"qwen": {"o1": lab("General hardware")}}, "gold_sample")
    assert items[0]["opportunityId"] == "o1" and items[0]["reason"] == "gold_sample"
    assert items[0]["textBundle"] == {"vehicle": "SEWP", "title": "t", "description": "d", "lines": None, "attachmentExcerpt": "a"}
    assert items[0]["proposals"]["qwen"]["components"] == ["General hardware"]

def test_stratified_ids_covers_rare_subclasses_and_respects_per_vehicle():
    sample, labels = {}, {}
    for i in range(100):
        sample[f"s{i}"] = {"id": f"s{i}", "vehicle": "SEWP"}
        labels[f"s{i}"] = lab("General hardware" if i < 95 else "Training")
    ids = stratified_ids(labels, sample, per_vehicle=20)
    assert len(ids) == 20
    assert sum(labels[i].components == ["Training"] for i in ids) >= 2

def test_build_items_truncation_preserves_none():
    sample = {"o1": {"id": "o1", "vehicle": "SEWP", "title": "a" * 600, "description": None, "lines": None, "attachment_text": "b" * 6000}}
    items = build_items(["o1"], sample, {"qwen": {"o1": lab("General hardware")}}, "gold_sample")
    assert len(items[0]["textBundle"]["title"]) == 500
    assert items[0]["textBundle"]["description"] is None
    assert items[0]["textBundle"]["lines"] is None
    assert len(items[0]["textBundle"]["attachmentExcerpt"]) == 5000

def test_stratified_ids_deterministic_vehicle_order():
    sample, labels = {}, {}
    for i in range(50):
        sample[f"g{i}"] = {"id": f"g{i}", "vehicle": "GSA 2GIT"}
        sample[f"s{i}"] = {"id": f"s{i}", "vehicle": "SEWP"}
        labels[f"g{i}"] = lab("General hardware")
        labels[f"s{i}"] = lab("General hardware")
    ids1 = stratified_ids(labels, sample, per_vehicle=10, seed=42)
    ids2 = stratified_ids(labels, sample, per_vehicle=10, seed=42)
    assert ids1 == ids2
    # GSA 2GIT < SEWP alphabetically, so GSA ids should appear first
    gsa_indices = [ids1.index(i) for i in ids1 if i.startswith("g")]
    sewp_indices = [ids1.index(i) for i in ids1 if i.startswith("s")]
    assert max(gsa_indices) < min(sewp_indices)

def test_blind_map_is_deterministic_per_seed_and_uses_both_orders():
    ids = [f"d{i}" for i in range(40)]
    m1, m2 = blind_map(ids, ("jev", "qwen"), 7), blind_map(ids, ("jev", "qwen"), 7)
    assert m1 == m2
    assert blind_map(ids, ("jev", "qwen"), 8) != m1
    orders = {(v["A"], v["B"]) for v in m1.values()}
    assert orders == {("jev", "qwen"), ("qwen", "jev")}          # neither model is pinned to a key
    assert all(set(v.values()) == {"jev", "qwen"} for v in m1.values())

def test_build_items_hides_the_source_behind_the_blind_key():
    sample = {"o1": {"id": "o1", "vehicle": "SEWP", "title": "t", "description": "d"}}
    labels = {"jev": {"o1": lab("General hardware")}, "qwen": {"o1": lab("Training")}}
    keymap = {"o1": {"A": "qwen", "B": "jev"}}
    p = build_items(["o1"], sample, labels, "disagreement", keymap)[0]["proposals"]
    assert set(p) == {"A", "B"}
    assert p["A"]["components"] == ["Training"] and p["A"]["source"] == "A"
    assert p["B"]["components"] == ["General hardware"] and p["B"]["source"] == "B"

def test_disagreement_ids_splits_the_limit_between_the_two_strata():
    jev, qwen = {}, {}
    for i in range(50):                       # 50 class disagreements
        jev[f"c{i}"], qwen[f"c{i}"] = lab("General hardware"), Label(
            "Software", ["Subscription/SaaS"], "new", "networking", "not applicable",
            {"rfi_market_research": False, "text_insufficient": False, "brand_name_only": False}, {}, "qwen")
    for i in range(50):                       # 50 component-set-only disagreements
        jev[f"p{i}"], qwen[f"p{i}"] = lab("General hardware"), lab("Consumables/Supplies")
    ids = disagreement_ids(jev, qwen, 20)
    assert len(ids) == 20
    assert sum(i.startswith("c") for i in ids) == 10 and sum(i.startswith("p") for i in ids) == 10
    assert disagreement_ids(jev, qwen, 20) == ids                      # deterministic
    thin = {k: v for k, v in jev.items() if k.startswith("c")}
    assert len(disagreement_ids(thin, {k: qwen[k] for k in thin}, 20)) == 20   # one stratum fills in
