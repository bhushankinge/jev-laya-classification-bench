from pipeline.review_queue import build_items, stratified_ids
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
