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
