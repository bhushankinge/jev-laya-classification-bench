from pipeline.schema import (CLASSES, SUBCLASSES, LIFECYCLE, DOMAINS, FULFILLMENT, FLAGS,
                             SUBCLASS_TO_CLASS, Label, map_qwen)

def test_vocabulary_sizes():
    assert len(CLASSES) == 7 and len(SUBCLASSES) == 12 and len(LIFECYCLE) == 5
    assert len(DOMAINS) == 15 and len(FULFILLMENT) == 4 and len(FLAGS) == 3
    assert set(SUBCLASS_TO_CLASS.values()) <= set(CLASSES)

def test_map_qwen_hardware_and_support():
    raw = {"components": [
        {"what": "switches", "kind": "physical product", "lifecycle": "new purchase", "domain": "networking", "brand": "Cisco"},
        {"what": "SmartNet", "kind": "support/maintenance contract", "lifecycle": "renewal", "domain": "networking", "brand": "Cisco"}],
        "primary_index": 0, "brand_name_only": True, "confidence": "high",
        "fulfillment_mode": "configured build", "rfi_market_research": False, "text_insufficient": False}
    lab = map_qwen(raw)
    assert lab.primary_class == "Hardware"
    assert lab.components == ["General hardware", "Support/Maintenance contract"]
    assert lab.lifecycle == "new" and lab.domain == "networking"
    assert lab.fulfillment_mode == "configured build"
    assert lab.flags == {"rfi_market_research": False, "text_insufficient": False, "brand_name_only": True}
    assert lab.confidence["primary_class"] == 0.9 and lab.source == "qwen"

def test_map_qwen_without_hardware_is_not_applicable():
    raw = {"components": [{"what": "Adobe", "kind": "software subscription/SaaS", "lifecycle": "renewal",
                           "domain": "business/productivity software", "brand": "Adobe"}],
           "primary_index": 0, "brand_name_only": False, "confidence": "low"}
    lab = map_qwen(raw)
    assert lab.primary_class == "Software" and lab.components == ["Subscription/SaaS"]
    assert lab.lifecycle == "renewal" and lab.fulfillment_mode == "not applicable"
    assert lab.confidence["primary_class"] == 0.5
