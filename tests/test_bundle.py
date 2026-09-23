from pipeline import bundle
from pipeline.schema import CLASSES, SUBCLASSES, LIFECYCLE, DOMAINS, FULFILLMENT, FLAGS

ROW = {"vehicle": "SEWP", "title": "80 Dell laptops", "rfq_type": "RFQ",
       "description": "Dell Pro Max 16 x80\nRequirements apply for Providers with Established Authorized Reseller Programs (EARP)\nTAA Compliant Products Only",
       "lines": "Dell MC16255 Dell Pro Max 16 x80", "attachment_text": "See SOW section 2"}

def test_variant_a_shape():
    q = bundle.questions("A")
    assert set(q["primary_class"]["criteria"]) == set(CLASSES)
    assert sum(1 for k in q if k.startswith("has_")) == len(SUBCLASSES)
    assert list(q["lifecycle"]["criteria"]) == LIFECYCLE
    assert set(q["domain"]["criteria"]) == set(DOMAINS)
    assert set(q["fulfillment_mode"]["criteria"]) == set(FULFILLMENT)
    for f in FLAGS:
        assert q[f]["type"] == "noul"
    assert all(v["type"] in ("choice", "noul") for v in q.values())

def test_variant_b_uses_single_subclass_choice():
    q = bundle.questions("B")
    assert set(q["primary_subclass"]["criteria"]) == set(SUBCLASSES)
    assert not any(k.startswith("has_") for k in q)

def test_state_variants_strip_boilerplate_and_grow():
    s1, s2, s3 = (bundle.state(ROW, v) for v in ("S1", "S2", "S3"))
    assert "EARP" not in s1 and "TAA Compliant" not in s1
    assert "Dell MC16255" not in s1 and "Dell MC16255" in s2
    assert "SOW section 2" not in s2 and "SOW section 2" in s3
