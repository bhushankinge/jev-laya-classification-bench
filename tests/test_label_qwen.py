import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("qc", pathlib.Path("qwen_discovery/qwen_classify.py"))
qc = importlib.util.module_from_spec(spec); spec.loader.exec_module(qc)

def test_schema_has_fulfillment_and_flags():
    props = qc.SCHEMA["properties"]
    assert props["fulfillment_mode"]["enum"] == ["à la carte", "configured build", "mixed", "not applicable"]
    assert props["rfi_market_research"]["type"] == "boolean"
    assert props["text_insufficient"]["type"] == "boolean"
    for k in ("fulfillment_mode", "rfi_market_research", "text_insufficient"):
        assert k in qc.SCHEMA["required"]
    assert "configured build" in qc.SYSTEM
