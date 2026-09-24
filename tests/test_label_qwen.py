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


def test_concurrency_clamped_at_48(monkeypatch):
    monkeypatch.setenv("CONC", "200")
    spec = importlib.util.spec_from_file_location("qc_clamped", pathlib.Path("qwen_discovery/qwen_classify.py"))
    qc_clamped = importlib.util.module_from_spec(spec); spec.loader.exec_module(qc_clamped)
    assert qc_clamped.CONCURRENCY == 48


def test_imports_without_endpoint_configured(monkeypatch):
    """CI has no secrets: the endpoint is read when a request is sent, not at import."""
    monkeypatch.delenv("QWEN_ENDPOINT", raising=False)
    monkeypatch.delenv("PCAI_API_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("qc_noenv", pathlib.Path("qwen_discovery/qwen_classify.py"))
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
