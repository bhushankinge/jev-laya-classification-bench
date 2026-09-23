from pipeline.label_jev import parse_response

def test_parse_keeps_answers_usage_and_model():
    js = {"model": "jev-1.13.0", "usage": {"input_tokens": 480, "output_tokens": 70},
          "answers": {"primary_class": {"type": "choice", "choice": "Hardware", "confidence": 0.91,
                                        "probabilities": {"Hardware": 0.91, "Software": 0.05}},
                      "brand_name_only": {"type": "noul", "noul": 0.12}}}
    out = parse_response(js)
    assert out["model"] == "jev-1.13.0" and out["usage"]["input_tokens"] == 480
    assert out["answers"]["primary_class"]["choice"] == "Hardware"
    assert out["answers"]["brand_name_only"]["noul"] == 0.12


def test_call_returns_an_error_row_when_answers_are_incomplete(monkeypatch):
    import pipeline.label_jev as lj

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"model": "jev-1", "usage": {}, "answers": {"primary_class": {"choice": "Hardware"}}}

    class Sess:
        def post(self, *a, **k): return Resp()

    monkeypatch.setattr(lj.time, "sleep", lambda *a: None)
    out = lj.call(Sess(), "key", {"id": "x", "vehicle": "SEWP"}, {}, "S1")
    assert out["id"] == "x"
    assert out["error"] == "incomplete answers: lifecycle, domain, fulfillment_mode"
