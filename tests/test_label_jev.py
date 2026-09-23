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
