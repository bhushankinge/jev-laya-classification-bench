import json
import tempfile
from pathlib import Path
import pytest

from pipeline.mapping import map_typed, load_labels
from pipeline import common


def choice(c, conf, probs=None):
    return {"type": "choice", "choice": c, "confidence": conf, "probabilities": probs or {c: conf}}


def noul(p):
    return {"type": "noul", "noul": p}


def test_variant_a_maps_components_from_nouls_and_gates_fulfillment():
    ans = {"primary_class": choice("Hardware", 0.93),
           "has_general_hardware": noul(0.95), "has_support_maintenance_contract": noul(0.71),
           "has_subscription_saas": noul(0.2), "lifecycle": choice("new", 0.8), "domain": choice("networking", 0.7),
           "fulfillment_mode": choice("configured build", 0.66),
           "rfi_market_research": noul(0.02), "text_insufficient": noul(0.1), "brand_name_only": noul(0.9)}
    lab = map_typed(ans, "A", "jev")
    assert lab.primary_class == "Hardware"
    assert lab.components == ["General hardware", "Support/Maintenance contract"]
    assert lab.fulfillment_mode == "configured build" and lab.flags["brand_name_only"] is True
    assert lab.confidence["primary_class"] == 0.93 and lab.confidence["has_general_hardware"] == 0.95
    assert lab.source == "jev"


def test_no_hardware_forces_not_applicable_and_empty_components_fall_back_to_primary():
    ans = {"primary_class": choice("Software", 0.8), "has_subscription_saas": noul(0.4),
           "lifecycle": choice("renewal", 0.9), "domain": choice("cybersecurity", 0.6),
           "fulfillment_mode": choice("à la carte", 0.5),
           "rfi_market_research": noul(0.1), "text_insufficient": noul(0.1), "brand_name_only": noul(0.1)}
    lab = map_typed(ans, "A", "laya")
    assert lab.components == ["Subscription/SaaS"]   # fallback: primary class's first subclass
    assert lab.fulfillment_mode == "not applicable"


def test_variant_b_single_subclass():
    ans = {"primary_class": choice("Services", 0.7), "primary_subclass": choice("Training", 0.65),
           "lifecycle": choice("new", 0.7), "domain": choice("other", 0.5),
           "fulfillment_mode": choice("not applicable", 0.9),
           "rfi_market_research": noul(0.1), "text_insufficient": noul(0.1), "brand_name_only": noul(0.1)}
    lab = map_typed(ans, "B", "jev")
    assert lab.components == ["Training"] and lab.primary_class == "Services"


def test_load_labels_dedupes_by_latest_id():
    """Test that load_labels uses latest_by_id to favor success rows over earlier error rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Monkeypatch LABELS_DIR
        old_labels_dir = common.LABELS_DIR
        common.LABELS_DIR = tmppath
        try:
            # Create directory structure
            test_dir = tmppath / "jev" / "A-run"
            test_dir.parent.mkdir(parents=True, exist_ok=True)

            # Write JSONL: error row for "x", then success row for "x"
            jsonl_path = test_dir.with_suffix('.jsonl')
            error_row = {
                "id": "x",
                "answers": {},
                "error": "first attempt failed"
            }
            success_row = {
                "id": "x",
                "answers": {
                    "primary_class": choice("Hardware", 0.9),
                    "has_general_hardware": noul(0.85),
                    "lifecycle": choice("new", 0.8),
                    "domain": choice("networking", 0.8),
                    "fulfillment_mode": choice("à la carte", 0.7),
                    "rfi_market_research": noul(0.05),
                    "text_insufficient": noul(0.05),
                    "brand_name_only": noul(0.05),
                }
            }
            with open(jsonl_path, "w") as f:
                f.write(json.dumps(error_row) + "\n")
                f.write(json.dumps(success_row) + "\n")

            # Load and verify only the success row is returned
            labels = load_labels("jev", "A-run")
            assert "x" in labels
            assert len(labels) == 1
            assert labels["x"].primary_class == "Hardware"
            assert labels["x"].components == ["General hardware"]
        finally:
            common.LABELS_DIR = old_labels_dir
