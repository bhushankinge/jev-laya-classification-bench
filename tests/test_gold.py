from pipeline.gold import fulfillment_from_lines

DISTI = "TD SYNNEX"

def line(partner=None, mfr=None, extracted=None):
    return {"partner": partner, "manufacturer": mfr, "extracted_data": extracted}

def test_ccw_fingerprint_is_configured_build():
    lines = [line(DISTI, "Cisco", '{"raw":{"ccw_line_number":"1. 3","deal_id":"123"}}')]
    assert fulfillment_from_lines(lines) == "configured build"

def test_all_distributor_lines_are_a_la_carte():
    lines = [line(DISTI, "Belkin"), line("Ingram Micro", "Logitech"), line("D&H", "Samsung")]
    assert fulfillment_from_lines(lines) == "à la carte"

def test_oem_heavy_without_distributor_is_configured_build():
    lines = [line(None, "Hewlett Packard Enterprise CO") for _ in range(8)]
    assert fulfillment_from_lines(lines) == "configured build"

def test_ccw_plus_distributor_addons_is_mixed():
    lines = [line(DISTI, "Cisco", '{"raw":{"cisco_quote_id":"Q1"}}'), line(DISTI, "Belkin"), line(DISTI, "Belkin")]
    assert fulfillment_from_lines(lines) == "mixed"

def test_ambiguous_is_unlabeled():
    lines = [line(None, "Cisco"), line(None, "Apple")]
    assert fulfillment_from_lines(lines) is None
