from pathlib import Path
from pipeline import common

def test_paths_exist_and_are_under_repo():
    root = Path(__file__).resolve().parents[1]
    assert common.SAMPLE == root / "qwen_discovery" / "sample.jsonl"
    assert common.GOLD_DIR == root / "gold"
    assert common.LABELS_DIR == root / "labels"
    assert common.RESULTS_DIR == root / "results"

def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    common.append_jsonl(p, {"id": "a", "v": 1})
    common.append_jsonl(p, {"id": "b", "v": 2})
    assert common.read_jsonl(p) == [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    assert common.done_ids(p) == {"a", "b"}

def test_done_ids_excludes_error_rows_so_they_retry(tmp_path):
    p = tmp_path / "x.jsonl"
    common.append_jsonl(p, {"id": "a", "v": 1})
    common.append_jsonl(p, {"id": "b", "error": "boom"})
    assert common.done_ids(p) == {"a"}

def test_latest_by_id_lets_a_later_success_win_over_an_earlier_error(tmp_path):
    p = tmp_path / "x.jsonl"
    common.append_jsonl(p, {"id": "a", "error": "boom"})
    common.append_jsonl(p, {"id": "a", "v": 1})
    assert common.latest_by_id(p) == {"a": {"id": "a", "v": 1}}

def rows(*spec):
    return [{"id": i, "vehicle": v} for i, v in spec]

SAMPLE_ROWS = rows(("a", "SEWP"), ("b", "SEWP"), ("c", "SEWP"), ("d", "GSA MAS"), ("e", "GSA MAS"))

def test_select_rows_limits_before_the_done_filter():
    # the limit picks the same prefix every run; already-done rows drop out of it, not extra rows in
    assert [r["id"] for r in common.select_rows(SAMPLE_ROWS, limit=3, done={"b"})] == ["a", "c"]

def test_select_rows_balances_vehicles_within_the_limit():
    ids = [r["id"] for r in common.select_rows(SAMPLE_ROWS, limit=4, balanced=True)]
    assert ids == ["d", "a", "e", "b"]            # GSA MAS and SEWP interleaved, not one vehicle

def test_select_rows_restricts_to_ids_from_a_gold_file(tmp_path):
    gold = tmp_path / "gold.jsonl"
    common.append_jsonl(gold, {"id": "c", "classes": ["Hardware"]})
    common.append_jsonl(gold, {"id": "e", "classes": ["Software"]})
    assert [r["id"] for r in common.select_rows(SAMPLE_ROWS, ids_from=gold)] == ["c", "e"]
    assert [r["id"] for r in common.select_rows(SAMPLE_ROWS, ids_from={"a"})] == ["a"]
