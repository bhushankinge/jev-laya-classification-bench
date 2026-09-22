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
