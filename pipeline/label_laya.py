"""Label sample rows with Laya locally. Run with the bench venv:

    $LAYA_PYTHON -m pipeline.label_laya --variant A --state S2 --verify
    $LAYA_PYTHON -m pipeline.label_laya --variant A --state S2
"""
import argparse, json, sys, time
from pathlib import Path
from . import common, bundle

MODEL_DIR = Path(common.env("LAYA_MODEL_DIR"))   # local copy of convaiinnovations/laya


def run(variant, state_variant, limit=None, verify=False, ids_from=None, balanced=False,
        model_dir=None, tag=None, dtype=None):
    if not common.SAMPLE.exists():
        raise SystemExit("sample missing: run python3 qwen_discovery/qwen_classify.py 0 to build it")
    import laya
    from laya import Agent  # bench venv only
    agent = Agent(str(model_dir or MODEL_DIR), device="cuda")
    if dtype:                                   # override the checkpoint's amp_dtype (bf16 on cc >= 8)
        import torch
        agent.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    rows = common.read_jsonl(common.SAMPLE)
    out_path = common.LABELS_DIR / "laya" / f"{variant}-{state_variant}{'-' + tag if tag else ''}.jsonl"
    todo = common.select_rows(rows, limit, common.done_ids(out_path), ids_from, balanced)
    if verify:
        todo = todo[:3]
    q = bundle.questions(variant)
    print(f"{len(todo)} to do -> {out_path}", file=sys.stderr)
    t0 = time.time()
    for n, r in enumerate(todo, 1):
        t1 = time.perf_counter()
        try:
            res = agent.predict(bundle.state(r, state_variant), q)
            out = {"id": r["id"], "model": "laya", "answers": res["answers"],
                   "latency_ms": round((time.perf_counter() - t1) * 1e3, 1),
                   "laya": laya.__version__, "dtype": str(agent.dtype)}
        except Exception as e:  # noqa: BLE001
            out = {"id": r["id"], "error": repr(e)[:200]}
        if verify:
            print(json.dumps(out, indent=1, default=str)[:3000])
        else:
            common.append_jsonl(out_path, out)
        if n % 200 == 0:
            print(f"{n}/{len(todo)} {n/(time.time()-t0):.1f} rows/s", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=["A", "B", "C"])
    ap.add_argument("--state", default="S2", choices=["S1", "S2", "S3"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--ids-from", help="jsonl whose ids the run is restricted to (e.g. a gold file)")
    ap.add_argument("--vehicle-balanced", action="store_true", help="interleave vehicles so --limit is not one vehicle")
    ap.add_argument("--model-dir", help="checkpoint directory (default LAYA_MODEL_DIR)")
    ap.add_argument("--tag", help="suffix for the output file: labels/laya/<variant>-<state>-<tag>.jsonl")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], help="force the CUDA autocast dtype")
    a = ap.parse_args()
    run(a.variant, a.state, a.limit, a.verify, a.ids_from, a.vehicle_balanced, a.model_dir, a.tag, a.dtype)
