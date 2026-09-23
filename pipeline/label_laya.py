"""Label sample rows with Laya locally. Run with the bench venv:

    <bench-repo>/.venv/bin/python -m pipeline.label_laya --variant A --state S2 --verify
    <bench-repo>/.venv/bin/python -m pipeline.label_laya --variant A --state S2
"""
import argparse, json, sys, time
from pathlib import Path
from . import common, bundle

MODEL_DIR = Path("<bench-repo>/models/laya")


def run(variant, state_variant, limit=None, verify=False):
    from laya import Agent  # bench venv only
    agent = Agent(str(MODEL_DIR), device="cuda")
    rows = common.read_jsonl(common.SAMPLE)
    out_path = common.LABELS_DIR / "laya" / f"{variant}-{state_variant}.jsonl"
    todo = [r for r in rows if r["id"] not in common.done_ids(out_path)]
    todo = todo[:3] if verify else (todo[:limit] if limit else todo)
    q = bundle.questions(variant)
    print(f"{len(todo)} to do -> {out_path}", file=sys.stderr)
    t0 = time.time()
    for n, r in enumerate(todo, 1):
        t1 = time.perf_counter()
        try:
            res = agent.predict(bundle.state(r, state_variant), q)
            out = {"id": r["id"], "model": "laya", "answers": res.get("answers", res),
                   "latency_ms": round((time.perf_counter() - t1) * 1e3, 1)}
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
    a = ap.parse_args()
    run(a.variant, a.state, a.limit, a.verify)
