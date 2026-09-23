"""Label sample rows with Jev. Resumable; ≤ MAX_RPS requests/second; logs tokens and latency.
Error rows are retried on the next run; readers must take the last row per id.

    python3 -m pipeline.label_jev --variant A --state S2 --verify          # 3 rows, prints raw answers
    python3 -m pipeline.label_jev --variant A --state S2 --limit 2000
    python3 -m pipeline.label_jev --variant A --state S2 --ids-from gold/composition-<date>.jsonl \
        --vehicle-balanced --limit 1000
"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from . import common, bundle

URL = "https://api.typesafe.ai/v1/systemone"
MAX_RPS = 10.0
REQUIRED = ("primary_class", "lifecycle", "domain", "fulfillment_mode")


def parse_response(js: dict) -> dict:
    return {"model": js.get("model"), "usage": js.get("usage") or {}, "answers": js.get("answers") or {}}


def call(sess, key, row, questions, state_variant):
    body = {"state": bundle.state(row, state_variant), "questions": questions, "model": "jev-latest"}
    t0 = time.perf_counter()
    err = "rate limited (429) on every attempt"
    for attempt in range(3):
        try:
            r = sess.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1)); continue
            r.raise_for_status()
            out = parse_response(r.json())
            missing = [k for k in REQUIRED if k not in out["answers"]]
            if missing:   # a partial answer set is an error row, not a label to be mapped later
                return {"id": row["id"], "error": f"incomplete answers: {', '.join(missing)}"}
            out.update({"id": row["id"], "latency_ms": round((time.perf_counter() - t0) * 1e3, 1)})
            return out
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:200]; time.sleep(1 + attempt)
    return {"id": row["id"], "error": err}


def run(variant, state_variant, limit=None, verify=False, concurrency=8, ids_from=None, balanced=False):
    if not common.SAMPLE.exists():
        raise SystemExit("sample missing: run python3 qwen_discovery/qwen_classify.py 0 to build it")
    rows = common.read_jsonl(common.SAMPLE)
    out_path = common.LABELS_DIR / "jev" / f"{variant}-{state_variant}.jsonl"
    todo = common.select_rows(rows, limit, common.done_ids(out_path), ids_from, balanced)
    if verify:
        todo = todo[:3]
    q, key, sess = bundle.questions(variant), common.jev_key(), requests.Session()
    print(f"{len(todo)} to do -> {out_path}", file=sys.stderr)
    t0, n = time.time(), 0
    with ThreadPoolExecutor(concurrency) as ex:
        futs = []
        for i, r in enumerate(todo):
            delay = i / MAX_RPS - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)
            futs.append(ex.submit(call, sess, key, r, q, state_variant))
        for f in as_completed(futs):
            res = f.result(); n += 1
            if verify:
                print(json.dumps(res, indent=1)[:3000])
            else:
                common.append_jsonl(out_path, res)
            if n % 100 == 0:
                print(f"{n}/{len(todo)} {n/(time.time()-t0):.1f} req/s", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=["A", "B", "C"])
    ap.add_argument("--state", default="S2", choices=["S1", "S2", "S3"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--ids-from", help="jsonl whose ids the run is restricted to (e.g. a gold file)")
    ap.add_argument("--vehicle-balanced", action="store_true", help="interleave vehicles so --limit is not one vehicle")
    a = ap.parse_args()
    run(a.variant, a.state, a.limit, a.verify, ids_from=a.ids_from, balanced=a.vehicle_balanced)
