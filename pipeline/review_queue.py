"""Push rows to the CRM Classification Review queue and pull verdicts back as human gold.

    python3 -m pipeline.review_queue enqueue --run gold-2026-09 --reason gold_sample --per-vehicle 200 \
        --qwen v2 --jev A-S2 --laya A-S2 --token-file ~/.crm-review-token --base https://<crm-host>
    python3 -m pipeline.review_queue pull --run gold-2026-09 --token-file ~/.crm-review-token --base https://<crm-host>
"""
import argparse, random, sys
from collections import defaultdict
from datetime import date
from pathlib import Path
import requests
from . import common
from .mapping import load_labels


def _cap(value, n):
    """Truncate string to n chars, preserve None."""
    return value[:n] if isinstance(value, str) else None


def blind_map(ids, sources=("jev", "qwen"), seed=1):
    """{id: {"A": source, "B": source}} - the per-row key shuffle for blind review, so a reviewer
    cannot learn which model sits behind which proposal from its position."""
    rng = random.Random(seed)
    out = {}
    for i in ids:
        s = list(sources)
        rng.shuffle(s)
        out[i] = dict(zip(("A", "B"), s))
    return out


def build_items(ids, sample_by_id, labels_by_source, reason, keymap=None):
    items = []
    for i in ids:
        s = sample_by_id[i]
        text_bundle = {
            "vehicle": s.get("vehicle"),
            "title": _cap(s.get("title"), 500),
            "description": _cap(s.get("description"), 20000),
            "lines": _cap(s.get("lines"), 5000),
            "attachmentExcerpt": _cap(s.get("attachment_text"), 5000),
        }
        proposals = {src: labs[i].to_dict() for src, labs in labels_by_source.items() if i in labs}
        if keymap and i in keymap:
            proposals = {k: dict(proposals[src], source=k) for k, src in keymap[i].items() if src in proposals}
        items.append({"opportunityId": i, "reason": reason, "textBundle": text_bundle, "proposals": proposals})
    return items


def stratified_ids(labels, sample_by_id, per_vehicle, seed=1):
    """Round-robin over (vehicle, first component subclass) so rare subclasses are represented."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for i, l in labels.items():
        if i in sample_by_id:
            buckets[(sample_by_id[i]["vehicle"], l.components[0])].append(i)
    for b in buckets.values():
        rng.shuffle(b)
    out = []
    for veh in sorted({v for v, _ in buckets}):
        keys = sorted(k for k in buckets if k[0] == veh)
        picked = []
        while len(picked) < per_vehicle and any(buckets[k] for k in keys):
            for k in keys:
                if buckets[k] and len(picked) < per_vehicle:
                    picked.append(buckets[k].pop())
        out.extend(picked)
    return out


def disagreement_ids(jev, qwen, limit, seed=1):
    """Half the budget on rows where the two models pick different classes, half on rows that agree
    on the class but not on the component set; either stratum fills the other's shortfall. Class
    disagreements are the loud ones and would otherwise take the whole queue."""
    shared = [i for i in jev if i in qwen]
    cls = [i for i in shared if jev[i].primary_class != qwen[i].primary_class]
    comp = [i for i in shared if jev[i].primary_class == qwen[i].primary_class
            and set(jev[i].components) != set(qwen[i].components)]
    rng = random.Random(seed)
    rng.shuffle(cls); rng.shuffle(comp)
    n_cls = min(len(cls), max(limit // 2, limit - len(comp)))
    return cls[:n_cls] + comp[:limit - n_cls]


def _headers(token_file):
    return {"Authorization": f"Bearer {Path(token_file).expanduser().read_text().strip()}"}


def enqueue(base, token_file, run, reason, items):
    created = skipped = 0
    for k in range(0, len(items), 100):
        r = requests.post(f"{base}/api/classification-reviews/enqueue", json={"runId": run, "items": items[k:k + 100]},
                          headers=_headers(token_file), timeout=60)
        r.raise_for_status()
        d = r.json()["data"]; created += d["created"]; skipped += d["skipped"]
    print(f"enqueued {created}, skipped {skipped}", file=sys.stderr)


def pull(base, token_file, run):
    r = requests.get(f"{base}/api/classification-reviews/export", params={"runId": run, "status": "confirmed,corrected"},
                     headers=_headers(token_file), timeout=120)
    r.raise_for_status()
    out = common.GOLD_DIR / f"human-{date.today().strftime('%Y%m%d')}.jsonl"
    # verdicts are model-agnostic, so nothing needs de-anonymizing; the gold row just records
    # where the A/B key map lives, for anyone scoring per model later.
    blind = common.GOLD_DIR / f"blind-{run}.jsonl"
    n = 0
    for row in r.json()["data"]:
        common.append_jsonl(out, {"id": row["opportunityId"], "verdict": row["verdict"], "reviewer_id": row["reviewerId"],
                                  "reviewed_at": row["reviewedAt"], "reason": row["reason"], "run_id": row["runId"],
                                  "blind_map": str(blind) if blind.exists() else None})
        n += 1
    print(f"wrote {n} verdicts -> {out}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["enqueue", "pull"])
    ap.add_argument("--run", required=True); ap.add_argument("--base", required=True); ap.add_argument("--token-file", required=True)
    ap.add_argument("--reason", default="gold_sample", choices=["gold_sample", "disagreement", "fulfillment", "low_confidence"])
    ap.add_argument("--per-vehicle", type=int, default=200); ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--qwen", default="v2"); ap.add_argument("--jev"); ap.add_argument("--laya")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "pull":
        pull(a.base, a.token_file, a.run); sys.exit()
    sample = {r["id"]: r for r in common.read_jsonl(common.SAMPLE)}
    labels = {k: load_labels(k, v) for k, v in (("qwen", a.qwen), ("jev", a.jev), ("laya", a.laya)) if v}
    keymap = None
    if a.reason == "disagreement":
        ids = disagreement_ids(labels["jev"], labels["qwen"], a.limit, a.seed)
        keymap = blind_map(ids, ("jev", "qwen"), a.seed)
        blind_path = common.GOLD_DIR / f"blind-{a.run}.jsonl"
        for i in ids:
            common.append_jsonl(blind_path, {"id": i, **keymap[i]})
        print(f"blind key map -> {blind_path}", file=sys.stderr)
    elif a.reason == "fulfillment":
        hw = {i: l for i, l in labels["qwen"].items() if l.fulfillment_mode != "not applicable"}
        ids = stratified_ids(hw, sample, a.per_vehicle, a.seed)
    else:
        ids = stratified_ids(labels["qwen"], sample, a.per_vehicle, a.seed)
    enqueue(a.base, a.token_file, a.run, a.reason, build_items(ids, sample, labels, a.reason, keymap))
