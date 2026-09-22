#!/usr/bin/env python3
"""Data-driven classification discovery for SEWP / GSA MAS / GSA 2GIT.

Phase A (sample): pull a random sample per vehicle from the CRM database (read-only) with
title, description, structured lines, attachment text, existing tags, quote types.
Phase B (label): ask the PCAI Qwen endpoint for a structured, fine-grained
extraction per opportunity (closed `kind`/`lifecycle`/`domain`, plus free-text
`what` so anything outside the closed sets surfaces). Resumable JSONL.

Credentials are read at runtime from local protected files, never persisted.
"""
import json, os, random, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg2, psycopg2.extras, requests

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "sample.jsonl"
LABELS = HERE / "labels.jsonl"
VEHICLES = {"SEWP": 6000, "GSA MAS": 3000, "GSA 2GIT": 3000}
ENDPOINT = ("https://<qwen-endpoint>"
            ".example.internal/v1/chat/completions")
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
CONCURRENCY = int(os.environ.get("CONC", "8"))

KINDS = ["physical product", "software license (perpetual)", "software subscription/SaaS",
         "cloud/hosting service", "support/maintenance contract", "warranty/extended warranty",
         "professional/consulting service", "installation/integration service", "training",
         "consumables/supplies", "furniture/facilities", "other"]
LIFECYCLE = ["new purchase", "renewal", "upgrade/expansion/add-on", "replacement/refresh", "unknown"]
DOMAINS = ["networking", "compute/servers", "end-user devices", "storage/backup", "printing/imaging",
           "audio-visual", "cybersecurity", "business/productivity software", "infrastructure/platform software",
           "data center/power/cooling", "telecom/mobility", "furniture/facilities", "lab/scientific/medical",
           "cables/accessories/peripherals", "other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "components": {"type": "array", "minItems": 1, "maxItems": 6, "items": {
            "type": "object",
            "properties": {
                "what": {"type": "string", "maxLength": 60},
                "kind": {"type": "string", "enum": KINDS},
                "lifecycle": {"type": "string", "enum": LIFECYCLE},
                "domain": {"type": "string", "enum": DOMAINS},
                "brand": {"type": ["string", "null"], "maxLength": 40},
            },
            "required": ["what", "kind", "lifecycle", "domain", "brand"],
            "additionalProperties": False,
        }},
        "primary_index": {"type": "integer", "minimum": 0, "maximum": 5},
        "brand_name_only": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "unusual": {"type": ["string", "null"], "maxLength": 120},
    },
    "required": ["components", "primary_index", "brand_name_only", "confidence", "unusual"],
    "additionalProperties": False,
}

SYSTEM = f"""You analyze US federal IT procurement solicitations (RFQs) on behalf of an IT reseller.
Read the solicitation text and identify every distinct thing the government is buying.

Rules:
- One component per distinct offering type. Merge many part numbers of the same type into one component (e.g. 40 laptops + 40 docks = "laptops and docks" OR two components if they differ in kind).
- `what`: short generic noun phrase (1-5 words), not a part number. Examples: "laptops", "network switches", "Adobe subscription", "VMware support renewal", "rack installation", "office chairs".
- `kind` must be one of: {KINDS}
- `lifecycle`: renewal ONLY if text indicates renewing/extending an existing license, subscription, support or warranty. New hardware buys are "new purchase" unless they say replacement/refresh/upgrade.
- `domain` must be one of: {DOMAINS}
- `brand`: OEM/publisher if stated, else null.
- `primary_index`: index of the component that dominates the requirement.
- `brand_name_only`: true if the solicitation restricts to a specific brand / no substitutes.
- `unusual`: anything the buyer wants that does NOT fit the kind list well (else null). Be specific.
- `confidence`: how sure you are given the text quality.
Return only JSON matching the schema."""


def db():
    t = Path("<credentials-file>").read_text()
    row = next(l for l in t.splitlines() if "<db-row>" in l)
    pw = re.search(r"Admin: \*\*<db-user> / ([^*]+)\*\*", row).group(1).strip()
    c = psycopg2.connect(host="<db-host>", port=5432, dbname="<db-name>",
                         user="<db-user>", password=pw, connect_timeout=8)
    c.set_session(readonly=True, autocommit=True)
    return c


def api_key():
    for line in Path("<ai-services-env>").read_text().splitlines():
        if line.startswith("PCAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("PCAI_API_KEY missing")


def build_sample():
    if SAMPLE.exists():
        return
    cur = db().cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    out = SAMPLE.open("w")
    for vehicle, n in VEHICLES.items():
        cur.execute("""
        WITH s AS (
          SELECT o.id, o.client_rfx, o.title, o.classification_tags, o.created_at
          FROM public.opportunities o JOIN public.vehicles v ON v.id=o.vehicle_id
          WHERE v.name=%s ORDER BY md5(o.id::text) LIMIT %s)
        SELECT s.id::text AS id, s.client_rfx, s.title, s.classification_tags, s.created_at::date::text AS created,
          cd.description, cd.type AS rfq_type,
          (SELECT string_agg(coalesce(manufacturer_name,'')||' '||coalesce(mpn,'')||' '||coalesce(product_description,'')||' x'||coalesce(quantity::text,''), E'\n')
             FROM (SELECT * FROM public.opportunity_line_items WHERE opportunity_id=s.id LIMIT 15) li) AS lines,
          (SELECT string_agg(c.text, E'\n' ORDER BY c.chunk_index)
             FROM (SELECT text, chunk_index FROM rag.chunks WHERE opportunity_id=s.id ORDER BY document_id, chunk_index LIMIT 3) c) AS attachment_text,
          (SELECT array_agg(DISTINCT d.value) FROM public.quotes q JOIN public.quote_line_items qli ON qli.quote_id=q.id
             JOIN public.dropdown_options d ON d.id=qli.product_type_id WHERE q.opportunity_id=s.id) AS quote_types
        FROM s LEFT JOIN LATERAL (SELECT description, type FROM public.opportunity_co_data WHERE opportunity_id=s.id
                                  ORDER BY updated_at DESC NULLS LAST LIMIT 1) cd ON true
        """, (vehicle, n))
        k = 0
        for r in cur.fetchall():
            r["vehicle"] = vehicle
            out.write(json.dumps(r, default=str) + "\n"); k += 1
        print(vehicle, k, file=sys.stderr)
    out.close()


BOILER = re.compile(r"(Requirements apply for Providers with Established Authorized Reseller Programs \(EARP\)|"
                    r"Requires products from Authorized Resellers|EPEAT Level - Gold, Silver or Bronze|"
                    r"TAA Compliant Products Only)\s*", re.I)


def prompt_for(r):
    desc = BOILER.sub("", r.get("description") or "").strip()
    parts = [f"Contract vehicle: {r['vehicle']}", f"Title: {r.get('title') or ''}"]
    if r.get("rfq_type"): parts.append(f"RFQ type: {r['rfq_type']}")
    parts.append(f"Description:\n{desc[:2500]}")
    if r.get("lines"): parts.append(f"Line items:\n{r['lines'][:1500]}")
    if r.get("attachment_text"): parts.append(f"Attachment excerpt:\n{r['attachment_text'][:2000]}")
    return "\n\n".join(parts)


_key = {"v": api_key(), "t": time.time()}
_lock = threading.Lock()


def call(r, sess):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 600,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_for(r)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "rfq", "schema": SCHEMA, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(4):
        try:
            resp = sess.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {_key['v']}"}, timeout=180, verify=False)
            if resp.status_code == 401:
                with _lock: _key["v"] = api_key()
                continue
            resp.raise_for_status()
            j = resp.json()
            txt = j["choices"][0]["message"]["content"]
            return {"id": r["id"], "label": json.loads(txt), "usage": j.get("usage")}
        except Exception as e:  # noqa
            err = repr(e)[:200]
            time.sleep(2 * (attempt + 1))
    return {"id": r["id"], "error": err}


def label_all(limit=None):
    done = set()
    if LABELS.exists():
        for line in LABELS.open():
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    rows = [json.loads(l) for l in SAMPLE.open()]
    todo = [r for r in rows if r["id"] not in done]
    if limit: todo = todo[:limit]
    print(f"{len(done)} done, {len(todo)} to do", file=sys.stderr)
    requests.packages.urllib3.disable_warnings()
    out = LABELS.open("a"); t0 = time.time(); n = 0; toks = 0
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        sess = requests.Session()
        futs = [ex.submit(call, r, sess) for r in todo]
        for f in as_completed(futs):
            res = f.result(); out.write(json.dumps(res) + "\n"); out.flush(); n += 1
            toks += (res.get("usage") or {}).get("total_tokens", 0)
            if n % 50 == 0:
                el = time.time() - t0
                print(f"{n}/{len(todo)} {n/el:.2f} opp/s {toks/el:.0f} tok/s eta {((len(todo)-n)/(n/el))/60:.0f} min", file=sys.stderr)
    out.close()


if __name__ == "__main__":
    build_sample()
    label_all(int(sys.argv[1]) if len(sys.argv) > 1 else None)
