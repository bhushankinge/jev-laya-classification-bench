"""Gold sets from quotes (spec Section 5). Read-only the CRM database. Ids and labels only, no text."""
import argparse, json, re, sys
from collections import Counter
from datetime import date
from . import common

TOP = {"hardware": "Hardware", "software": "Software", "services": "Services",
       "maintenance": "Maintenance & Support", "other": "Other"}
DISTRIBUTORS = re.compile(r"synnex|ingram|d&h|b&h", re.I)
CONFIGURATOR = re.compile(r"ccw_line_number|deal_id|cisco_quote_id|oca_quote|iquote|dell_quote|premier_quote|dell contract#", re.I)
OEM_HEAVY_MIN_LINES = 8


def _fingerprinted(l: dict) -> bool:
    """A line is fingerprinted per the SQL-computed 'fingerprint' bool (DB path), or by
    regex over 'extracted_data' when that key is absent (unit tests)."""
    if "fingerprint" in l:
        return l["fingerprint"] is True
    return bool(CONFIGURATOR.search(l.get("extracted_data") or ""))


def fulfillment_from_lines(lines: list[dict]) -> str | None:
    """Rule from spec Section 5 (fix round 1: TD SYNNEX is the fulfillment partner on
    nearly every line including configured OEM builds, so distributor presence must not
    veto "configured build"). Returns a FULFILLMENT value or None (unlabeled)."""
    if not lines:
        return None
    ccw_lines = [_fingerprinted(l) for l in lines]
    manufacturers = [l.get("manufacturer") for l in lines]
    per_mfr = Counter(m for m in manufacturers if m)
    heavy = {m for m, n in per_mfr.items() if n >= OEM_HEAVY_MIN_LINES}
    configured = bool(any(ccw_lines) or heavy)
    fingerprinted_mfrs = {manufacturers[i] for i, c in enumerate(ccw_lines) if c and manufacturers[i]}
    unexplained = any(
        manufacturers[i] and not ccw_lines[i]
        and manufacturers[i] not in heavy and manufacturers[i] not in fingerprinted_mfrs
        for i in range(len(lines))
    )   # a line with no manufacturer (freight, fees) explains nothing and must not force "mixed"
    if configured and unexplained:
        return "mixed"
    if configured:
        return "configured build"
    disti_lines = [bool(DISTRIBUTORS.search(l.get("partner") or "")) for l in lines]
    if all(disti_lines):
        return "à la carte"
    return None


def catalog_fingerprints(conn) -> dict[str, int]:
    cur = common.cursor(conn)
    cur.execute("SELECT extracted_data FROM public.quote_line_items TABLESAMPLE SYSTEM (1) WHERE extracted_data IS NOT NULL LIMIT 5000")
    keys = Counter()
    for r in cur.fetchall():
        try:
            d = json.loads(r["extracted_data"])
            keys.update(k.lower() for k in (d.get("raw") or {}).keys())
        except (ValueError, AttributeError, TypeError):
            continue
    return dict(keys.most_common(60))


# The latest typed quote per opportunity. Both gold builds share it so composition and
# fulfillment describe the same quote.
LATEST_QUOTE = """
      WITH typed AS (
        SELECT q.id, q.opportunity_id, v.name vehicle, coalesce(q.updated_at, q.created_at) t
        FROM public.quotes q JOIN public.vehicles v ON v.id=q.vehicle_id
        WHERE q.opportunity_id IS NOT NULL AND v.name IN ('SEWP','GSA MAS','GSA 2GIT')
          AND EXISTS (SELECT 1 FROM public.quote_line_items x WHERE x.quote_id=q.id AND x.product_type_id IS NOT NULL)),
      latest AS (SELECT *, row_number() OVER (PARTITION BY opportunity_id ORDER BY t DESC NULLS LAST, id DESC) rn FROM typed)
"""


def build_composition(conn, out_path):
    cur = common.cursor(conn)
    cur.execute(LATEST_QUOTE + """
      SELECT l.opportunity_id::text id, l.vehicle, l.id::text quote_id, array_agg(DISTINCT lower(d.value)) types
      FROM latest l JOIN public.quote_line_items qli ON qli.quote_id=l.id
      JOIN public.dropdown_options d ON d.id=qli.product_type_id
      WHERE l.rn=1 GROUP BY 1,2,3""")
    n = 0
    for r in cur.fetchall():
        classes = sorted({TOP.get(t, "Other") for t in r["types"]})
        common.append_jsonl(out_path, {"id": r["id"], "vehicle": r["vehicle"], "classes": classes, "quote_id": r["quote_id"]})
        n += 1
    return n


def build_fulfillment(conn, out_path):
    """Hardware lines of the latest typed quote per opportunity. Non-hardware lines (services,
    maintenance, freight rows typed as other) say nothing about how hardware is sourced."""
    cur = common.cursor(conn)
    cur.execute(LATEST_QUOTE + """
      SELECT l.opportunity_id::text id, max(l.vehicle) vehicle,
             json_agg(json_build_object('partner', p.name, 'manufacturer', m.name, 'fingerprint', (qli.extracted_data ~* %s))) lines
      FROM latest l JOIN public.quote_line_items qli ON qli.quote_id=l.id
      JOIN public.dropdown_options d ON d.id=qli.product_type_id
      LEFT JOIN public.partners p ON p.id=qli.partner_id
      LEFT JOIN public.manufacturers m ON m.id=qli.manufacturer_id
      WHERE l.rn=1 AND lower(d.value)='hardware'
      GROUP BY 1""", (CONFIGURATOR.pattern,))
    counts = Counter()
    for r in cur.fetchall():
        mode = fulfillment_from_lines(r["lines"])
        counts[mode] += 1
        if mode:
            ev = {"n_lines": len(r["lines"]), "ccw": any(_fingerprinted(l) for l in r["lines"])}
            common.append_jsonl(out_path, {"id": r["id"], "vehicle": r["vehicle"], "fulfillment_mode": mode, "evidence": ev})
    return dict(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["composition", "fulfillment", "fingerprints"])
    a = ap.parse_args()
    tag = date.today().strftime("%Y%m%d")
    out = common.GOLD_DIR / f"{a.what}-{tag}.jsonl"
    if a.what != "fingerprints" and out.exists():
        sys.exit(f"{out} exists; gold files are append-only - move or delete it first")
    conn = common.db()
    if a.what == "composition":
        print("rows:", build_composition(conn, out))
    elif a.what == "fulfillment":
        print("counts:", build_fulfillment(conn, out))
    else:
        print(json.dumps(catalog_fingerprints(conn), indent=1))
