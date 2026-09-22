#!/usr/bin/env python3
"""Aggregate Qwen labels into distributions and compare against the legacy tags."""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "sample.jsonl").open()}
L = {}
for l in (HERE / "labels.jsonl").open():
    r = json.loads(l)
    if "label" in r: L[r["id"]] = r["label"]
CABLE = __import__("re").compile(r"cable|cord|adapter|connector|mount|accessor|kit|rail|shelf|speakerphone|dongle|transceiver|sfp", re.I)
SERVICEISH = __import__("re").compile(r"service|hosting|cloud|saas|internet|bandwidth|satellite|backup|azure|aws|akamai|zscaler|network as a", re.I)
fixed = 0
for lab in L.values():
    for c in lab["components"]:
        if c["kind"] == "cloud/hosting service" and CABLE.search(c["what"]) and not SERVICEISH.search(c["what"]):
            c["kind"] = "physical product"; fixed += 1  # Qwen mislabel; it even flagged its own error in `unusual`
print("cloud->physical corrections:", fixed, file=sys.stderr)

# Map fine-grained kind -> legacy-style top level for apples-to-apples comparison
TOP = {"physical product": "Hardware", "consumables/supplies": "Hardware",
       "software license (perpetual)": "Software", "software subscription/SaaS": "Software",
       "cloud/hosting service": "Software",
       "support/maintenance contract": "Maintenance & Support", "warranty/extended warranty": "Maintenance & Support",
       "professional/consulting service": "Services", "training": "Services",
       "installation/integration service": "Installation & Integration",
       "furniture/facilities": "Furniture / Facilities", "other": "Other"}
LEGACY = {"hardware": "Hardware", "hardware/new": "Hardware", "software/new": "Software", "software/renewal": "Software",
          "software": "Software", "services/maintenance": "Maintenance & Support", "maintenance": "Maintenance & Support",
          "services/installation": "Installation & Integration", "services": "Services", "services/training": "Services",
          "furniture": "Furniture / Facilities", "other": "Other"}


def pct(a, b): return round(100.0 * a / b, 1) if b else 0.0


def dist(counter, denom, min_n=0):
    return [{"value": k, "count": v, "percent": pct(v, denom)} for k, v in counter.most_common() if v >= min_n]


out = {"vehicles": {}}
for vehicle in ("SEWP", "GSA MAS", "GSA 2GIT"):
    ids = [i for i, s in S.items() if s["vehicle"] == vehicle and i in L]
    n = len(ids)
    kind_inc, kind_primary, life_primary, dom_primary, combo, top_combo, top_inc = (Counter() for _ in range(7))
    sw_life, hw_life, ms_life, brand_only, conf, unusual, whats_other, ncomp = Counter(), Counter(), Counter(), Counter(), Counter(), [], Counter(), Counter()
    kind_life = defaultdict(Counter)
    agree_top = Counter(); sw_confusion = Counter(); tagged = 0; flags = Counter()
    for i in ids:
        lab, s = L[i], S[i]
        comps = lab["components"]; p = comps[min(lab["primary_index"], len(comps) - 1)]
        ncomp[len(comps)] += 1
        kinds = sorted({c["kind"] for c in comps}); kind_inc.update(kinds); combo[" + ".join(kinds)] += 1
        tops = sorted({TOP[c["kind"]] for c in comps}); top_inc.update(tops); top_combo[" + ".join(tops)] += 1
        kind_primary[p["kind"]] += 1; life_primary[p["lifecycle"]] += 1; dom_primary[p["domain"]] += 1
        for c in comps:
            kind_life[c["kind"]][c["lifecycle"]] += 1
            if c["kind"] == "other": whats_other[c["what"].lower()] += 1
        brand_only[lab["brand_name_only"]] += 1; conf[lab["confidence"]] += 1
        if lab.get("unusual"): unusual.append(lab["unusual"])
        u = (lab.get("unusual") or "").lower() + " " + (s.get("title") or "").lower()
        if re.search(r"\brfi\b|market research|request for information|sources sought", u): flags["rfi"] += 1
        if re.search(r"attached|attachment|bom\b|spreadsheet", u) and re.search(r"no specific|does not list|not list|only reference|references an", u): flags["attachment_only"] += 1
        # legacy comparison
        tags = [t for t in (s.get("classification_tags") or []) if t in LEGACY]
        if tags:
            tagged += 1
            legacy_tops = {LEGACY[t] for t in tags}; q_tops = set(tops)
            agree_top["exact set match"] += legacy_tops == q_tops
            agree_top["legacy ⊆ qwen"] += legacy_tops <= q_tops
            agree_top["primary in legacy"] += TOP[p["kind"]] in legacy_tops
            # software lifecycle check: legacy new vs renewal against Qwen software components
            sw_q = [c for c in comps if TOP[c["kind"]] == "Software"]
            for t in tags:
                if t in ("software/new", "software/renewal") and sw_q:
                    ql = "renewal" if any(c["lifecycle"] == "renewal" for c in sw_q) else "new/other"
                    sw_confusion[(t, ql)] += 1
    out["vehicles"][vehicle] = {
        "labeled": n, "sample_requested": {"SEWP": 6000, "GSA MAS": 3000, "GSA 2GIT": 3000}[vehicle],
        "components_per_opp": dist(ncomp, n),
        "kind_incidence": dist(kind_inc, n), "kind_primary": dist(kind_primary, n),
        "kind_combinations_top15": dist(combo, n)[:15],
        "top_level_incidence": dist(top_inc, n), "top_level_primary": dist(Counter(TOP[k] for k in kind_primary.elements()), n),
        "top_level_combinations": dist(top_combo, n),
        "lifecycle_primary": dist(life_primary, n),
        "lifecycle_by_kind": {k: dist(v, sum(v.values())) for k, v in kind_life.items()},
        "domain_primary": dist(dom_primary, n),
        "brand_name_only_percent": pct(brand_only[True], n),
        "rfi_percent": pct(flags["rfi"], n), "attachment_only_percent": pct(flags["attachment_only"], n),
        "multi_component_percent": pct(sum(c for k, c in ncomp.items() if k > 1), n),
        "confidence": dist(conf, n),
        "other_whats_top30": dist(whats_other, sum(whats_other.values()))[:30],
        "unusual_count": len(unusual), "unusual_sample": unusual[:60],
        "legacy_comparison": {"tagged_in_sample": tagged, **{k: {"count": v, "percent": pct(v, tagged)} for k, v in agree_top.items()},
                              "software_new_vs_renewal": {f"legacy={a} | qwen={b}": c for (a, b), c in sorted(sw_confusion.items())}},
    }
(HERE / "analysis.json").write_text(json.dumps(out, indent=2))
for v, d in out["vehicles"].items():
    print(f"\n### {v}  (n={d['labeled']})")
    print("kind incidence:", [(x['value'], x['percent']) for x in d['kind_incidence']])
    print("top-level primary:", [(x['value'], x['percent']) for x in d['top_level_primary']])
    print("lifecycle primary:", [(x['value'], x['percent']) for x in d['lifecycle_primary']])
    print("domain primary:", [(x['value'], x['percent']) for x in d['domain_primary']][:10])
    print("legacy:", d["legacy_comparison"])
    print("other whats:", [(x['value'], x['count']) for x in d['other_whats_top30']][:12])
