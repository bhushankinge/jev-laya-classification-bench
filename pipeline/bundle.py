"""Jev/Laya question bundles (variants A/B/C) and state text builders (S1/S2/S3). Spec Section 4."""
import re
from .schema import CLASSES, SUBCLASSES, LIFECYCLE, DOMAINS, FULFILLMENT, SUBCLASS_TO_CLASS

BOILER = re.compile(r"(Requirements apply for Providers with Established Authorized Reseller Programs \(EARP\)|"
                    r"Requires products from Authorized Resellers|EPEAT Level - Gold, Silver or Bronze|"
                    r"TAA Compliant Products Only)\s*", re.I)

CLASS_DEF = {
    "Hardware": "physical IT products or consumables: endpoints, servers, network gear, storage, AV, print, power, cables, toner",
    "Software": "software licenses, subscriptions, SaaS, cloud or hosting services",
    "Maintenance & Support": "support or maintenance contracts, warranties, SmartNet-type coverage, sustainment",
    "Services": "labor against a SOW: professional, consulting, engineering, managed services, training",
    "Installation & Integration": "site installation, rack and stack, configuration, deployment, cutover, removal",
    "Furniture / Facilities": "seating, desks, tables, cabinets, carts, lecterns, shelving",
    "Other": "freight, fees, pass-through or too generic to tell",
}
SUBCLASS_DEF = {
    "General hardware": "physical IT products with a serial or asset identity (devices, servers, switches, AV, cables, peripherals)",
    "Consumables/Supplies": "items consumed in use: toner, ink, paper, media, batteries, office supplies",
    "Perpetual license": "software bought once with an indefinite right to use",
    "Subscription/SaaS": "software with a term (annual, multi-year, monthly), on-prem or hosted",
    "Cloud/Hosting": "consumption or hosting service: IaaS, GovCloud credits, bandwidth, backup-as-a-service",
    "Support/Maintenance contract": "vendor support, software maintenance, break/fix, technical support, sustainment",
    "Warranty": "manufacturer or third-party warranty, extended or uplifted, sold as its own line",
    "Professional/Consulting": "labor delivered against a SOW or PWS: engineering, implementation, migration, managed services, staffing",
    "Training": "instructor-led or on-demand training, courses, certification vouchers",
    "Installation/Integration": "site installation, rack and stack, configuration, deployment, cutover, removal",
    "Furniture/Facilities": "seating, desks, tables, casegoods, storage cabinets, carts, lecterns, shelving",
    "Other": "freight, fees, pass-through, or a generic request with no detail",
}
LIFECYCLE_DEF = {
    "new": "a first-time purchase; new hardware is new unless replacement or upgrade is stated",
    "renewal": "text states an existing license, subscription, support or warranty is renewed, extended or co-termed",
    "upgrade/expansion": "adding seats, capacity or modules to something already owned",
    "replacement/refresh": "replacing or refreshing existing equipment",
    "unknown": "the text does not say",
}
FULFILLMENT_DEF = {
    "à la carte": "every hardware item is an orderable SKU with a list or contract price a rep can add from a distributor or catalog; no OEM configurator step",
    "configured build": "at least one item must be configured in an OEM portal (Cisco CCW, HPE OCA or iQuote, Dell Premier, NetApp, Pure) where price depends on configuration, deal registration and human input",
    "mixed": "a configured core plus à la carte add-ons",
    "not applicable": "no hardware is being bought",
}
FLAG_Q = {
    "rfi_market_research": "Is this notice a request for information, sources sought or market research rather than a purchase?",
    "text_insufficient": "Does the text fail to say what is being bought, pointing only to an attachment, BOM or spreadsheet?",
    "brand_name_only": "Does the notice restrict the purchase to a specific brand with no substitutes?",
}


def questions(variant: str) -> dict:
    q = {"primary_class": {"type": "choice", "instructions": "Which class dominates what the government is buying?",
                           "criteria": CLASS_DEF}}
    if variant == "A":
        for s in SUBCLASSES:
            q["has_" + re.sub(r"[^a-z]+", "_", s.lower()).strip("_")] = {
                "type": "noul", "instructions": f"Does the solicitation include {SUBCLASS_DEF[s]}?"}
    elif variant == "B":
        q["primary_subclass"] = {"type": "choice", "instructions": "Which component subclass dominates?", "criteria": SUBCLASS_DEF}
    elif variant == "C":
        for cls in CLASSES:
            subs = {s: SUBCLASS_DEF[s] for s, c in SUBCLASS_TO_CLASS.items() if c == cls}
            if len(subs) > 1:
                q["subclass_of_" + re.sub(r"[^a-z]+", "_", cls.lower()).strip("_")] = {
                    "type": "choice", "instructions": f"If {cls} is present, which subclass?", "criteria": subs}
        for s in SUBCLASSES:
            q["has_" + re.sub(r"[^a-z]+", "_", s.lower()).strip("_")] = {
                "type": "noul", "instructions": f"Does the solicitation include {SUBCLASS_DEF[s]}?"}
    else:
        raise ValueError(variant)
    q["lifecycle"] = {"type": "choice", "instructions": "Lifecycle of the dominant component", "criteria": LIFECYCLE_DEF}
    q["domain"] = {"type": "choice", "instructions": "Solution domain of the dominant component", "criteria": {d: "" for d in DOMAINS}}
    q["fulfillment_mode"] = {"type": "choice", "instructions": "How would a reseller source the hardware?", "criteria": FULFILLMENT_DEF}
    for f, text in FLAG_Q.items():
        q[f] = {"type": "noul", "instructions": text}
    return q


HAS_KEY = {s: "has_" + re.sub(r"[^a-z]+", "_", s.lower()).strip("_") for s in SUBCLASSES}


def state(row: dict, variant: str) -> str:
    desc = BOILER.sub("", row.get("description") or "").strip()
    parts = [f"Contract vehicle: {row.get('vehicle', '')}", f"Title: {row.get('title') or ''}"]
    if row.get("rfq_type"):
        parts.append(f"RFQ type: {row['rfq_type']}")
    parts.append(f"Description:\n{desc[:2000]}")
    if variant in ("S2", "S3") and row.get("lines"):
        parts.append(f"Line items:\n{row['lines'][:1200]}")
    if variant == "S3" and row.get("attachment_text"):
        parts.append(f"Attachment excerpt:\n{row['attachment_text'][:1500]}")
    return "\n\n".join(parts)
