"""The one label schema every labeler is mapped onto (spec Section 2)."""
from dataclasses import dataclass, field, asdict

CLASSES = ["Hardware", "Software", "Maintenance & Support", "Services",
           "Installation & Integration", "Furniture / Facilities", "Other"]
SUBCLASSES = ["General hardware", "Consumables/Supplies", "Perpetual license", "Subscription/SaaS",
              "Cloud/Hosting", "Support/Maintenance contract", "Warranty", "Professional/Consulting",
              "Training", "Installation/Integration", "Furniture/Facilities", "Other"]
SUBCLASS_TO_CLASS = {
    "General hardware": "Hardware", "Consumables/Supplies": "Hardware",
    "Perpetual license": "Software", "Subscription/SaaS": "Software", "Cloud/Hosting": "Software",
    "Support/Maintenance contract": "Maintenance & Support", "Warranty": "Maintenance & Support",
    "Professional/Consulting": "Services", "Training": "Services",
    "Installation/Integration": "Installation & Integration",
    "Furniture/Facilities": "Furniture / Facilities", "Other": "Other",
}
LIFECYCLE = ["new", "renewal", "upgrade/expansion", "replacement/refresh", "unknown"]
DOMAINS = ["networking", "compute/servers", "end-user devices", "storage/backup", "printing/imaging",
           "audio-visual", "cybersecurity", "business/productivity software", "infrastructure/platform software",
           "data center/power/cooling", "telecom/mobility", "furniture/facilities", "lab/scientific/medical",
           "cables/accessories/peripherals", "other"]
FULFILLMENT = ["à la carte", "configured build", "mixed", "not applicable"]
FLAGS = ["rfi_market_research", "text_insufficient", "brand_name_only"]
HARDWARE_SUBCLASSES = {"General hardware", "Consumables/Supplies"}

# Qwen labeler vocabulary -> schema
QWEN_KIND = {
    "physical product": "General hardware", "consumables/supplies": "Consumables/Supplies",
    "software license (perpetual)": "Perpetual license", "software subscription/SaaS": "Subscription/SaaS",
    "cloud/hosting service": "Cloud/Hosting", "support/maintenance contract": "Support/Maintenance contract",
    "warranty/extended warranty": "Warranty", "professional/consulting service": "Professional/Consulting",
    "training": "Training", "installation/integration service": "Installation/Integration",
    "furniture/facilities": "Furniture/Facilities", "other": "Other",
}
QWEN_LIFECYCLE = {"new purchase": "new", "renewal": "renewal", "upgrade/expansion/add-on": "upgrade/expansion",
                  "replacement/refresh": "replacement/refresh", "unknown": "unknown"}
QWEN_CONFIDENCE = {"high": 0.9, "medium": 0.7, "low": 0.5}


@dataclass
class Label:
    primary_class: str
    components: list[str]
    lifecycle: str
    domain: str
    fulfillment_mode: str
    flags: dict
    confidence: dict = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def map_qwen(raw: dict) -> Label:
    comps = raw["components"]
    primary = comps[min(raw.get("primary_index", 0), len(comps) - 1)]
    subclasses = []
    for c in comps:
        s = QWEN_KIND[c["kind"]]
        if s not in subclasses:
            subclasses.append(s)
    conf = QWEN_CONFIDENCE[raw.get("confidence", "medium")]
    has_hw = bool(HARDWARE_SUBCLASSES & set(subclasses))
    fulfillment = raw.get("fulfillment_mode") if has_hw else "not applicable"
    return Label(
        primary_class=SUBCLASS_TO_CLASS[QWEN_KIND[primary["kind"]]],
        components=subclasses,
        lifecycle=QWEN_LIFECYCLE[primary["lifecycle"]],
        domain=primary["domain"],
        fulfillment_mode=fulfillment or "not applicable",
        flags={"rfi_market_research": bool(raw.get("rfi_market_research", False)),
               "text_insufficient": bool(raw.get("text_insufficient", False)),
               "brand_name_only": bool(raw.get("brand_name_only", False))},
        confidence={k: conf for k in ("primary_class", "components", "lifecycle", "domain", "fulfillment_mode")},
        source="qwen",
    )
