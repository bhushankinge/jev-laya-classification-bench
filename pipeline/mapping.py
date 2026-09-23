"""Map typed answers (Jev/Laya) and Qwen output onto schema.Label."""
from . import common
from .schema import Label, SUBCLASSES, SUBCLASS_TO_CLASS, HARDWARE_SUBCLASSES, map_qwen
from .bundle import HAS_KEY

FIRST_SUBCLASS = {}
for s, c in SUBCLASS_TO_CLASS.items():
    FIRST_SUBCLASS.setdefault(c, s)


def _conf(a: dict) -> float:
    if a.get("type") == "noul":
        p = float(a["noul"]); return round(max(p, 1 - p), 4)
    return float(a.get("confidence", 0.0))


def map_typed(answers: dict, variant: str, source: str, noul_threshold: float = 0.5) -> Label:
    conf = {k: _conf(v) for k, v in answers.items()}
    primary = answers["primary_class"]["choice"]
    if variant == "B":
        comps = [answers["primary_subclass"]["choice"]]
    else:
        comps = [s for s in SUBCLASSES if HAS_KEY[s] in answers and float(answers[HAS_KEY[s]]["noul"]) >= noul_threshold]
        if not comps:
            mentioned_for_primary = [s for s in SUBCLASSES if HAS_KEY[s] in answers and SUBCLASS_TO_CLASS[s] == primary]
            comps = [mentioned_for_primary[0]] if mentioned_for_primary else [FIRST_SUBCLASS[primary]]
    has_hw = bool(HARDWARE_SUBCLASSES & set(comps))
    fulfillment = answers["fulfillment_mode"]["choice"] if has_hw else "not applicable"
    flags = {f: float(answers[f]["noul"]) >= noul_threshold for f in ("rfi_market_research", "text_insufficient", "brand_name_only")}
    return Label(primary_class=primary, components=comps, lifecycle=answers["lifecycle"]["choice"],
                 domain=answers["domain"]["choice"], fulfillment_mode=fulfillment, flags=flags,
                 confidence=conf, source=source)


def load_labels(source: str, run: str) -> dict[str, Label]:
    rows = common.latest_by_id(common.LABELS_DIR / source / f"{run}.jsonl").values()
    out = {}
    for r in rows:
        if "error" in r:
            continue
        if source == "qwen":
            out[r["id"]] = map_qwen(r["label"])
        else:
            variant = run.split("-")[0]
            out[r["id"]] = map_typed(r["answers"], variant, source)
    return out
