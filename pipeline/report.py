"""HTML results report for one run.  python3 -m pipeline.report --run <name>"""
import argparse, base64, html, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import common

CSS = "body{font-family:Aptos,Calibri,Arial,sans-serif;color:#263746;font-size:10pt;margin:24px}h1{color:#17324D}h2{color:#17324D;border-bottom:2px solid #E76F51}table{border-collapse:collapse;margin:6pt 0 12pt;font-size:9pt}th{background:#17324D;color:#fff;padding:5px;text-align:left}td{padding:4px 6px;border:1px solid #D7E0E7}img{max-width:720px;display:block;margin:8px 0}"


def table(headers, rows):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def num(v, spec=".3f", none="n/a"):
    """None means "we could not measure it" and must never render as 0.000."""
    return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else none


def curve_png(curve, title, path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot([p["coverage"] for p in curve], [p["precision"] for p in curve], color="#184E77")
    ax.axhline(0.95, color="#E76F51", ls="--", lw=1); ax.set_xlabel("coverage (share auto-accepted)"); ax.set_ylabel("precision")
    ax.set_title(title, loc="left", fontsize=10); ax.set_ylim(0.5, 1.0); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def main(run):
    d = common.RESULTS_DIR / run
    m = json.loads((d / "metrics.json").read_text())
    P = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(run)}</title><style>{CSS}</style></head><body>",
         f"<h1>Classification experiment: {html.escape(run)}</h1><p>Sources: {html.escape(json.dumps(m['sources']))}. Labeled: {html.escape(json.dumps(m['n_labeled']))}.</p>"]
    P.append(f"<p>{html.escape(m.get('note', ''))}</p>")
    for s, r in m["by_source"].items():
        full, pq = r["full"], r["full"]["primary_vs_quote_gold"]
        fq = full["fulfillment_vs_quote_gold"]
        P.append(f"<h2>{html.escape(s)}</h2>")
        P.append(table(["Metric", "Value"], [
            ["Primary class accuracy vs quote gold", f"{num(pq['accuracy'])} (n={pq['n']})"],
            *([["Primary class accuracy, paired ids",
                f"{num(r['paired']['primary_vs_quote_gold']['accuracy'])} (n={r['paired']['primary_vs_quote_gold']['n']})"]]
              if "paired" in r else []),
            ["ECE (primary class)", num(pq["ece"], ".4f")],
            ["Cutoff for 95% precision", num(pq["cutoff_95"], ".2f", "not achieved")],
            ["Coverage at that cutoff", num(pq["coverage_at_cutoff"])],
            ["Distinct confidence values (primary class)", pq["n_distinct_confidences"]],
            ["Class set exact match vs quote gold", num(full["class_set_exact_vs_quote_gold"]["accuracy"])],
            ["Fulfillment accuracy vs quote gold", f"{num(fq['accuracy'])} (n={fq['n']})"],
            ["Cutoff for 90% fulfillment precision", num(fq["cutoff_90"], ".2f", "not achieved")],
            *[[f"Flag rate: {k}", num(v)] for k, v in r["flags_rate"].items()],
            *([[f"vs human: {k}", num(v) if not isinstance(v, dict) else num(v.get("cutoff_90_components"), ".2f", "not achieved")]
               for k, v in r["vs_human_gold"].items()] if "vs_human_gold" in r else []),
        ]))
        P.append(table(["Class", "Precision", "Recall", "F1", "Support"],
                       [[c, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", v["support"]] for c, v in pq["per_class"].items()]))
        P.append(curve_png(pq["curve"], f"{s}: precision vs coverage (primary class)", d / f"curve_{s}.png"))
    if "gate" in m:
        g = m["gate"]
        P.append("<h2>Auto-accept gate (Jev, cutoff %s)</h2>" % num(g["cutoff"], ".2f", "not achieved"))
        P.append(table(["Policy", "Coverage", "Precision (primary)", "Precision (fulfillment)", "Queued: flag / confidence / disagreement"],
                       [[k, num(v["coverage"]), num(v["precision_primary"]), num(v["precision_fulfillment"]),
                         f"{v['queued_by_flag']} / {v['queued_by_confidence']} / {v['queued_by_disagreement']}"]
                        for k, v in g.items() if isinstance(v, dict)]))
    if "jev_vs_qwen_primary" in m:
        cls = sorted({k for k in m["jev_vs_qwen_primary"]} | {k for v in m["jev_vs_qwen_primary"].values() for k in v})
        P.append("<h2>Jev vs Qwen agreement (primary class; rows Jev, columns Qwen)</h2>")
        P.append(table([""] + cls, [[a] + [m["jev_vs_qwen_primary"].get(a, {}).get(b, 0) for b in cls] for a in cls]))
    P.append("</body></html>")
    (d / "report.html").write_text("".join(P))
    print(d / "report.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    main(ap.parse_args().run)
