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
    for s, r in m["by_source"].items():
        pq = r["primary_vs_quote_gold"]
        P.append(f"<h2>{html.escape(s)}</h2>")
        P.append(table(["Metric", "Value"], [
            ["Primary class accuracy vs quote gold", f"{pq['accuracy']:.3f} (n={pq['n']})"],
            ["ECE (primary class)", pq["ece"]], ["Cutoff for 95% precision", pq["cutoff_95"]],
            ["Component set exact match vs quote gold", f"{r['component_set_exact_vs_quote_gold']['accuracy']:.3f}"],
            ["Fulfillment accuracy vs quote gold", f"{r['fulfillment_vs_quote_gold']['accuracy']:.3f} (n={r['fulfillment_vs_quote_gold']['n']})"],
            ["Cutoff for 90% fulfillment precision", r["fulfillment_vs_quote_gold"]["cutoff_90"]],
            *[[f"Flag rate: {k}", f"{v:.3f}"] for k, v in r["flags_rate"].items()],
            *([[f"vs human: {k}", f"{v:.3f}" if isinstance(v, float) else v] for k, v in r["vs_human_gold"].items()] if "vs_human_gold" in r else []),
        ]))
        P.append(table(["Class", "Precision", "Recall", "F1", "Support"],
                       [[c, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", v["support"]] for c, v in pq["per_class"].items()]))
        P.append(curve_png(pq["curve"], f"{s}: precision vs coverage (primary class)", d / f"curve_{s}.png"))
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
