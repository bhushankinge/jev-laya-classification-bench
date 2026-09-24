from matplotlib.image import imread
from figures import make_figures as mf

FIGURES = ["hero", "precision_coverage", "e1_variants", "agreement", "per_class_f1", "gate"]


def test_figures_render_from_committed_results(tmp_path):
    written = {p.name: p for p in mf.main(tmp_path)}
    for name in FIGURES:
        for theme in ("light", "dark"):
            assert written[f"{name}-{theme}.png"].stat().st_size > 10_000
    h, w = imread(written["social_preview.png"]).shape[:2]
    assert (w, h) == (1280, 640)                                   # GitHub social preview size
    assert imread(written["hero-dark.png"])[2, 2, :3].mean() < 0.2  # dark variant really has a dark surface


def test_headline_numbers_come_from_committed_metrics():
    e2, e1 = mf.load()
    v = mf.hero_values(e2)
    assert round(v["jev"][0], 3) == 0.919 and round(v["jev"][1], 3) == 0.865
    assert v["qwen"][1] is None and v["laya"][1] is None           # no 95% cutoff reachable
    assert len(e1) == 9
