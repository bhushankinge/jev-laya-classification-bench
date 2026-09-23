"""Run the extended Qwen labeler into labels/qwen/v2.jsonl.

    python3 -m pipeline.label_qwen 5      # smoke
    python3 -m pipeline.label_qwen        # all remaining rows
"""
import os, runpy, sys
from . import common

os.environ["QWEN_LABELS"] = str(common.LABELS_DIR / "qwen" / "v2.jsonl")
os.environ.setdefault("CONC", "48")
(common.LABELS_DIR / "qwen").mkdir(parents=True, exist_ok=True)
sys.argv = ["qwen_classify.py"] + sys.argv[1:]
runpy.run_path(str(common.ROOT / "qwen_discovery" / "qwen_classify.py"), run_name="__main__")
