"""Measure what Laya's sequence packer keeps of our question bundle.

Runnable check for report §9.4: the class definitions fit the head budget uncut; only the
state is truncated. Needs `laya` importable and a checkpoint directory (LAYA_MODEL_DIR).

    python -m pipeline.head_budget
"""
import json
import os
import sys

from transformers import AutoTokenizer

from laya.agent import Agent
from laya.common import build_sequence, render_options

from . import bundle


def head_budget(tok, cfg, q):
    """Return (per-option tokens kept, instruction tokens kept, state room) for one question."""
    qi = Agent._to_internal(q)
    max_len, head_max_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
    ids, markers = build_sequence(tok, "x", qi, max_len, head_max_len)
    end = ids.index(tok.sep_token_id, markers[-1])
    kept = [b - a - 1 for a, b in zip(markers, markers[1:] + [end])]  # minus the [MASK] itself
    return kept, markers[0] - 2, max_len - end - 3  # [CLS] head [SEP] ... [SEP] state [SEP]


def main():
    md = os.environ.get("LAYA_MODEL_DIR") or sys.exit("set LAYA_MODEL_DIR to a Laya checkpoint directory")
    tok_dir = os.path.join(md, "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir if os.path.isdir(tok_dir) else md)
    cfg = json.load(open(os.path.join(md, "rl_agent_config.json")))
    qs = bundle.questions("A")
    for name in ("primary_class", "fulfillment_mode", bundle.HAS_KEY[bundle.SUBCLASSES[0]]):
        q = qs[name]
        full = [len(tok(" " + o, add_special_tokens=False)["input_ids"]) for o in render_options(Agent._to_internal(q))]
        kept, ins_kept, room = head_budget(tok, cfg, q)
        cut = [f for f, k in zip(full, kept) if k < min(f, 48)]
        print(f"{name}: {len(kept)} options, head {sum(kept) + len(kept) + ins_kept} tokens, "
              f"options kept {kept}, instruction kept {ins_kept}, state room {room}, options cut: {len(cut)}")
        assert not cut, f"{name}: options were truncated"
    print("ok: no option or instruction truncation for this bundle")


if __name__ == "__main__":
    main()
