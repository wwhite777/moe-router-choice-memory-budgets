"""E3 runner: measured-vs-analytical memory table for both backbone families.

Writes result/e3_measured_memory.json and prints the violation/slack table.
Run on any free GPU: CUDA_VISIBLE_DEVICES=0 python3 scripts/run-measured-memory.py
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatedmoe.sim_measured_memory import measure_block, measure_layer
from gatedmoe.util_config_io import RESULT_DIR

FAMILIES = [
    # (name, b, S, d, f_shared, expert widths)
    ("e1_main", 16, 64, 128, 256, [64, 96, 128, 192, 256, 384, 512, 1024]),
    ("e4_mcu", 8, 16, 32, 64, [32, 48, 64, 96, 128, 192, 256, 512]),
]


def main():
    out = {"blocks": [], "layers": []}
    for name, b, S, d, fs, widths in FAMILIES:
        for inplace in (False, True):
            for f in widths:
                r = measure_block(b, S, d, f, inplace=inplace)
                r["family"] = name
                out["blocks"].append(r)
                v_chat = r["measured_alloc_peak"] > r["chat"]
                v_plus = r["measured_alloc_peak"] > r["chat_plus"]
                v_saved = r["live_model_bytes"] > r["chat_plus"]
                print(f"[{name} inplace={int(inplace)}] f={f:5d} "
                      f"Chat={r['chat']/1024:8.0f}K Chat+={r['chat_plus']/1024:8.0f}K "
                      f"alloc={r['measured_alloc_peak']/1024:8.0f}K "
                      f"saved+2P={r['live_model_bytes']/1024:8.0f}K "
                      f"viol[Chat/{'+' if v_plus else '-'}Chat+/saved]="
                      f"{int(v_chat)}/{int(v_plus)}/{int(v_saved)}", flush=True)
            # Layer-level: smallest admitted, mid, largest expert
            for f in (widths[0], widths[len(widths) // 2], widths[-1]):
                r = measure_layer(b, S, d, fs, f, inplace=inplace)
                r["family"] = name
                out["layers"].append(r)
                v = r["measured_alloc_peak"] > r["analytic_layer_bound"]
                print(f"[{name} inplace={int(inplace)}] LAYER f={f:5d} "
                      f"bound={r['analytic_layer_bound']/1024:8.0f}K "
                      f"alloc={r['measured_alloc_peak']/1024:8.0f}K viol={int(v)}",
                      flush=True)

    path = Path(RESULT_DIR) / "e3_measured_memory.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    blocks = out["blocks"]
    n = len(blocks)
    v_plus = sum(1 for r in blocks if r["measured_alloc_peak"] > r["chat_plus"])
    v_saved = sum(1 for r in blocks if r["live_model_bytes"] > r["chat_plus"])
    print(f"\nSummary: {n} block cells; alloc-peak>Chat+ in {v_plus}; "
          f"saved-tensor(+2P)>Chat+ in {v_saved}. Saved to {path}")


if __name__ == "__main__":
    main()
