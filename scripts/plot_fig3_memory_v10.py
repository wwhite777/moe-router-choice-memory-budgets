"""Fig 3 (two-column layout) — accounting-model validation and
device-anchored feasibility.

Same data and logic as an earlier single-column version, with these changes:
  * figure width 6.75 in; fonts sized for print (labels/titles 9 pt,
    ticks/legend 8 pt); saved without bbox cropping (PDF exactly 6.75 in);
  * panel (a) title names the plotted subset (main backbone, b=16, S=64;
    only configurations without in-place ReLU);
  * legend entry "(0/32 > C+)" -> "(<= C+ in all 32 configs)";
  * panel (a) legend split (analytical curves inside, measured below);
    panel (b) device tick labels rotated so they stay at 8 pt without overlapping.

(a) E3 (e1_main, non-inplace, b=16, S=64): analytical Chat and Chat+ vs
    measured live saved-tensor bytes (+2P) and framework-allocated peak
    (median of the 3 repeats in measured_alloc_peak_all), per expert width.
(b) Device-anchored ladder (E4 backbone, b=8, S=16): device SRAM budgets vs
    realized |F| and the max measured virtual peak across all runs.

Data: result/frozen_2026-08-24/{e3_measured_memory.json,s2_summary.csv}.
Output: result/figure/v10/fig3_memory.{pdf,png[,pptx]}.
Every plotted value is printed; lines starting with "VAL|" are full precision.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "result" / "frozen_2026-08-24"
OUT = ROOT / "result" / "figure" / "v10"
OUT.mkdir(parents=True, exist_ok=True)
NAME = "fig3_memory"

plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

e3 = json.load(open(FROZEN / "e3_measured_memory.json"))
n_all = len(e3["blocks"])
n_viol = sum(b["live_model_bytes"] > b["chat_plus"] for b in e3["blocks"])
print(f"E3 all configs: {n_viol}/{n_all} with saved-tensor+2P > Chat+")
blocks = [b for b in e3["blocks"] if b["family"] == "e1_main" and not b["inplace"]]
blocks.sort(key=lambda b: b["f"])
assert all(b["b"] == 16 and b["S"] == 64 for b in blocks)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.9), layout="constrained")
# Larger vertical pad than the default 3 pt: the bottom legend line was
# clipped by ~1.3 pt at the canvas edge with the default padding.
fig.get_layout_engine().set(h_pad=8 / 72)

widths = [b["f"] for b in blocks]
chat = [b["chat"] / 1024 for b in blocks]
chatp = [b["chat_plus"] / 1024 for b in blocks]
saved = [b["live_model_bytes"] / 1024 for b in blocks]
# Median over the 3 repeats: robust to the one-time CUDA/cuBLAS first-touch
# initialization captured in the very first repeat (f=64: 20.8 MB vs 3.4 MB).
alloc = [np.median(b["measured_alloc_peak_all"]) / 1024 for b in blocks]
print(f"── Fig3(a) encoded table (KB, e1_main non-inplace, b=16, S=64, n={len(blocks)}) ──")
for b, w, c, cp, s, a in zip(blocks, widths, chat, chatp, saved, alloc):
    print(f"f={w}: Chat={c:.0f} Chat+={cp:.0f} saved+2P={s:.0f} alloc={a:.0f} "
          f"(repeats KB={[round(x / 1024) for x in b['measured_alloc_peak_all']]})")
    print(f"VAL|fig3|a|f={w}|chat={c!r}|chat_plus={cp!r}|saved={s!r}|alloc={float(a)!r}")

ax1.plot(widths, chat, "--", color="#0072B2", lw=1.4, label="Analytical $\\hat{C}$ (gate)")
ax1.plot(widths, chatp, "-", color="#0072B2", lw=1.4,
         label="Analytical $\\hat{C}_+$ (auditable bound)")
ax1.plot(widths, saved, "o", color="#009E73", ms=4,
         label="Measured saved-tensor + params/grads\n($\\leq \\hat{C}_+$ in all 32 configs)")
ax1.plot(widths, alloc, "^", color="#D55E00", ms=4,
         label="Framework-allocated peak, median of 3 repeats\n(incl. autograd transients — outside the model)")
ax1.set_xscale("log", base=2)
ax1.set_xticks(widths)
ax1.set_xticklabels(widths)
ax1.minorticks_off()
ax1.set_xlabel("Expert hidden width $f$")
ax1.set_ylabel("Memory (KB)")
ax1.set_title("(a) Accounting model vs. measurement\n(main backbone, b=16, S=64)")

# (b) device ladder from e4 rows
e4 = defaultdict(lambda: {"peaks": [], "F": None})
for r in csv.DictReader(open(FROZEN / "s2_summary.csv")):
    if r["grid"] == "e4":
        d = e4[int(r["budget_kb"])]
        d["peaks"].append(int(r["peak_sram_kb"]))
        d["F"] = int(r["expected_F"])
DEVICES = {192: "STM32F407\n192 KB", 256: "MCUNet-class\n256 KB",
           320: "STM32F746\n320 KB", 512: "ESP32-S3\n512 KB",
           1024: "STM32H743\n1 MB", 2048: "i.MX RT1170\n2 MB"}
budgets = sorted(e4)
print("\n── Fig3(b) encoded table ──")
xs = np.arange(len(budgets))
for i, b in enumerate(budgets):
    peak = max(e4[b]["peaks"])
    print(f"budget={b}KB device={DEVICES[b].split(chr(10))[0]} |F|={e4[b]['F']} "
          f"max_measured_peak={peak}KB n_runs={len(e4[b]['peaks'])}")
    print(f"VAL|fig3|b|budget={b}|F={e4[b]['F']}|max_peak={peak}|n={len(e4[b]['peaks'])}")
    ax2.bar(i, b, color="#999999", alpha=0.35, width=0.62,
            label="Device SRAM budget" if i == 0 else None)
    ax2.bar(i, peak, color="#0072B2", alpha=0.85, width=0.38,
            label="Max virtual peak over all runs" if i == 0 else None)
    ax2.text(i, b + 40, f"|F|={e4[b]['F']}", ha="center", va="bottom",
             fontsize=8, fontweight="bold")
print(f"total e4 runs = {sum(len(d['peaks']) for d in e4.values())}")
ax2.set_xticks(xs)
ax2.set_xticklabels([DEVICES[b].replace("\n", ", ") for b in budgets],
                    rotation=30, ha="right", rotation_mode="anchor")
ax2.set_ylabel("KB")
ax2.set_ylim(0, 2450)
ax2.set_title("(b) Device-anchored budgets:\npeak $\\leq$ budget in all 450 runs")
ax2.legend(loc="upper left", frameon=False)

# Panel (a) legend split: the two analytical curves inside the empty upper
# left, the two measured series below the axes (no data occlusion).
h1, l1 = ax1.get_legend_handles_labels()
leg_in = ax1.legend(h1[:2], l1[:2], loc="upper left", frameon=False,
                    handlelength=1.8)
ax1.add_artist(leg_in)
ax1.legend(h1[2:], l1[2:], loc="upper center", bbox_to_anchor=(0.45, -0.3),
           frameon=False, handlelength=1.8)

# No suptitle: caption text lives in LaTeX.
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"{NAME}.{ext}", dpi=300)
w, h = fig.get_size_inches()
print(f"\nsaved {OUT}/{NAME}.pdf/.png  size={w:.2f}x{h:.2f} in")

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Fig. 3 — Accounting model and device-anchored feasibility"
    s.shapes.title.text_frame.paragraphs[0].font.size = Pt(28)
    s.shapes.add_picture(str(OUT / f"{NAME}.png"), Inches(0.4), Inches(1.5),
                         width=Inches(12.533))
    prs.save(OUT / f"{NAME}.pptx")
    print(f"saved {OUT}/{NAME}.pptx")
except ImportError as e:
    print(f"python-pptx not importable ({e}); no PPTX written")
