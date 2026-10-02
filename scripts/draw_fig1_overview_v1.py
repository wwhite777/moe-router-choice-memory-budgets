"""Figure 1 (overview schematic) as vector art, v1 (2026-10-02).

Draws the paper's overview -- (a) the budget-gated MoE block, (b) the five study components, (c) the main takeaways --
as a vector PDF with embedded TrueType fonts (resolution-independent). Content follows the paper (Sections 3-6); the
takeaways in (c) carry the scope stated in the abstract and results. Requires matplotlib (drawn with 3.10.5).

Usage: python3 scripts/draw_fig1_overview_v1.py <out_dir>
Writes <out_dir>/Figure_1.pdf and <out_dir>/Figure_1_preview.png (400 dpi). Fails (exit 1) if any text label extends
outside the box that contains it or outside the figure, so a layout change cannot silently clip or overlap text.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Circle, Rectangle

W, H = 6.75, 4.55                                   # inches; included at \textwidth (scale ~0.8)
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

NAVY = "#1f3b73"
BLUE, BLUE_F = "#2b5ca8", "#e3edf8"
GREEN, GREEN_F = "#2e7d32", "#dcefd8"
GRAY, GRAY_F = "#8a8a8a", "#e9e9e9"
ORANGE, ORANGE_F = "#d9661f", "#fde6cf"
PURPLE, PURPLE_F = "#6a3d9a", "#f3ecf8"
TEAL = "#1f7f86"
INK = "#222222"

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
CHECKS = []                                         # (text artist, container (x0, y0, x1, y1) or None, label)


def box(x0, y0, w, h, fc="white", ec=INK, lw=0.8, ls="-", r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                lw=lw, ls=ls, zorder=z))
    return (x0, y0, x0 + w, y0 + h)


def text(x, y, s, size=6.6, color=INK, weight="normal", ha="center", va="center", inside=None, label=None,
         ls=1.25, z=5, style="normal"):
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z,
                linespacing=ls, fontstyle=style)
    CHECKS.append((t, inside, label or s[:40]))
    return t


def arrow(x0, y0, x1, y1, color=INK, lw=0.8, head=5.5, z=3, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                 zorder=z, shrinkA=0, shrinkB=0, linestyle=ls))


def line(xs, ys, color=INK, lw=0.8, z=3):
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="butt")


def panel(x0, x1, title):
    box(x0, 0.04, x1 - x0, H - 0.08, fc="white", ec="#9a9a9a", lw=0.7, r=0.06, z=0)
    text((x0 + x1) / 2, H - 0.20, title, size=9, color=NAVY, weight="bold", inside=(x0, 0.04, x1, H - 0.04))


# ---------------------------------------------------------------- (a) budget-gated MoE block
AX0, AX1 = 0.04, 2.98
panel(AX0, AX1, "(a) Budget-gated MoE block")
L, R = 0.70, 2.90                                   # left/right edge of the main column in (a)

# gate inputs: five small boxes
inputs = [("live memory", r"$M_t$"), ("budget", r"$B_{\mathrm{SRAM}}$"), ("batch, seq.", r"$(b,S)$"),
          ("width", r"$(d,f)$"), ("dtype", r"$\tau$")]
widths = [0.64, 0.50, 0.58, 0.44, 0.38]
gap = (R - 0.12 - sum(widths)) / (len(widths) - 1)
x = 0.12
y_in0, y_in1 = 3.80, 4.13
for (a, b), w in zip(inputs, widths):
    c = box(x, y_in0, w, y_in1 - y_in0, fc="#f6f6f6", ec="#555555", lw=0.7, r=0.04)
    text(x + w / 2, (y_in0 + y_in1) / 2, a + "\n" + b, size=6.2, inside=c)
    arrow(x + w / 2, y_in0, x + w / 2, 3.62, lw=0.7, head=4.5)
    x += w + gap

# analytical memory gate
GL = 0.12                                           # the gate spans the whole column, under all five inputs
g = box(GL, 3.06, R - GL, 0.56, fc=BLUE_F, ec=BLUE, lw=1.0)
text((GL + R) / 2, 3.52, "Analytical memory gate", size=7.2, color=NAVY, weight="bold", inside=g)
text((GL + R) / 2, 3.37, r"$\hat{C}_e = 2P_e + A_e$", size=7.0, inside=g)
text((GL + R) / 2, 3.195, r"expert $e$ is feasible if $M_t + \hat{C}_e \leq B_{\mathrm{SRAM}}$", size=6.6, inside=g)

# input mini-batch on the left, feeding the gate and the shared FFN
xi = box(0.10, 2.50, 0.54, 0.40, fc=BLUE_F, ec=BLUE, lw=0.9)
text(0.37, 2.70, "input\nmini-batch\n" + r"$x_t$", size=6.2, color=NAVY, inside=xi)
arrow(0.37, 2.90, 0.37, 3.06, color=BLUE)

# feasible set into the expert pool
arrow((L + R) / 2, 3.06, (L + R) / 2, 2.92)
text((L + R) / 2 + 0.07, 2.99, r"$\mathcal{F}$ (feasible set)", size=6.4, ha="left", inside=(L, 2.92, R, 3.06))
pool = box(L, 2.12, R - L, 0.80, fc="white", ec="#666666", lw=0.8, ls=(0, (3, 2)))
text((L + R) / 2, 2.835, r"$K=8$ routable experts", size=6.4, inside=pool)
text((L + R) / 2, 2.205, r"each expert: Linear$\to$ReLU$\to$Linear", size=6.0, color="#444444", inside=pool)
ew, n = 0.21, 8
eg = (R - L - 0.12 - n * ew) / (n - 1)
for k in range(n):
    ex = L + 0.06 + k * (ew + eg)
    feas = k < 5
    c = box(ex, 2.33, ew, 0.40, fc=GREEN_F if feas else GRAY_F, ec=GREEN if feas else GRAY, lw=0.8, r=0.03)
    text(ex + ew / 2, 2.53, rf"$E_{k + 1}$", size=6.8, color=INK if feas else "#666666", inside=c)

# router
RW = 1.46                                           # router box width; the note takes the rest of the column
arrow(L + RW / 2, 2.12, L + RW / 2, 1.99)
rt = box(L, 1.45, RW, 0.54, fc=PURPLE_F, ec=PURPLE, lw=1.0)
text(L + RW / 2, 1.885, r"router: chooses within $\mathcal{F}$ only", size=6.0, color=PURPLE, weight="bold", inside=rt)
for (s, cx, cy) in [("cyclic", 0.25, 1.72), ("random", 0.75, 1.72), ("debt-based", 0.25, 1.545), ("learned top-1", 0.75, 1.545)]:
    text(L + cx * RW, cy, s, size=6.2, inside=(L + (cx - 0.25) * RW, 1.45, L + (cx + 0.25) * RW, 1.80))
line([L + RW / 2, L + RW / 2], [1.49, 1.79], color="#b59ccf", lw=0.6)
line([L + 0.06, L + RW - 0.06], [1.633, 1.633], color="#b59ccf", lw=0.6)
note = box(L + RW + 0.05, 1.45, R - L - RW - 0.05, 0.54, fc="white", ec=PURPLE, lw=0.7, ls=(0, (2, 1.5)))
text(L + RW + 0.05 + (R - L - RW - 0.05) / 2, 1.72, "static budget:\ndebt-based\n" + r"$\equiv$ round-robin", size=5.9,
     color=PURPLE, inside=note, ls=1.15)

# select one expert; selected expert, shared FFN, sum, output
sx, sy = L + RW / 2, 0.56                           # router, selected expert, sum and output share one axis
arrow(sx, 1.45, sx, 1.22)
text(sx + 0.06, 1.335, r"select exactly one $e^{*} \in \mathcal{F}$", size=6.3, ha="left", inside=(sx, 1.22, R, 1.45))
se = box(sx - 0.21, 0.80, 0.42, 0.42, fc=ORANGE_F, ec=ORANGE, lw=1.0, r=0.04)
text(sx, 1.01, r"$e^{*}$" + "\nselected", size=6.2, inside=se)
sf = box(0.10, 0.80, 0.66, 0.42, fc=BLUE_F, ec=BLUE, lw=1.0, r=0.04)
text(0.43, 1.01, "shared FFN\n(always on)", size=6.2, color=NAVY, weight="bold", inside=sf)
line([0.37, 0.37], [2.50, 1.26], color=BLUE)
arrow(0.37, 1.26, 0.37, 1.22, color=BLUE)
ax.add_patch(Circle((sx, sy), 0.075, fc="white", ec=INK, lw=0.8, zorder=4))
text(sx, sy - 0.004, "+", size=8, weight="bold", inside=(sx - 0.075, sy - 0.075, sx + 0.075, sy + 0.075))
arrow(sx, 0.80, sx, sy + 0.075)
line([0.43, 0.43], [0.80, sy], color=BLUE)
arrow(0.43, sy, sx - 0.075, sy, color=BLUE)
arrow(sx, sy - 0.075, sx, 0.40)
ob = box(sx - 0.30, 0.14, 0.60, 0.26, fc=BLUE_F, ec=BLUE, lw=0.9, r=0.04)
text(sx, 0.27, r"output $y_t$", size=6.4, color=NAVY, inside=ob)
ck = box(L + 1.30, 0.14, R - L - 1.30, 0.94, fc="white", ec=BLUE, lw=0.7, ls=(0, (2, 1.5)))
text(L + 1.30 + (R - L - 1.30) / 2, 0.61, "layer-local\nactivation\ncheckpointing\n(accounting model):\nlive memory\nresets each layer",
     size=5.9, color=NAVY, inside=ck, ls=1.15)

# ---------------------------------------------------------------- (b) study design
BX0, BX1 = 3.04, 4.92
panel(BX0, BX1, "(b) Study design")
hub = box(BX0 + 0.08, 3.84, BX1 - BX0 - 0.16, 0.36, fc="#f6f6f6", ec=NAVY, lw=0.8, r=0.08)
text((BX0 + BX1) / 2, 4.02, "one frozen protocol,\nmatched comparisons", size=6.2, color=NAVY, weight="bold", inside=hub, ls=1.15)
studies = [
    ("S1", "Controlled choice set", "#1f4e9c", "8 equal-cost experts;\nfixed 8 MB budget;\n" + r"vary $m \in \{1,2,3,4,6,8\}$"),
    ("S2", "Natural budget sweep", GREEN, "heterogeneous expert widths;\n" + r"budgets realize $|\mathcal{F}|=0,\ldots,8$;" + "\ntwo CIFAR vision backbones"),
    ("S3", "LM arm", ORANGE, "4-layer causal transformer,\n" + r"same gate; $|\mathcal{F}| \in \{0,4,8\}$"),
    ("S4", "Token-level routing", PURPLE, "memory-knapsack gate vs.\ncapacity-factor control"),
    ("S5", "Accounting audit", TEAL, "tensor-level validation of\nthe analytical bound;\n32/32 configurations"),
]
top, bot, sg = 3.76, 0.12, 0.07
sh = (top - bot - sg * (len(studies) - 1)) / len(studies)
for i, (tag, title, col, body) in enumerate(studies):
    y1 = top - i * (sh + sg)
    y0 = y1 - sh
    c = box(BX0 + 0.08, y0, BX1 - BX0 - 0.16, sh, fc="white", ec=col, lw=1.0)
    ax.add_patch(FancyBboxPatch((BX0 + 0.14, y1 - 0.27), 0.22, 0.20, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=col, ec=col, zorder=2))
    text(BX0 + 0.25, y1 - 0.17, tag, size=6.6, color="white", weight="bold",
         inside=(BX0 + 0.14, y1 - 0.27, BX0 + 0.36, y1 - 0.07))
    text(BX0 + 0.42, y1 - 0.17, title, size=6.8, color=col, weight="bold", ha="left", inside=c)
    text(BX0 + 0.42, y1 - 0.30, body, size=6.2, ha="left", va="top", inside=c, ls=1.2)

# ---------------------------------------------------------------- (c) main takeaways
CX0, CX1 = 4.98, 6.71
panel(CX0, CX1, "(c) Main takeaways")
take = [
    ("1", "Hard budget first", BLUE,
     "The memory gate determines\nthe feasible set " + r"$\mathcal{F}$" + " before\nthe router acts."),
    ("2", "Vision regime", GREEN,
     "Schedulers reach similar\naccuracy by different routes;\na task-trained gate (main\nbackbone, CIFAR-100,\n"
     + r"$|\mathcal{F}|=4, 8$" + ") is better but\nnear the shared path."),
    ("3", "Small LM regime", ORANGE,
     r"($d=128$, all experts feasible)" + "\nRandom and learned routers\nend training worse, but not\nat the best-validation\ncheckpoint."),
    ("4", "Practical guidance", PURPLE,
     "First check that the routed\npath helps (here little or not\nat all); in the LM, prefer\nregular schedules or\nvalidation-based stopping."),
]
nl = [t[3].count("\n") + 1 for t in take]
lh = 0.108                                          # inches per body line at 6.2 pt
heights = [0.29 + k * lh for k in nl]
ctop, cgap = 4.16, 0.07
y1 = ctop
for (num, lead, col, body), hh in zip(take, heights):
    y0 = y1 - hh
    c = box(CX0 + 0.07, y0, CX1 - CX0 - 0.14, hh, fc="white", ec=col, lw=1.0)
    ax.add_patch(Circle((CX0 + 0.22, y1 - 0.15), 0.085, fc=col, ec=col, zorder=2))
    text(CX0 + 0.22, y1 - 0.153, num, size=6.6, color="white", weight="bold",
         inside=(CX0 + 0.135, y1 - 0.235, CX0 + 0.305, y1 - 0.065))
    text(CX0 + 0.36, y1 - 0.15, lead, size=6.8, color=col, weight="bold", ha="left", inside=c)
    text(CX0 + 0.14, y1 - 0.27, body, size=6.2, ha="left", va="top", inside=c, ls=1.2)
    y1 = y0 - cgap
lg = box(CX0 + 0.07, 0.12, CX1 - CX0 - 0.14, y1 - 0.12 + cgap - 0.06, fc="white", ec="#888888", lw=0.6,
         ls=(0, (2, 1.5)))
ly = [lg[3] - 0.13 - i * 0.135 for i in range(3)]
for yy, (fc, ec, s) in zip(ly, [(GREEN_F, GREEN, "feasible expert"), (GRAY_F, GRAY, "infeasible expert"),
                                (ORANGE_F, ORANGE, "selected expert")]):
    ax.add_patch(Rectangle((CX0 + 0.17, yy - 0.045), 0.14, 0.09, fc=fc, ec=ec, lw=0.8, zorder=2))
    text(CX0 + 0.38, yy, s, size=6.2, ha="left", inside=lg)


# ---------------------------------------------------------------- layout self-check (fail closed)
def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad, n = [], 0
    for t, cont, lab in CHECKS:
        bb = t.get_window_extent(renderer=r)
        (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        n += 1
        lim = cont or (0, 0, W, H)
        pad = 0.012
        if x0 < lim[0] + pad or x1 > lim[2] - pad or y0 < lim[1] + pad * 0.5 or y1 > lim[3] - pad * 0.5:
            bad.append(f"{lab!r}: text ({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f}) vs box {tuple(round(v, 3) for v in lim)}")
    if n == 0:
        sys.exit("layout check: no text checked")
    if bad:
        print("\n".join(bad))
        sys.exit(f"layout check FAILED: {len(bad)} of {n} labels outside their boxes")
    fig.savefig(out / "Figure_1.pdf")
    fig.savefig(out / "Figure_1_preview.png", dpi=400)
    print(f"layout check PASS: {n} labels inside their boxes; wrote {out / 'Figure_1.pdf'} and the 400-dpi preview")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
