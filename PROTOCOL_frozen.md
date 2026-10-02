# GatedMoE — Frozen Experimental Protocol (S0)
Frozen: 2026-08-18, BEFORE any study run was launched. Amendments require a dated
entry in the project log stating what changed and why; results obtained before an
amendment are never silently re-scored under new rules.

## 1. Endpoints and model selection
- PRIMARY endpoint: final-epoch (epoch 30) top-1 accuracy on the held-out TEST set.
- SECONDARY endpoint: test accuracy at the epoch with best VALIDATION accuracy.
- Validation set: the LAST 5,000 indices (45000–49999) of the canonical CIFAR
  train split, evaluated with test transforms. Fixed, seed-independent.
  Training uses indices 0–44999. The test set is NEVER used for selection.
- "best_test_acc" (max over epochs of test acc) remains in results.json for
  continuity but is BANNED from the manuscript (an earlier draft's inconsistency: final-epoch
  in Table 2 vs best-top-1 in Fig. 2 — resolved here).
- Loss endpoint: test NLL (cross-entropy) reported alongside accuracy.

## 2. Statistics (pre-declared)
- Screening seeds: {42, 123, 7}. Confirmatory seeds add {2026, 31415}.
  Transition cells and the all-feasible cell get +5 seeds
  {60486, 5150, 777, 888, 999} → n=10 where the claim depends on them.
- Paired per-seed differences; Holm-Bonferroni across cells per comparison family.
- EQUIVALENCE: two one-sided tests (TOST) with pre-declared margins:
  CIFAR-10/100: epsilon = 0.005 absolute accuracy (0.5 pp).
  Tiny-ImageNet / WikiText-2 MoE-LM: epsilon = 0.010 (1.0 pp / equivalent ppl margin).
  Sensitivity analysis at 0.5x and 2x the margin.
- Language rules: cells passing TOST are "practically equivalent (TOST, eps=...)";
  cells failing are "not shown equivalent" — NEVER "equivalent".
  Failure to reject a difference is not equivalence. Markers != significance.
  Post-hoc thresholds labeled post-hoc.

## 3. S1 — Controlled choice-set study (identification-clean)
- Question: at FIXED expert identity/capacity, does the NUMBER of admissible
  experts change router differences? (Selection freedom, no capacity confound.)
- Backbone: main d=128 model; K=8 experts, ALL widths 256 (equal cost).
- Budget: fixed 8 MB (continuity with an earlier draft); every expert individually fits; the
  admissible mask is the ONLY constraint. |F| = m by construction.
- Masks: m in {1, 2, 3, 4, 6, 8}; for m < 8 two subsets per m, sampled once
  with rng seed 777 and FROZEN in configs/s1_grid.json (shared across methods
  and seeds); m=8 is the full set. 11 mask-cells.
- Methods (4): round_robin, isolated random (own RNG stream), gatedmoe_base
  (Debt-Base), learned_top1 with the same admissible mask (logits masked
  before softmax; aux loss over admissible only).
  Debt-Grad is excluded from S1 (appendix/S2 only — too close to Debt-Base).
- Dataset: CIFAR-100 (harder task = more headroom for differences).
- Matched comparisons: same seed => identical model init and identical batch
  order across methods (dedicated DataLoader generator; router RNG isolated).

## 4. S2 — Natural budget sweep (ecological validity)
- Heterogeneous widths [64,96,128,192,256,384,512,1024]; budget grid realizing
  |F| = 0..8 at b=16 (configs/e1_grid.json); C10 + C100; 5 methods
  (round_robin, gatedmoe_base, gatedmoe_g, learned_top1, random).
- Interpretation rule (pre-declared): S2 gaps CONFOUND selection freedom with
  capacity access; causal statements about "routing freedom" come from S1 only.
  S2 answers "does the budget knob change outcomes in realistic heterogeneous
  pools"; the capacity-composition of F at each budget is reported.

## 5. E4 — Device-anchored MCU backbone
- d=32, patch 8, b=8; budgets {192K STM32F407, 256K MCUNet-class, 320K
  STM32F746, 512K ESP32-S3, 1M STM32H743, 2M i.MX RT1170}; CIFAR-10.
- Wording rule: "memory scales corresponding to representative embedded
  systems" — never "validated MCU training".

## 6. Diagnostics logged per run (S1/S2/E4)
- Per-step per-layer chosen expert (routes_train.npy, int8, -1 = null path).
- Per-layer expert x class activation-count matrix (route-class MI, per-expert
  class specialization).
- Feasible-set histogram per epoch; gate-rejection fraction; nontrivial-decision
  fraction (|F| >= 2); utilization entropy (debt_stats).
- Val/test accuracy + NLL per epoch.
- Pairwise route-disagreement is computed post-hoc from routes_train.npy between
  methods matched on (cell, seed).

## 7. E3 — Framework-level tensor-memory validation (naming rule)
- Three concepts reported separately, never merged: (1) analytical model bytes
  (Chat, Chat+), (2) live saved-tensor bytes (autograd-graph walk), (3) framework
  allocated AND reserved (torch.cuda.memory_stats, peak reset per phase,
  baseline context subtracted, 3 repeats).
- Violation rule: measured saved-tensor peak must be <= Chat+ in 100% of cells;
  any violation => the formula is revised and the gate RERUN, not excused.
- This is "framework-level tensor-memory validation", NOT hardware validation.

## 8. Decision gate G-SM1 (Aug 27, on partial S1+S2)
(a) Router gaps grow with m in S1 (replicated at n>=10 boundary seeds, Holm+TOST)
    => threshold-study paper with positive guideline.
(b) Flat everywhere incl. m=8 with TOST-equivalence => strong controlled negative;
    E5 + E6-LM become load-bearing for generality.
(c) Ambiguous => add boundary seeds; if still ambiguous, write (b) with the
    ambiguity stated. No manufactured wins.
Submission gate (all three required): identification-clean S1; replicated
transition OR narrow-CI equivalence at high m; zero unexplained Chat+ violations.

## Amendment 3 (2026-08-26, pre-registered BEFORE these runs; basis: external
## reviews of an earlier draft)
New arms (LM unless noted), all with primary endpoint final test ppl and the
declared LM margin (1% of the round-robin cell mean):
- A3.1 learned_top1_aux0: learned top-1 with auxiliary-loss weight 0, |F| in
  {4, 8}, n=10. Deconfounds irregularity from the aux loss: if it lands near
  random, irregularity carries learned's deficit; if near learned, aux is a
  co-culprit.
- A3.2 cycle_shuffle: a fresh random permutation each cycle (low gap-CV,
  between fixed-permutation and i.i.d.), |F|=8, n=10 — the intermediate point
  of the irregularity dose-response.
- A3.3 d=256 replication: {round_robin, fixed_cadence_random, random,
  learned_top1} at the all-feasible budget only, n=10, d=256/f_s=512, same
  expert pool. Tests whether the cadence effect survives a 4x-larger backbone.
- A3.4 (analysis, no runs): irregularity is MEASURED as per-expert
  inter-activation-gap coefficient of variation from existing route traces;
  the dose-ordering claim binds to this statistic.
- A3.5 (analysis, no runs): token-routing comparison re-scoped. Analytic
  result: no capacity factor complies below shared+sum(2P_e)+K bytes
  (= 9.2 MB here); at the sole compliant budget the existing control IS
  memory-matched and knapsack-vs-capacity is tested paired with the failed
  run included. "Parity at a discount" language is withdrawn.
- A3.6 (analysis, no runs): S1 subset-as-random-factor analysis; across-m
  trend language narrowed to within-cell claims unless subset effect is
  negligible. Equivalence reporting moves to three categories
  (equivalent / different / unresolved) everywhere.

## Amendment 4 (2026-09-30, declared BEFORE these runs; basis: an independent code audit)
Defects found by the audit in the LM code path (vision is unaffected: its evaluation already advances the debt copy,
and it has no position/permutation routers):
- (1) LM evaluation never advanced the evaluation copy of the debt counters, so the debt-based arms (gatedmoe_base,
  gatedmoe_g) evaluated every window with the expert chosen at the start of evaluation, while round-robin cycled.
- (2) LM evaluation saved/restored only `current` and the RNG state of batch-level routers; it advanced the
  position/permutation of fixed_cadence_random and cycle_shuffle and thereby changed their later TRAINING routes.
- (3) The logged LM `train_ppl` of learned arms exponentiated the objective including the 0.01-weighted balancing
  penalty, not task cross-entropy.
Fix (gatedmoe/lm_train_loop.py): eval_ppl snapshots and restores `current`, `pos`, `order` and the RNG state of every
router and steps the evaluation debt copy after each window, exactly as vision evaluation does; `train_ppl` = task
cross-entropy for all arms, `train_obj_ppl` = the optimized objective. Regression tests: test/test_lm_eval_invariance.py
(state unchanged by evaluation; training routes invariant to inserted evaluation; uniform-debt evaluation routes equal
round-robin; the pre-A4 function fails all three).
Rerun: the complete LM grid with identical methods, budgets, seeds and hyperparameters, configs/e6lm3_grid.json —
e6lm3_* (200 runs: 8 arms at their budgets x 10 seeds) and e6lm3d256_* (40 runs) — in a dedicated environment
(Python 3.10, torch 2.10.0+cu128; the original runs used an unrecorded PyTorch 2.x build). Endpoints and analyses are
unchanged: final-epoch test perplexity (primary), test perplexity at the best-validation epoch (secondary); Table-1
family = Holm over the 12 tests {Debt-Base, Debt-Grad, random, learned_top1} x 3 budgets vs round-robin; intervention
contrasts Holm over three; LM equivalence margin 1% of the round-robin cell mean (specified in Amendment 3). The
corrected runs become the paper's LM evidence; the e6lm2_/e6lmd256_ runs stay unchanged on disk and are reported as
superseded (defects 1-2 affect their debt, fixed-cadence and cycle-shuffle arms). No vision run is repeated.

## Amendment 5 (2026-10-01, declared BEFORE these runs; basis: two external reviews of the manuscript)
Purpose: test (a) whether a router that receives the task-loss gradient changes the router comparisons, and (b) whether
the routed experts can improve on the shared-only model at this scale. Code paths, hyperparameters, data, epochs and
seeds are unchanged; seeds {7, 42, 123, 777, 888, 999, 2026, 5150, 31415, 60486} everywhere. Environment: the
Amendment-4 venv (Python 3.10.12, torch 2.10.0+cu128) plus torchvision 0.25.0+cu128 and pillow 11.3.0, added on
2026-10-01 for the vision data path (torch unchanged). Grid: configs/a5_grid.json (52 jobs).
New arms:
- A5.1 gated_top1, LM d=128, budgets B24457k (|F|=4) and B42487k (|F|=8); 20 runs, names e6lm5_*. gated_top1 equals
  learned_top1 (input = batch- and token-mean hidden state; softmax over admissible experts; highest-probability feasible
  expert; balancing penalty sum(p^2), weight 0.01) EXCEPT that the chosen expert's output is multiplied by its routing
  probability (Switch-style), so the router receives the task-loss gradient. The weighting also scales the expert output
  (by p, about 1/8 at initialization); this is part of the arm and is reported as such.
- A5.2 gated_top1, vision S2 main backbone, CIFAR-100, budgets B7333k (|F|=4) and B17350k (|F|=8); 20 runs (e1a5_*).
- A5.3 dense ceiling, vision S2 main backbone, CIFAR-100, B17350k configuration; DenseRouter: all admissible experts every
  step, outputs averaged, memory gate ignored (it exceeds the budget by design: a capacity diagnostic, not a budgeted
  method); 10 runs (e1a5_*).
- Build check (2 runs, e1a5chk_*; queued first): e1_c100_round_robin_B17350k_s42 and e1_c100_learned_top1_B17350k_s42
  rerun unchanged in the current setup (PyTorch build, data-loader settings and code state together; the August runs did
  not record the loader worker count). If any endpoint of either differs from the original, the vision comparators of A5.2/A5.3
  (round_robin and learned_top1 at B7333k and B17350k, round_robin at B3907k; 50 runs) are rerun in the current
  environment BEFORE any A5.2/A5.3 contrast is computed, and the contrasts use the rerun comparators.
Endpoints: unchanged (primary = final-epoch test metric; secondary = test metric at the best-validation epoch).
Tests (paired by seed, two-sided paired t; Holm within each family, separately per endpoint):
- LM family (4): gated_top1 vs round_robin and vs learned_top1 at |F|=4 and |F|=8, comparators = e6lm3_* runs.
- Vision family (4): the same four contrasts, comparators = e1_c100_* runs (or their reruns, see build check).
- Dense family (2): dense vs round_robin at B3907k (|F|=0, shared-only) and vs round_robin at B17350k (|F|=8).
- Equivalence: TOST of gated_top1 vs round_robin at the existing margins (vision 0.005 accuracy; LM 1% of the
  round-robin cell mean).
- Sensitivity, declared now: two-sided sign test on the same paired differences (the LM differences are bimodal).
- Diagnostics (descriptive): per-seed best-validation epoch and the number of seeds whose best is the last epoch;
  gap-CV and utilization entropy of the gated arm.
Reporting: every result is reported whatever its direction. These arms add evidence; they change no existing endpoint,
family, run or headline number, and no further amendment will be made to them before the paper is submitted.
