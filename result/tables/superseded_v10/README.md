# Superseded LM tables (kept for reference; not used by the paper)

These six files were the language-model (LM) outputs of the earlier analysis. They come from
the e6lm2_ (d=128) and e6lmd256_ (d=256) runs, which Amendment 4 (PROTOCOL_frozen.md)
supersedes. In those runs, LM evaluation did not advance the debt counters, and it changed
the schedule state of the fixed-cadence and cycle-shuffle routers.

- `lm_table_v10.csv`: LM table.
- `lm_summary.csv`, `lm_curves.csv`: per-run and per-epoch LM rows. These are copies of the
  frozen ones in result/frozen_2026-08-24/, which also come from e6lm2_.
- `lm_secondary.csv`: the LM part of scripts/analyze-secondary-endpoint.py, computed on
  e6lm2_.
- `numbers_v10.csv`, `extras_v10.csv`: numbers from the earlier scripts, including their LM
  rows.

The paper's LM numbers come from the corrected runs e6lm3_/e6lm3d256_, via
result/tables/lm_v11/, numbers_v11.csv and extras_v11.csv. The same analysis applied to
the superseded grid is in result/tables/lm_v11_superseded_e6lm2/. The vision rows of
numbers_v10.csv and extras_v10.csv are unaffected by Amendment 4.
