# Ablation experiment plan (paper §4)

This file documents the six ablation variants from the paper §4 plus the
recipe to reproduce each of them. None of these variants were run in this
offline reproduction (no Porto, no LibCity-Datasets, no GPU). The plan
below is suitable for an AutoDL RTX-4090 session.

## Variants

| ID | Variant | What to change in `porto.json` / model code |
|----|---------|----------------------------------------------|
| (full) | TrajRL | use defaults from `porto.json` |
| A | w/o MFTime-Enc | set `add_time_in_day=false`, `add_day_in_week=false`, `add_time_interval=false`; in `libcity/model/trajectory_embedding/trajrl.py` `TrajRL.__init__` keep `self.embedding = NodeEmbedding(... add_time_in_day=False, add_day_in_week=False, add_time_interval=False ...)` |
| B | w/o Time Interval | only `add_time_interval=false`, keep period/dow encodings |
| C | w/o MSTraj-Enc | replace multi-scale encoder with a single Transformer encoder: e.g. comment out scale_blocks 1..3, change `self.e_blocks = 1` so only block 0 (vanilla Transformer) runs |
| D | w/o InPatch-Enc | inside `Encoder`, replace `PatchGRUEncoder` by `nn.Linear(d_model, d_model)` — see comment in `libcity/model/layers/basics.py` |
| E | w/o TrajRec | set `mlm_ratio=0` in `PatchContrastiveSplitMLMExecutor.json` |
| F | w/o CL | set `contra_ratio=0` in the same JSON |

## CLI invocations for each variant (everything else from the README kept)

```bash
# A. w/o MFTime-Enc
python run_model.py \
  --model TrajRLContrastiveLM --dataset porto --config porto \
  --gpu_id 0 --split true --masking_ratio 0.2 \
  --distribution geometric \
  --out_data_argument1 trun --out_data_argument2 shift \
  --add_time_in_day false --add_day_in_week false --add_time_interval false

# B. w/o Time Interval
# same, but only --add_time_interval false

# C. w/o MSTraj-Enc
# edit libcity/model/trajectory_embedding/trajrl.py:
#   self.e_blocks = 1   # only block 0 runs
# and re-launch

# D. w/o InPatch-Enc
# edit libcity/model/layers/sequencepooler.py or replace PatchGRUEncoder
# in Encoder with nn.Linear(d_model, d_model) (operating on bs*num_patch, d_model)

# E. w/o TrajRec
# pass --mlm_ratio 0  (defaults to the executor JSON's 1.0)

# F. w/o CL
# pass --contra_ratio 0 (defaults to 1.0)
```

## Where the qualitative claims live (paper §4)

> • The variant "w/o MFTime-Enc" performs the worst, demonstrating the importance of the multi-faceted time encoder.
> • "w/o MSTraj-Enc" shows a clear drop, indicating the multi-scale encoder is more suitable than a vanilla Transformer for trajectory spatio-temporal features.
> • "w/o TrajRec" and "w/o CL" both degrade performance.

The expected ordering (best → worst), per the paper:
full > B ≈ E ≈ F > C > D > A.

The numeric ablation table itself was **not** captured in the public
transcript I located; reproducing the numbers requires a full GPU run.
