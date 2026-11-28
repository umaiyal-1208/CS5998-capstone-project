# GRU: the first sequence model

Made by `scripts/train_gru.py` on the padded sequences from
`scripts/build_sequences.py` (13 months x 289 inputs + 11 category codes per
customer). Trained on the same 321,239 training customers as the baselines
and scored on the same 68,837 validation customers. 41 minutes on a laptop CPU
(4 threads), 249,705 parameters. The test set was not loaded.

## Results

| Model | Score | 95% CI |
|---|---|---|
| Logistic regression (baseline) | 0.7808 | 0.7732 - 0.7870 |
| LightGBM (best baseline) | 0.7907 | 0.7849 - 0.7966 |
| GRU | 0.7862 | 0.7779 - 0.7925 |
| **Blend of GRU and LightGBM (mean rank)** | **0.7943** | **0.7882 - 0.7997** |

## What the results show

1. **On its own, the GRU does not beat LightGBM.** It scores 0.7862, which is
   0.0048 below LightGBM. The paired bootstrap puts the gap between -0.0090
   and -0.0009, so the difference is small but real. This is the outcome the
   Project Definition named as the main model-performance risk: gradient
   boosting is very strong on tabular data.

2. **But the GRU gets close from raw statements alone.** LightGBM works from
   1,085 hand-built summaries (last, mean, min, max, std and change of every
   feature). The GRU got no summaries at all, only the months in order, and
   learned its own representation. It already beats logistic regression on
   those summaries (0.7862 against 0.7808).

3. **Blending the two beats either one, and that gain is real.** Simply
   averaging the two models' rankings scores 0.7943, which is +0.0038 above
   LightGBM. The paired bootstrap interval, +0.0011 to +0.0060, stays above
   zero, and the blend wins in all 200 resamples. Nothing was tuned for the
   blend (equal weights, no search), so this is not an artefact of fitting to
   the validation set. The two models make different mistakes, which means
   the GRU has learned something from the order of the statements that the
   summary features do not capture. This is the clearest answer so far to the
   project's question, and it is why the ensemble is in the plan.

4. **The GRU does best where histories are short.** For customers with 1 to 5
   statements it scores 0.6259 against LightGBM's 0.6152 (+0.0107). For full
   13-month histories it is slightly behind (-0.0033), and for 6 to 12 months
   further behind (-0.0093). These groups are small (3,904 and 6,912
   customers) and have no confidence intervals yet, so this is a lead to test,
   not a finding. A plausible reason: with few statements, summary statistics
   such as std and change are empty or meaningless, while the GRU just reads
   what is there.

5. **The GRU starts to overfit after about 6 epochs.** Training loss keeps
   falling (0.2397 to 0.2038), but the validation score peaks at epoch 6
   (0.7862) and then drifts down even after the learning rate is halved. Early
   stopping kept the epoch 6 weights. More regularisation (dropout, weight
   decay) and tuning are the obvious next steps.

## What comes next

- **Transformer encoder:** attention over all 13 months at once, with each
  statement treated as a token, as planned in the Project Definition.
- **Tune both models with Optuna**, on the validation set only.
- **Proper ensemble:** blend weights chosen by cross-validation rather than
  fixed at 50/50.
- **Confidence intervals by history length**, to test finding 4.
- **Final comparison on the untouched test set**, once, at the end.

## Caveats

- The GRU's best epoch was chosen on the validation set, as LightGBM's number
  of trees was, so both validation scores are slightly optimistic in the same
  way. The test set will give the unbiased comparison.
- Only one seed was run. Neural network results vary a little between seeds;
  the final comparison will average several.

---

## Script output

Train 321,239 customers, validation 68,837; 13 months x 289 inputs + 11 categories (4s to load). 4 CPU threads.

GRU with 249,705 parameters.

| Epoch | Train loss | Valid score | Gini | Top 4% | LR | Minutes |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2397 | 0.7810 | 0.9179 | 0.6441 | 1e-03 | 5.9 |
| 2 | 0.2269 | 0.7840 | 0.9193 | 0.6487 | 1e-03 | 4.9 |
| 3 | 0.2230 | 0.7852 | 0.9207 | 0.6497 | 1e-03 | 5.7 |
| 4 | 0.2203 | 0.7830 | 0.9202 | 0.6458 | 1e-03 | 4.8 |
| 5 | 0.2182 | 0.7851 | 0.9208 | 0.6495 | 1e-03 | 4.2 |
| 6 | 0.2130 | 0.7862 | 0.9213 | 0.6510 | 5e-04 | 6.5 |
| 7 | 0.2106 | 0.7842 | 0.9202 | 0.6481 | 5e-04 | 3.1 |
| 8 | 0.2084 | 0.7822 | 0.9198 | 0.6445 | 5e-04 | 3.1 |
| 9 | 0.2038 | 0.7824 | 0.9200 | 0.6449 | 3e-04 | 2.9 |

Best epoch 6: validation score 0.7862. Training took 41 minutes.

## GRU against the best baseline

Same 68,837 validation customers for every row.

| Model                    |   Score | 95% CI          |   Gini |   Top 4% |
|:-------------------------|--------:|:----------------|-------:|---------:|
| LightGBM (best baseline) |  0.7907 | 0.7849 - 0.7966 | 0.923  |   0.6583 |
| GRU                      |  0.7862 | 0.7779 - 0.7925 | 0.9213 |   0.651  |
| Blend: mean rank         |  0.7943 | 0.7882 - 0.7997 | 0.9245 |   0.6642 |

GRU minus LightGBM: -0.0048, 95% CI -0.0090 to -0.0009 (paired bootstrap); GRU ahead in 0% of resamples.
Blend minus LightGBM: +0.0038, 95% CI +0.0011 to +0.0060; blend ahead in 100% of resamples.

## By history length

| History        | Customers   | Default rate   |   LightGBM |    GRU |   GRU - LightGBM |
|:---------------|:------------|:---------------|-----------:|-------:|-----------------:|
| All 13 months  | 58,021      | 23.2%          |     0.8148 | 0.8116 |          -0.0033 |
| 6 to 12 months | 6,912       | 42.9%          |     0.6812 | 0.6719 |          -0.0093 |
| 1 to 5 months  | 3,904       | 35.9%          |     0.6152 | 0.6259 |           0.0107 |
