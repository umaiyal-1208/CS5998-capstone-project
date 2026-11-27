# Baselines

Made by `scripts/train_baselines.py` on the summary features from
`scripts/build_features.py` (one row of 1,085 features per customer). Trained
on the 321,239 training customers and scored on the 68,837 validation
customers with the competition metric (`docs/metric.md`). The test set was not
loaded. The full run took 19 minutes on a 16 GB laptop CPU.

## Results

| Model | Imbalance handling | Score | 95% CI |
|---|---|---|---|
| Logistic regression | none | 0.7808 | 0.7732 - 0.7870 |
| **LightGBM** | **none** | **0.7907** | **0.7849 - 0.7966** |

## What the results show

1. **LightGBM is the stronger baseline, and the gap is real.** It beats
   logistic regression by +0.0101. A paired bootstrap over the same validation
   customers puts the difference between +0.0062 and +0.0145, and LightGBM wins
   in every one of the 200 resamples. So **0.7907 is the score the sequence
   models have to beat.**

2. **Logistic regression is a strong floor, not a weak one.** 0.7808 is only
   about one point below LightGBM. Most of the predictive power comes from the
   summary features themselves (especially the last statement and the mean), and
   a linear model can already use them. The extra point LightGBM gains comes
   from non-linear effects and interactions between features.

3. **Correcting for class imbalance did not help either model.** The
   Project Definition planned to compare class weights with resampling. Both
   came out slightly worse than plain training: class weights by 0.001 to 0.003,
   undersampling by 0.003 to 0.004, though the confidence intervals overlap.
   This is expected for this metric, which depends only on how customers are
   ranked. Reweighting shifts the predicted probabilities but barely changes
   their order, and undersampling throws away about two-thirds of the
   non-defaulters the model could learn from. Plain training is used from here on.

4. **Payment and Balance features carry most of the signal.** P_2 alone
   (last, mean, min and max together) accounts for about 39% of LightGBM's
   split gain. The 3 Payment features give 40% of the gain, Balance 26%,
   Delinquency 21%, Risk 7% and Spend 6%. Delinquency has the most columns (540
   of the 1,085) but much of it is sparse, as the audit found.

5. **The latest statement matters most, but history adds to it.** "Last" values
   account for 50% of the gain. The other half comes from statistics over the
   whole history (mean 21%, min 9%, max 9%, change 6%, std 5%). This supports
   the project's question: the history carries information that the latest
   month alone does not. The question for the GRU is whether reading it in
   order captures more than these summaries do.

6. **History length gets almost no gain directly.** The audit showed customers
   with shorter histories default more often, yet `n_statements` contributes
   close to 0% of the gain. The likely reason is that the same information is
   already present in other features. For example, `std` and `change` are empty
   or zero for a one-statement customer, so the trees pick it up there instead.

## Comparison with published results

The competition's winning solutions scored about 0.81 on the private
leaderboard, using large ensembles and heavy feature engineering. A single
LightGBM at 0.79 on summary features is in the range expected for a baseline.
The numbers are not directly comparable, because these scores are on a held-out
part of the training data, not on the competition's private test set.

## Caveats

- LightGBM picks its number of trees (476) by early stopping on the validation
  set, so its validation score is slightly optimistic. The same will apply to
  the sequence models, so the comparison stays fair, and the final comparison
  will be on the untouched test set.
- No hyperparameter tuning has been done yet. The LightGBM settings are
  reasonable defaults. Optuna tuning is planned for the final milestone and
  should add a little.

---

## Script output

Train 321,239 customers, validation 68,837, 1085 features.

  logreg_none                score 0.7808  (1.8 min)
  logreg_class_weight        score 0.7795  (1.6 min)
  logreg_undersample         score 0.7781  (0.7 min)
  lightgbm_none              score 0.7907  (5.7 min)
  lightgbm_class_weight      score 0.7879  (6.2 min)
  lightgbm_undersample       score 0.7866  (2.3 min)

## Validation scores

Competition metric on 68,837 validation customers. 95% CI from 200 bootstrap resamples.

| Model    | Imbalance    |   Score | 95% CI          |   Gini |   Top 4% | Trees   |   Minutes |
|:---------|:-------------|--------:|:----------------|-------:|---------:|:--------|----------:|
| logreg   | none         |  0.7808 | 0.7732 - 0.7870 | 0.9173 |   0.6442 | -       |       1.8 |
| logreg   | class_weight |  0.7795 | 0.7727 - 0.7864 | 0.9171 |   0.6419 | -       |       1.6 |
| logreg   | undersample  |  0.7781 | 0.7710 - 0.7844 | 0.9163 |   0.64   | -       |       0.7 |
| lightgbm | none         |  0.7907 | 0.7849 - 0.7966 | 0.923  |   0.6583 | 476     |       5.7 |
| lightgbm | class_weight |  0.7879 | 0.7816 - 0.7948 | 0.9227 |   0.6532 | 508     |       6.2 |
| lightgbm | undersample  |  0.7866 | 0.7805 - 0.7923 | 0.9219 |   0.6513 | 342     |       2.3 |

Best LightGBM (lightgbm_none) minus best logistic regression (logreg_none): +0.0101, 95% CI +0.0062 to +0.0145 (paired bootstrap). It is above zero in 100% of resamples.

## LightGBM: which features it relies on

Share of total split gain, best LightGBM model.

| Family         | Gain share   |   Features |
|:---------------|:-------------|-----------:|
| Payment        | 40.3%        |         18 |
| Balance        | 25.6%        |        232 |
| Delinquency    | 21.1%        |        540 |
| Risk           | 7.1%         |        168 |
| Spend          | 5.9%         |        126 |
| History length | 0.0%         |          1 |

| Statistic   | Gain share   |
|:------------|:-------------|
| last        | 49.8%        |
| mean        | 21.1%        |
| min         | 9.3%         |
| max         | 8.8%         |
| change      | 6.1%         |
| std         | 4.9%         |
| nunique     | 0.0%         |
| count       | 0.0%         |

Top 15 features:

| Feature   | Gain share   |
|:----------|:-------------|
| P_2_last  | 16.9%        |
| P_2_mean  | 12.7%        |
| B_9_last  | 6.5%         |
| P_2_min   | 5.6%         |
| P_2_max   | 3.8%         |
| B_1_last  | 3.1%         |
| B_2_last  | 2.7%         |
| D_44_last | 2.6%         |
| R_1_last  | 1.9%         |
| B_18_last | 1.7%         |
| D_48_last | 1.7%         |
| B_37_last | 1.5%         |
| R_1_mean  | 1.4%         |
| B_3_last  | 1.1%         |
| D_39_last | 1.0%         |

Done in 19 minutes.
