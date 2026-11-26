# Customer split

Made by `scripts/make_split.py`. The split itself is written to
`data/splits/split.parquet` and is not committed, because customer IDs are part
of the competition data. Each set's fingerprint below (the first 16 hex digits
of a SHA-256 of its sorted customer IDs) lets anyone who reruns the script
confirm they have the same split.

## Decisions

- **Split by customer, not by row.** A customer's monthly statements are highly
  correlated. If some months were in training and others in test, the test score
  would reward recognising that customer rather than predicting for a new one.
  The output confirms no customer appears in two sets.
- **Stratified on the target.** All three sets have the same default rate
  (25.89%), so validation and test scores are not moved by chance differences
  in class balance.
- **Fixed seed (5998).** Rerunning the script gives the identical split; two
  consecutive runs produced byte-identical output.
- **The test set is held back.** Model selection and Optuna tuning use the
  validation set only. The test set is scored once per final model, at the end.

## Findings

1. The sizes are 321,239 / 68,837 / 68,837 customers, exactly 70/15/15.
2. Stratifying on the target alone also kept history length balanced: 84.1%,
   84.3% and 84.1% of customers have all 13 statements, and every shorter
   length is within 0.1 percentage points across the sets. History length is a
   strong signal (see `docs/data-audit.md`), so this check matters. A further
   stratification on history length was not needed.

---

## Script output

## Customer split

Seed 5998. Shares train 70%, valid 15%, test 15%, stratified on `target`.

| Split   | Customers   | Share   | Defaults   | Default rate   | Statements   | Full 13-month histories   | Fingerprint      |
|:--------|:------------|:--------|:-----------|:---------------|:-------------|:--------------------------|:-----------------|
| train   | 321,239     | 70.0%   | 83,180     | 25.89%         | 3,871,490    | 84.1%                     | 2b318c0e5bc57ca3 |
| valid   | 68,837      | 15.0%   | 17,824     | 25.89%         | 830,435      | 84.3%                     | 966f4e42fde126a2 |
| test    | 68,837      | 15.0%   | 17,824     | 25.89%         | 829,526      | 84.1%                     | 03ee6bcf30b52371 |

## Statements per customer, by split

|   Statements | train   | valid   | test   |
|-------------:|:--------|:--------|:-------|
|            1 | 1.1%    | 1.1%    | 1.1%   |
|            2 | 1.3%    | 1.3%    | 1.4%   |
|            3 | 1.3%    | 1.3%    | 1.3%   |
|            4 | 1.0%    | 1.0%    | 1.1%   |
|            5 | 1.0%    | 1.0%    | 1.0%   |
|            6 | 1.2%    | 1.1%    | 1.2%   |
|            7 | 1.1%    | 1.2%    | 1.1%   |
|            8 | 1.3%    | 1.4%    | 1.3%   |
|            9 | 1.4%    | 1.3%    | 1.4%   |
|           10 | 1.5%    | 1.5%    | 1.4%   |
|           11 | 1.3%    | 1.3%    | 1.3%   |
|           12 | 2.3%    | 2.3%    | 2.3%   |
|           13 | 84.1%   | 84.3%   | 84.1%  |

Customers appearing in more than one split: 0.
Customers without a split: 0.

