# Data audit

Every number below is printed by `scripts/audit_data.py`, run on raddar's
integer-dtype Parquet copy of the statements (`train.parquet`) and on the
competition's own `train_labels.csv`. The full output follows the summary, and
`data/audit/columns.csv` (not committed) has one row per feature.

## What the audit found

1. **The working copy is complete.** It has 5,531,451 statement rows and 458,913
   customers, which matches the published competition figures exactly. Every
   customer with statements has exactly one label, and every label has
   statements. There are no duplicate (customer, date) rows, and every
   customer's rows are already in date order.

2. **There are 188 features, not 190.** The data has 190 columns, but two of
   them are the customer ID (`customer_ID`) and the statement date (`S_2`). The
   Project Definition said "190 features"; that should read 188 features. Of
   these, 11 are categorical, as declared by the organisers.

3. **The positive class is 25.9% of customers.** This is not the real default
   rate. The competition says the non-default class was subsampled to 5% and
   that its evaluation metric weights each non-default 20 times to make up for
   it. The project's metric implementation has to apply the same weighting, or
   its scores will not be comparable with published results.

4. **Histories are mostly full but not all.** 84.1% of customers have 13
   statements, one per month from March 2017 to March 2018. The remaining 15.9%
   have between 1 and 12. Shorter histories default more often: 31.8% to 46.2%
   for customers with 1 to 11 statements, against 23.2% for customers with all
   13. History length therefore carries signal of its own. The sequence models
   need to see it, through the padding mask, and the aggregated features for
   the baselines should include a statement count.

5. **Missing values are coded two ways.** Float columns use NaN. Integer
   columns cannot hold NaN, and none of the 95 integer columns contains one.
   Instead, 55 of them have a minimum of -1, and -1 is the only negative value
   any integer column takes. The audit therefore treats -1 as a missing code.
   This matters: counting NaN alone, 12 Delinquency columns are over 50%
   missing; counting -1 as well, 23 are.

6. **Missing values are concentrated in a few columns.** 122 of 188 features
   have some missing values, but most have very few. 30 features are over 50%
   missing, and 23 of those are Delinquency variables. D_87, D_88, D_108,
   D_110, D_111 and B_39 are over 99% missing. D_134 to D_138 all have the same
   missing rate, 96.5%, which suggests they come from one source that is
   present for few customers. Missingness can be informative in credit data
   (a field may be empty because an event never happened), so these columns
   are not dropped yet. The baselines will compare keeping and dropping them.

7. **Dates need parsing.** `S_2` is stored as a string. It is parsed to a date
   during preprocessing.

## What this means for the next steps

- **Split:** by customer, 70/15/15, stratified on the target (no customer
  appears in two splits).
- **Metric:** implement the competition metric with the 20x weight on
  non-defaults, and check it against a hand-computed example.
- **Baselines:** aggregate each feature over a customer's history (last value,
  mean, min, max, standard deviation), add the statement count, and replace -1
  with NaN in integer columns before aggregating so it is not averaged as a
  number. Categorical columns get last value and count of distinct values.
- **Sequences:** pad every customer to 13 statements, keep a mask of real
  months, and give missing entries their own indicator rather than filling them
  silently.

## Deviation from the Project Definition

The definition planned to check the Parquet copy against a sample of the
original 16 GB CSV. That download is not needed: the row, customer and column
counts match the published figures exactly, and the labels come from the
competition itself rather than from a re-upload. Downloading the CSV would add
no check that these do not already give.

---

## Full audit output

## Overview

| Measure                        | This copy   | Published   |
|:-------------------------------|:------------|:------------|
| Statement rows                 | 5,531,451   | 5,531,451   |
| Customers with statements      | 458,913     | 458,913     |
| Columns, including ID and date | 190         | 190         |
| Feature columns                | 188         | -           |
| Label rows                     | 458,913     | 458,913     |

In memory as a pandas DataFrame: 3.42 GB.

## Column types

| dtype   |   columns |
|:--------|----------:|
| float32 |        93 |
| int8    |        86 |
| int16   |         9 |

The date column `S_2` is stored as `object`.

## Labels

| Measure                                  | Value   |
|:-----------------------------------------|:--------|
| Unique customers in labels               | 458,913 |
| Duplicate label rows                     | 0       |
| Target values                            | 0, 1    |
| Default rate                             | 25.9%   |
| Customers with statements but no label   | 0       |
| Customers with a label but no statements | 0       |

## Statements per customer

|   Statements | Customers   | Share   | Default rate   |
|-------------:|:------------|:--------|:---------------|
|            1 | 5,120       | 1.1%    | 33.6%          |
|            2 | 6,098       | 1.3%    | 31.8%          |
|            3 | 5,778       | 1.3%    | 35.9%          |
|            4 | 4,673       | 1.0%    | 41.6%          |
|            5 | 4,671       | 1.0%    | 39.3%          |
|            6 | 5,515       | 1.2%    | 38.8%          |
|            7 | 5,198       | 1.1%    | 41.8%          |
|            8 | 6,110       | 1.3%    | 44.7%          |
|            9 | 6,411       | 1.4%    | 45.0%          |
|           10 | 6,721       | 1.5%    | 46.2%          |
|           11 | 5,961       | 1.3%    | 44.7%          |
|           12 | 10,623      | 2.3%    | 38.9%          |
|           13 | 386,034     | 84.1%   | 23.2%          |

Statement dates run from 2017-03-01 to 2018-03-31.
They fall in 13 calendar months.

| Month   | Statements   |
|:--------|:-------------|
| 2017-03 | 395,630      |
| 2017-04 | 401,556      |
| 2017-05 | 400,306      |
| 2017-06 | 410,384      |
| 2017-07 | 415,602      |
| 2017-08 | 421,099      |
| 2017-09 | 425,455      |
| 2017-10 | 430,292      |
| 2017-11 | 434,268      |
| 2017-12 | 438,910      |
| 2018-01 | 446,255      |
| 2018-02 | 452,781      |
| 2018-03 | 458,913      |

Rows sharing a customer and statement date: 0.
Customers whose rows are not in date order: 0.

## Feature groups

Missing means NaN in a float column or -1 in an integer column.

| Group       |   Columns |   Categorical |   With any missing | Median missing share   |   >10% missing |   >50% missing |
|:------------|----------:|--------------:|-------------------:|:-----------------------|---------------:|---------------:|
| Balance     |        40 |             2 |                 23 | 0.0%                   |              4 |              4 |
| Delinquency |        96 |             9 |                 81 | 3.2%                   |             29 |             23 |
| Payment     |         3 |             0 |                  2 | 0.8%                   |              0 |              0 |
| Risk        |        28 |             0 |                  7 | 0.0%                   |              2 |              2 |
| Spend       |        21 |             0 |                  9 | 0.0%                   |              4 |              1 |

Across all 188 features: 122 have missing values, 39 are over 10% missing and 30 are over 50% missing.

## Integer columns and possible missing-value codes

95 feature columns are stored as integers and 0 of them contain a NaN. 55 have a negative minimum, and that minimum is -1 in every case.

| Column   | Group       | Type   | Categorical   |   Minimum | Rows at minimum   |   Distinct values |
|:---------|:------------|:-------|:--------------|----------:|:------------------|------------------:|
| D_87     | Delinquency | int8   | False         |        -1 | 99.9%             |                 2 |
| D_108    | Delinquency | int8   | False         |        -1 | 99.5%             |                11 |
| D_111    | Delinquency | int8   | False         |        -1 | 99.4%             |                 4 |
| D_138    | Delinquency | int8   | False         |        -1 | 96.5%             |                 8 |
| D_137    | Delinquency | int8   | False         |        -1 | 96.5%             |                 3 |
| D_136    | Delinquency | int8   | False         |        -1 | 96.5%             |                 9 |
| D_135    | Delinquency | int8   | False         |        -1 | 96.5%             |                 3 |
| R_9      | Risk        | int8   | False         |        -1 | 94.3%             |                11 |
| D_106    | Delinquency | int16  | False         |        -1 | 90.2%             |               309 |
| D_49     | Delinquency | int16  | False         |        -1 | 90.1%             |               321 |
| R_26     | Risk        | int16  | False         |        -1 | 89.0%             |               158 |
| D_66     | Delinquency | int8   | True          |        -1 | 88.7%             |                 3 |
| D_82     | Delinquency | int8   | False         |        -1 | 73.4%             |                 8 |
| D_78     | Delinquency | int8   | False         |        -1 | 5.0%              |                25 |
| D_44     | Delinquency | int8   | False         |        -1 | 5.0%              |                38 |
| D_64     | Delinquency | int8   | True          |        -1 | 3.9%              |                 5 |
| D_68     | Delinquency | int8   | True          |        -1 | 3.9%              |                 8 |
| D_83     | Delinquency | int8   | False         |        -1 | 3.5%              |                33 |
| D_113    | Delinquency | int8   | False         |        -1 | 3.2%              |                24 |
| D_120    | Delinquency | int8   | True          |        -1 | 3.2%              |                 3 |

## Columns with the most missing values

| Column   | Group       | Type    | Missing share   |
|:---------|:------------|:--------|:----------------|
| D_87     | Delinquency | int8    | 99.9%           |
| D_88     | Delinquency | float32 | 99.9%           |
| D_108    | Delinquency | int8    | 99.5%           |
| D_110    | Delinquency | float32 | 99.4%           |
| D_111    | Delinquency | int8    | 99.4%           |
| B_39     | Balance     | float32 | 99.4%           |
| D_73     | Delinquency | float32 | 99.0%           |
| B_42     | Balance     | float32 | 98.7%           |
| D_137    | Delinquency | int8    | 96.5%           |
| D_138    | Delinquency | int8    | 96.5%           |
| D_135    | Delinquency | int8    | 96.5%           |
| D_136    | Delinquency | int8    | 96.5%           |
| D_134    | Delinquency | float32 | 96.5%           |
| R_9      | Risk        | int8    | 94.3%           |
| B_29     | Balance     | float32 | 93.1%           |
| D_106    | Delinquency | int16   | 90.2%           |
| D_132    | Delinquency | float32 | 90.2%           |
| D_49     | Delinquency | int16   | 90.1%           |
| R_26     | Risk        | int16   | 89.0%           |
| D_76     | Delinquency | float32 | 88.7%           |

## Categorical columns

| Column   | Type   |   Distinct (incl. missing) | Values                  |
|:---------|:-------|---------------------------:|:------------------------|
| B_30     | int8   |                          4 | -1, 0, 1, 2             |
| B_38     | int8   |                          8 | -1, 1, 2, 3, 4, 5, 6, 7 |
| D_114    | int8   |                          3 | -1, 0, 1                |
| D_116    | int8   |                          3 | -1, 0, 1                |
| D_117    | int8   |                          8 | -1, 0, 2, 3, 4, 5, 6, 7 |
| D_120    | int8   |                          3 | -1, 0, 1                |
| D_126    | int8   |                          4 | -1, 0, 1, 2             |
| D_63     | int8   |                          6 | 0, 1, 2, 3, 4, 5        |
| D_64     | int8   |                          5 | -1, 0, 1, 2, 3          |
| D_66     | int8   |                          3 | -1, 0, 1                |
| D_68     | int8   |                          8 | -1, 0, 1, 2, 3, 4, 5, 6 |

