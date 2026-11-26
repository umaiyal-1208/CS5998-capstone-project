# Evaluation metric

All models in this project are compared on the competition's own metric,
implemented in `scripts/metric.py`:

    M = 0.5 * (G + D)

- **G, normalised Gini.** It measures how well the model ranks defaulters above
  non-defaulters across the whole list. A value of 1 is a perfect ranking and 0
  is no better than random. It equals 2 x AUC - 1, except that here the
  customers are weighted.
- **D, default rate captured at 4%.** Take the 4% of customers the model ranks
  as riskiest and ask what share of all defaulters they contain. This is how a
  lender actually uses the score: only the top of the list gets reviewed or
  declined.

## The 20x weight

The training data has 25.9% defaulters (`docs/data-audit.md`), but the
organisers built it by keeping only 5% of non-defaulters. The metric gives each
non-defaulter a weight of 20 to undo that sampling, so G and D are computed as
if on the real population. This changes what "top 4%" means: with the weights,
the riskiest 4% is a much smaller set of rows than 4% of the file.

Leaving the weight out would give a different number that cannot be compared
with published results, so the weighting is the most important thing the tests
check.

## How it is checked

`tests/test_metric.py` (run with `python -m pytest tests -q`) checks:

1. **A four-customer example worked out by hand** in the test's docstring. It
   gives G = 2/23 and D = 0.5.
2. **Agreement with the reference code** the competition hosts published,
   copied into the test, on three random datasets of 20,000 customers.
3. **Sanity properties:** a perfect ranking has G = 1, a random ranking has G
   near 0, and transforming the scores without changing their order leaves
   M unchanged.

All 8 tests pass.

## What the metric does not measure

M depends only on the order of the predictions, so it says nothing about
whether a predicted 0.3 really means a 30% chance of default. Calibration will
be reported separately for the final models, because a lender setting credit
limits needs the probabilities themselves, not just the ranking.
