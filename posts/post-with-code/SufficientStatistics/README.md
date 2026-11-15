# Sufficient statistics in Bayesian modelling

A small, growing project: one blog post (`sufficient_statistics.qmd`) plus one self-contained notebook per
modelling family. Each notebook shows what the model's output is used for, then fits the same model twice, on raw
rows and on a sufficient (or approximately sufficient) summary, checks the posteriors agree, and times both.

| # | Notebook | Raw data | Summary T(y) | PyMC likelihood |
|---|----------|----------|--------------|-----------------|
| 01 | `notebooks/01_ab_tests_counts_and_moments.ipynb` | user-level conversions / revenue | (n, k); (n, ȳ, SS) | `Binomial`; `Normal` on ȳ + `Gamma` on SS |
| 02 | `notebooks/02_sequential_multinomial.ipynb` | event stream with arm labels | running counts S_n | `Multinomial` (+ closed-form Bayes factor) |
| 03 | `notebooks/03_survival_person_period.ipynb` | person-period rows | (d, r) or (d, E) per cell | `Binomial` cloglog; `Poisson` with exposure |
| 04 | `notebooks/04_funnels_markov_chains.ipynb` | transitions / user-stage rows | transition counts N_ij; stage head-counts | `Multinomial`; `Binomial` |
| 05 | `notebooks/05_linear_regression_gram.ipynb` | 1M × 10 design | (n, X'X, X'y, y'y) | `Potential` |
| 06 | `notebooks/06_approximate_sufficiency.ipynb` | shots with xG offsets | (G, X, V) per player | `Normal` pseudo-observation |
| 07 | `notebooks/07_competing_risks.ipynb` | person-period rows, K causes | (E, d_1..d_K) or (r, d_1..d_K) per cell | `Poisson` per cause; `Multinomial` |

The post presents competing risks (07) straight after survival (03), so its section order differs from the notebook
numbers. Notebook numbers are kept stable because the post links to them.

## Running

All notebooks use the `bayesian_causal_book` kernel (PyMC 6, ArviZ 1.x, nutpie). Execute one in place from the
`notebooks/` folder with:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=bayesian_causal_book --ExecutePreprocessor.timeout=1800 01_ab_tests_counts_and_moments.ipynb
```

Notebooks must be committed **with outputs**: the post pulls figures from them with
`{{< embed notebooks/<file>.ipynb#<label> >}}` and does not re-run them.

Each notebook writes `notebooks/timings/<notebook>.json` when it runs. The post's Timings table is built from these
files when the post renders, so it always matches the committed notebooks. Re-render the post after re-running a
notebook.

## Conventions

- **Plots:** colour marks the category (arm, cause, source, model); a filled translucent histogram or band is the
  raw-row fit; a solid line is the compressed fit; black dashed is an exact (conjugate or closed-form) answer; thin grey
  is the true value. Intervals are 89%.
- **Notebook structure:** statement of T(y) and the factorisation → simulate raw rows → full and compressed fits with the
  inline `timed_fit` helper → a figure showing what the output is used for → an equivalence figure → the timing table →
  "What compression gives up".
- **Labels:** `fig-<case>-use` (and other descriptive `fig-<case>-*` labels) for substantive figures,
  `fig-<case>-equiv` for the equivalence figure, `tbl-<case>-speed` for the timing table.
- **Voice:** plain statements; one claim per sentence; define terms where they first appear.

## Adding a case

1. Copy an existing notebook's structure and its `timed_fit` / `save_timings` helpers. Give each fit a `case` name
   (shared by the raw and compressed fit) and a `kind` (`raw` or `compressed`).
2. Label the embeddable cells as above.
3. Add a section to `sufficient_statistics.qmd` that embeds them, a row to the summary table there and here, and
   re-render the post.

Candidate future cases: deduplicated rows with frequency weights (any likelihood), over-dispersion and
counts-of-counts (Beta-binomial), GAMs and HSGPs with a Gaussian likelihood (basis Gram matrix), hierarchical regression
from per-group summaries, binned point processes, discrete choice with repeated choice sets, capture–recapture
histories, and hidden Markov models (where nothing compresses).
