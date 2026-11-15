# Sufficient statistics in Bayesian modelling

A small, growing project: one blog post (`sufficient_statistics.qmd`) plus one self-contained notebook per
modelling family. Each notebook fits the same model twice, on raw rows and on a sufficient (or approximately
sufficient) summary, checks the posteriors agree, and times both.

| # | Notebook | Raw data | Summary T(y) | PyMC likelihood |
|---|----------|----------|--------------|-----------------|
| 01 | `notebooks/01_ab_tests_counts_and_moments.ipynb` | user-level conversions / revenue | (n, k); (n, ȳ, SS) | `Binomial`; `Potential` |
| 02 | `notebooks/02_sequential_multinomial.ipynb` | event stream with arm labels | running counts S_n | `Multinomial` (+ closed-form Bayes factor) |
| 03 | `notebooks/03_survival_person_period.ipynb` | person-period rows | (d, r) or (d, E) per cell | `Binomial` cloglog; `Poisson` with exposure |
| 04 | `notebooks/04_funnels_markov_chains.ipynb` | transitions / user-stage rows | transition counts N_ij; stage head-counts | `Multinomial`; `Binomial` |
| 05 | `notebooks/05_linear_regression_gram.ipynb` | 1M × 10 design | (n, X'X, X'y, y'y) | `Potential` |
| 06 | `notebooks/06_approximate_sufficiency.ipynb` | shots with xG offsets | (G, X, V) per player | `Normal` pseudo-observation |

## Running

All notebooks use the `bayesian_causal_book` kernel (PyMC 6, ArviZ 1.x, nutpie). Execute one in place with:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=bayesian_causal_book --ExecutePreprocessor.timeout=1800 notebooks/01_ab_tests_counts_and_moments.ipynb
```

Notebooks must be committed **with outputs**: the post pulls figures and tables from them with
`{{< embed notebooks/<file>.ipynb#<label> >}}` and does not re-run them.

## Adding a case

1. Copy the structure of an existing notebook: statement of T(y) and the factorisation → simulate raw rows →
   full fit and compressed fit via the inline `timed_fit` helper → equivalence figure → speed table →
   "what you give up".
2. Label the two embeddable cells `#| label: fig-<case>-equiv` and `#| label: tbl-<case>-speed`.
3. Add a section to `sufficient_statistics.qmd` that embeds them, and a row to the summary table here and in the post.

Candidate future cases: Gaussian processes with gridded / binned inputs, multinomial logit with
choice-set counts, Poisson process intensities from binned event times, Dirichlet-multinomial
over-dispersion (where sufficiency fails and why).
