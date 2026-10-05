# Change-point mixed logit — testing routine

A discrete choice model in which each household's utility function is allowed to
pivot at a latent moment in its own purchase sequence, plus the simulation
harness needed to find out whether such a pivot can actually be detected.

> **Orientation.** `transformative_choice.qmd` is the published post and is
> self-contained: it writes its own model and simulator inline and imports only
> data plumbing from `choice_panel.py`. `_harness/` is the archival full-budget
> test rig that produced `results/`; it is not what the post runs, and it keeps
> its original parameter names. Two were renamed for the post, because the old
> ones were opaque to a reader: `tau_policy="late_escape"` → `"late_only"`,
> `tau_policy="interior"`/"aliased" → `"both_ends"`, and `alpha_pooled` →
> `pooled_shift`. `choice_panel.tau_bounds` accepts both spellings. Numbers in
> this README come from the harness at full budget and will not match the post's
> shorter chains exactly; the post also uses wider priors on the shift
> parameters (`mu_delta_alpha ~ Normal(0, 2.5)`, `sigma_delta_alpha ~
> HalfNormal(2)`), which the harness predates.

The motivation is L.A. Paul's *transformative experience*: an event after which
you evaluate options by a different utility function than the one you had going
in, so that your prior self could not have assessed the change. If that is a
real feature of preference rather than a philosopher's thought experiment, a
panel of repeated choices is where it should leave a trace. This code asks
whether the trace is recoverable.

It extends the panel mixed logit from the cracker case study in
`bayesian_causal_infer/03_structural_causal_models/discrete_choice_counterfactuals.qmd`,
which uses a train/test split on the same data to argue that the latent
preference ordering is *stable*. The change-point version turns that stability
claim into an estimable parameter rather than a background assumption.

## The model

```
w_nt  = sigmoid((t_nt - tau_n) / s)
U_njt = (alpha_j + w_nt * Delta_alpha_j)
        + (beta_n + w_nt * Delta_n) * price_njt
        + disp_njt * b_disp + feat_njt * b_feat
        + gamma * 1[household n chose j on trip t-1]     # optional
```

`w` runs smoothly from 0 before the pivot to 1 after it, which keeps `tau`
continuous and NUTS differentiable. `s` (`transition_width`) sets how abrupt the
pivot is. `Delta_alpha` is a shift in brand loyalty; `Delta_n` is a shift in
price sensitivity, partially pooled across households.

Specifications are selected by `shift_mode`:

| `shift_mode` | what pivots | reading |
| --- | --- | --- |
| `none` | nothing | the chapter's model; the null |
| `price` | price sensitivity | "I became price-conscious" |
| `alpha` | brand intercepts | "I became a different kind of buyer" |
| `both` | both | the full pivot |

and by `tau_mode`:

| `tau_mode` | τ | question |
| --- | --- | --- |
| `individual` | latent, one per household | *did* something transformative happen, and when? |
| `common` | latent, one shared | a contrast arm: a single pivot in tenure |
| `known` | supplied as data | given the event date, how far did utility move? |

`known` is the one that matches the philosophy. Paul's transformative
experiences — becoming a parent, being turned into a vampire — are singular and
**self-known**: the agent knows the date. What they cannot do beforehand is
evaluate the utility function they will hold afterwards. The estimand her
argument concerns is therefore Δ *given* τ, and hunting for τ is a separate,
much harder problem that her cases do not pose. The `known_tau` experiment
measures the gap between the two.

## Where τ is allowed to go decides what the model can mean

A change-point model has two ways to encode "no change", and keeping both is
fatal:

- τ past the last trip → `w ≈ 0` throughout → fitted utility is `alpha`
- τ before the first trip → `w ≈ 1` throughout → fitted utility is `alpha + Δalpha`

Both are the null model, merely re-parameterised. Leaving both open aliases
them: nothing fixes which segment is "pre", `delta_alpha` acquires a
label-switching symmetry, and the sampler falls apart. On the real cracker panel
this shows up as `sigma_tau ≈ 5.8`, τ pinned to its bounds, `delta_alpha` means
differing by a constant offset between chain pairs, and r_hat ≈ 23. What the
model is fitting at that point is a two-class segmentation of households, not a
change point — which fits the panel far better and means something else
entirely.

`tau_policy="late_escape"` (the default) closes the early door and leaves the
late one open. That alone drops `sigma_tau` to ≈0.8 and restores r_hat. It also
fixes Δ's meaning as "the shift away from the regime that held at the start".

**But reading a non-discovery off τ's position does not work**, and it is worth
being clear why. That behaviour — τ running to the end of the series for want of
anywhere better — requires that the change magnitude *cannot vanish*. Here Δ can
shrink to zero, which is the cheaper escape, so the likelihood goes flat in τ and
τ simply reverts to its prior: mid-window, looking for all the world like a
located pivot. Under a true null we measure `frac_persons_pivot_found = 1.00`
with `p_pivot_mean = 0.84`. The geometry is not carrying the information.

## The explicit no-change state

So the null gets its own state, marginalised at the household level:

```
p(y_n) = pi * p(y_n | pivoted, tau_n, Delta_n) + (1 - pi) * p(y_n | never pivoted)
```

Pass `mixture=True`. This adds `pi_pivot`, the population rate of
transformation, and `p_transformed`, a per-household posterior probability — the
non-discovery reported as a number rather than inferred from geometry or from a
failed diagnostic. No discrete sampling is needed; the indicator is summed out.

It separates two abilities that are easy to run together:

| true pivot rate | `pi_pivot` est | frac flagged | AUC (which household?) |
| --- | --- | --- | --- |
| 0.0 | 0.41 | **0.00** | — |
| 0.5 | 0.69 | — | **0.68** |
| 1.0 | 0.84 | **0.96** | — |

Population-level detection is reliable. Per-household identification is close to
guesswork. That is not a defect in the estimator, and it is not the aggregate
"swamping" the individual — there is no competition between the two levels. It is
a ratio of denominators, and `information_budget` computes it before any sampler
runs.

## The information budget

Per household, the evidence bearing on *its own* pivot is the KL divergence
between the pivot and no-pivot predictive distributions, accumulated over that
household's trips. Call it nats. To reach 90% posterior confidence on one
household you need about 2.2 nats (log 9).

| true Δ | median nats per household | frac of households identifiable | pooled nats, population question |
| --- | --- | --- | --- |
| 1.0 | 0.15 | 0.00 | 32.9 |
| 2.5 | 1.09 | 0.11 | 183.2 |
| 4.0 | 2.56 | 0.57 | 413.1 |

Read the Δ=1.0 row carefully. You can be overwhelmingly confident that
transformation is happening in this population while being unable to name a
single household it happened to. The per-household figure at Δ=2.5 — 1.09 nats,
11% clearing the bar — predicts the AUC of ≈0.68 above without fitting anything.

Three structural facts set the ceiling:

- **Within-household price range is median $0.26.** A Δ of 2.5 therefore buys a
  utility swing of about 0.64 logit units. That is the pivot's entire observable
  footprint, and it scales with the price variation a household happens to face.
- **Maximum choice entropy is 1.07 nats per trip** given these market shares, and
  the pivot explains a small part of it.
- **Only ~4 of 20 trips have `w` in (0.05, 0.95).** Just those few sit in the
  transition zone and speak to *when* τ is; the other sixteen are deep inside one
  regime and are silent on timing. A household whose prices happen to be similar
  either side of its τ contributes nothing about timing at any sample size.

This makes the design requirements explicit. To identify individual pivots you
need a larger Δ (≈4 on this price scale), more trips per household, more price
variation per household, or a known τ — which is what `tau_mode="known"` buys,
and why conditioning on the date roughly halves the posterior width on Δ.

Because `mixture=True` replaces the `Categorical` likelihood with a `Potential`,
those fits are not LOO-comparable against the others.

## Running it

```bash
python run_tests.py                    # all experiments
python run_tests.py null power habit   # selected experiments
```

Results are written as tidy CSVs to `results/`.

## The experiments

| name | question | what good looks like |
| --- | --- | --- |
| `calibration` | is the machinery self-consistent? | truth drawn from the prior is recovered with coverage at the nominal 94% |
| `null` | static but heterogeneous preferences | no pivot claimed; `tau` error equals a midpoint guess |
| `power` | how big must a pivot be to be seen? | detection rate rises with `Delta`; `tau` error beats the midpoint benchmark |
| `alpha_pivot` | can it tell *which* thing changed? | a loyalty pivot is attributed to loyalty, not to price |
| `habit` | state dependence vs. transformation | a habit-driven panel does not read as a pivot once habit is controlled |
| `degeneracy` | is the change point still a change point? | some remedy restores `sigma_tau` and `r_hat` |
| `tau_policy` | interior bound vs. late escape | late escape fixes `sigma_tau`; τ geometry alone does *not* signal non-discovery |
| `mixture` | non-discovery as a probability | flag rate tracks the true pivot rate; per-household AUC is weak |
| `known_tau` | what does not knowing *when* cost? | Δ contracts sharply once τ is given |
| `real` | what does the actual cracker panel say? | — |

## The operating envelope

The acceptance test (`envelope`): recover each household's own transformed
utility when a real event occurred — with the date known and with it estimated —
and report no change when nothing happened. Criteria:

| criterion | target |
| --- | --- |
| identify *who* | AUC ≥ 0.85 on which households pivoted |
| identify *what* | corr(Δα̂ₙ, true) ≥ 0.7 among pivoters |
| identify *when* | τ MAE among pivoters beats a midpoint guess |
| null is quiet | flag rate ≤ 0.10 when nobody pivoted |

Brand-switch pivot of ±d written reference-normalised as `(-d, -d, -2d, 0)`, half
of households pivoting, 2 replicates, 1000 draws × 4 chains. Zero divergences and
r_hat ≤ 1.02 throughout.

| switch | τ | AUC (who) | corr (what) | τ MAE | midpoint | flag rate, no event |
| --- | --- | --- | --- | --- | --- | --- |
| ±1.0 | known | 0.901 | 0.712 | — | — | 0.088 |
| ±1.0 | estimated | 0.893 | 0.704 | 2.73 | 4.22 | **0.221** ✗ |
| ±1.5 | known | 0.978 | 0.842 | — | — | 0.004 |
| ±1.5 | estimated | 0.944 | 0.846 | 2.40 | 3.59 | 0.015 |
| ±2.5 | known | 0.996 | 0.922 | — | — | 0.004 |
| ±2.5 | estimated | 0.985 | 0.879 | 1.39 | 3.81 | 0.007 |

**Everything passes at ±1.5 and above, with the date known or estimated.** The
envelope is:

- estimated τ: brand switch of **±1.5 or larger**
- known τ: **±1.0 or larger**

Two things worth reading carefully.

**±1.0 fails on the null, not on detection.** Its AUC (0.89) and Δα̂ₙ correlation
(0.70) both clear the bar; what breaks is the false-positive rate, which hits
0.221 when τ has to be estimated and only 0.088 when it is supplied. Searching
over τ is extra freedom, and at weak effect sizes that freedom gets spent finding
pivots in households that never had one. Knowing the date removes it. So the
value of a known event date is not mainly sharper estimates — it is a quieter
null.

**The cost of not knowing the date is modest where the signal is adequate.** At
±2.5, AUC goes 0.996 → 0.985 and correlation 0.922 → 0.879. τ itself is located
to within ~1.4 trips against a 3.8-trip benchmark. The earlier worry that only
~4 trips per household sit in the transition zone turns out to bound τ's
*precision* rather than its identifiability: four informative trips are enough
when each carries a 2.5-logit signal.

Read the null verdict off `frac_flagged_transformed`, not `pi_pivot`. With no
signal `pi_pivot` is weakly identified and reverts toward its prior (0.23–0.42
here) while the per-household responsibilities correctly flag almost nobody.

## Habit read as transformation

A household that switches brand once and then sticks with it traces exactly the
path a pivoting utility function traces. This is not a subtle confound in
scanner data — it is the dominant feature of it. The lagged-choice term is the
control, and the `habit` experiment measures how badly its absence misleads.

Simulating pure state dependence (γ=1.5) with **no pivot whatsoever**:

| fit | spurious pivot rate | `mu_delta_price` (truth 0) | households flagged | `beta_loyalty` (truth 1.5) |
| --- | --- | --- | --- | --- |
| blind to habit | **1.0** | −1.10 | 22% | — |
| controls for habit | **0.0** | −0.17 | 0% | 1.52 ✓ |

Without the control the model reports a transformative preference shift in data
that contains only habit. This is the single most important guard in the harness.

## Notes on the data

`choiceId` in `choice_crackers.csv` is a row counter, not a calendar index, so
there is no shared clock across households; the only time axis is
`personChoiceId`, the household's own trip number.

This costs us nothing. Transformative experience is a claim about an individual
utility function, not about a market: the interesting object is one person's
pivot on their own timeline, which is exactly what the trip index gives. The
absence of a calendar clock only rules out a question we are not asking — a
simultaneous market-wide shock — and `tau_mode="common"` is correspondingly a
contrast arm about tenure rather than an attempt at that question.

Three rows carry `price_nabisco == 0` and are dropped as recording errors.
Households with fewer than `min_trips` observations are dropped: a pivot needs
data on both sides of it.

## Files

- `changepoint_logit.py` — model, simulator, recovery scoring, harness
- `run_tests.py` — the seven experiments
- `results/` — generated CSVs
