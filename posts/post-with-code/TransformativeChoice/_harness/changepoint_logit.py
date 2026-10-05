"""Change-point mixed logit: a discrete choice model with a latent utility pivot.

Extends the panel mixed logit of the cracker case study so that each household's
utility function is allowed to pivot at a latent moment tau_n in its own choice
sequence. The pivot can move the price coefficient, the alternative-specific
intercepts, or both.

    w_nt  = sigmoid((t_nt - tau_n) / s)                     transition weight
    U_njt = (alpha_j + w_nt * Delta_alpha_j)
            + (beta_n + w_nt * Delta_n) * price_njt
            + disp_njt * b_disp + feat_njt * b_feat
            + gamma * 1[chose j last trip]                  optional habit term

w runs smoothly from 0 (pre-pivot) to 1 (post-pivot), so tau stays continuous
and NUTS keeps its gradients. s controls how abrupt the pivot is; s -> 0 is a
hard step.

Where tau is allowed to go decides what the model can mean. A change-point model
has two ways to encode "no change" — tau past the last trip (w ~ 0 throughout)
and tau before the first (w ~ 1 throughout) — and both are the null model merely
re-parameterised. Keeping both leaves them aliased, so nothing fixes which
segment is "pre", and delta_alpha acquires a label-switching symmetry that
destroys r_hat.

The default ``tau_policy="late_escape"`` closes the early door and leaves the
late one open. That fixes Delta's meaning as "the shift away from the regime that
held at the start", and it makes a posterior tau piled up past a household's
final trip into a *reported non-discovery*: the model saying "no pivot for this
household", as a number you can read (:func:`discovery_summary`) rather than a
diagnostic failure you have to interpret. Any mixture that survives across
households is then the interesting one, {pivoted} vs {did not}.

Two further failure modes this module exposes rather than hides:

* **Latent-class collapse.** With both doors open, tau runs to the boundaries,
  sigma_tau runs away, and w stops being a transition and becomes a binary class
  label — a two-class segmentation of households, which fits the cracker panel
  far better than any real pivot and means something else entirely.
  :func:`degeneracy_summary` detects it.
* **Habit mistaken for transformation.** A household that switches brand once
  and then sticks traces the same path as one whose utility pivoted. The
  optional lagged-choice term is the control; the ``habit`` experiment in
  ``run_tests.py`` measures how badly its absence misleads.

Module contents
---------------
prepare_panel            real cracker panel -> arrays the model consumes
lagged_choice_matrix     (N, J) "chose j last trip" indicator
make_changepoint_model   the PyMC model (shift_mode selects the nesting)
draw_truth_from_prior    one prior draw, as a flat dict of true values
fixed_truth              hand-specified truth, for null and power tests
simulate_choices         pure-numpy forward simulation from a truth dict
recovery_table           posterior vs. truth, per parameter
tau_bounds               where tau may go, and why the asymmetry matters
tau_diagnostics          pivot-timing error, against a midpoint-guess benchmark
discovery_summary        per-household "pivot found / not found" verdicts
individual_recovery      did we recover each household's own transformed utility?
information_budget       evidence available per household, before any fitting
mixture_summary          population pivot rate vs per-household identification
degeneracy_summary       has the change point stopped being a change point?
detection_summary        does the fit claim a pivot? (false-positive / power)
run_recovery             full replicate loop -> tidy DataFrames
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

ALTS = ["sunshine", "keebler", "nabisco", "private"]
COVARIATES = ["disp", "feat", "price"]  # price MUST stay last; index 2 is the pivoting slope
PRICE_IDX = 2


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


@dataclass
class Panel:
    """Geometry of a choice panel, in the shape the model wants."""

    X: np.ndarray  # (N, J, K) attributes, K ordered as COVARIATES
    y: np.ndarray  # (N,) chosen alternative, integer-coded into ALTS
    t: np.ndarray  # (N,) within-person trip index, 1-based
    grp_idx: np.ndarray  # (N,) person index into 0..P-1
    T_p: np.ndarray  # (P,) trips observed per person
    person_id: np.ndarray  # (P,) original personId labels
    L: np.ndarray | None = None  # (N, J) 1 if alt j was chosen on the previous trip
    coords: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def J(self) -> int:
        return self.X.shape[1]

    @property
    def P(self) -> int:
        return len(self.T_p)


def prepare_panel(
    crackers_df: pd.DataFrame,
    min_trips: int = 8,
    drop_bad_prices: bool = True,
    center_price: bool = True,
) -> Panel:
    """Reshape the cracker panel into (N, J, K) arrays plus index vectors.

    Parameters
    ----------
    min_trips
        Drop households with fewer than this many trips. A pivot needs enough
        observations on both sides of it to be visible at all.
    drop_bad_prices
        Three rows carry ``price_nabisco == 0``, which is a recording error
        rather than a free packet of crackers.
    center_price
        Subtract the grand mean price. Leaves the price *slope* unchanged but
        decorrelates it from the intercepts, which matters here because the
        pivot moves both.
    """
    df = crackers_df.copy()
    price_cols = [f"price_{a}" for a in ALTS]

    if drop_bad_prices:
        bad = (df[price_cols] <= 0).any(axis=1)
        df = df.loc[~bad]

    keep = df.groupby("personId")["personChoiceId"].transform("nunique") >= min_trips
    df = df.loc[keep].sort_values(["personId", "personChoiceId"]).reset_index(drop=True)

    # Re-index trips to a contiguous 1..T_p after any row drops.
    df["t"] = df.groupby("personId").cumcount() + 1

    grp_idx, person_id = pd.factorize(df["personId"])
    T_p = df.groupby("personId")["t"].max().loc[person_id].to_numpy()

    price = df[price_cols].to_numpy(dtype="float64")
    price_mean = price.mean() if center_price else 0.0

    X = np.stack(
        [
            df[[f"disp_{a}" for a in ALTS]].to_numpy(dtype="float64"),
            df[[f"feat_{a}" for a in ALTS]].to_numpy(dtype="float64"),
            price - price_mean,
        ],
        axis=-1,
    )

    y = df["choice"].map({a: i for i, a in enumerate(ALTS)}).to_numpy()

    # Lagged-choice indicator. A household that simply repeats last week's brand
    # is exhibiting habit, not a transformed utility function, and the two are
    # easy to confuse: see `lagged_choice_matrix`.
    L = lagged_choice_matrix(y, grp_idx, df["t"].to_numpy())

    coords = {
        "alts": ALTS,
        "covariates": COVARIATES,
        "normal_covariates": COVARIATES[:PRICE_IDX],
        # All alternatives bar the reference. Intercept shifts are defined over
        # this dimension so the pinned reference never becomes a free parameter
        # that the likelihood cannot see.
        "alts_free": ALTS[:-1],
        "person": person_id.astype(str),
        "obs": np.arange(len(df)),
    }

    panel = Panel(
        X=X,
        y=y,
        t=df["t"].to_numpy(dtype="float64"),
        grp_idx=grp_idx.astype("int32"),
        T_p=T_p.astype("float64"),
        person_id=person_id,
        L=L,
        coords=coords,
    )
    panel.price_mean = price_mean  # kept so coefficients can be read back in dollars
    return panel


def lagged_choice_matrix(
    y: np.ndarray, grp_idx: np.ndarray, t: np.ndarray, n_alts: int = len(ALTS)
) -> np.ndarray:
    """(N, J) indicator for "this household chose alternative j on its last trip".

    Rows at a household's first observed trip are all zero: there is no
    preceding choice to carry. Assumes rows are sorted by (person, trip), which
    :func:`prepare_panel` guarantees.
    """
    L = np.zeros((len(y), n_alts))
    prev = {}
    for i in range(len(y)):
        p = int(grp_idx[i])
        if t[i] > 1 and p in prev:
            L[i, prev[p]] = 1.0
        prev[p] = int(y[i])
    return L


TAU_POLICIES = ("late_escape", "interior")


def tau_bounds(
    T_p: np.ndarray,
    policy: str = "late_escape",
    min_frac: float = 0.25,
    pad_widths: float = 3.0,
    transition_width: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds on tau, per household. The choice of policy matters a great deal.

    ``"late_escape"`` (default) is asymmetric: tau may not occur before
    ``min_frac`` of the panel has elapsed, but it *may* run past the final trip.

    The asymmetry is the point. A change-point model has two ways to say "no
    change", and they are not equally useful:

    * tau past the end  -> w ~ 0 throughout -> fitted utility is ``alpha``
    * tau before the start -> w ~ 1 throughout -> fitted utility is
      ``alpha + Delta_alpha``

    Both are the null model, merely re-parameterised, and keeping both leaves
    them aliased: nothing distinguishes "pre" from "post", chains disagree about
    which segment is which, and ``delta_alpha`` picks up a label-switching
    symmetry that wrecks r_hat. Closing the early door leaves exactly one null
    expression and fixes Delta's meaning as "the shift away from the regime that
    held at the start".

    It also turns the failure into a read-out. A household whose posterior tau
    piles up past its last trip is the model reporting *no pivot found* for that
    household, which is a usable negative result rather than a diagnostic
    nuisance — see :func:`discovery_summary`. And any remaining mixture across
    households is then the substantively interesting one, {pivoted} vs {did
    not}, instead of a degenerate relabelling.

    ``"interior"`` forces tau strictly inside the window, closing *both* doors.
    The null then has to be expressed through ``Delta -> 0`` alone, and the model
    strains to place a mid-panel pivot in data that contains none. Kept for
    comparison; it is not the recommended default.

    Parameters
    ----------
    min_frac
        Fraction of each household's panel that must precede the pivot, so that
        the pre-regime is identified from real data.
    pad_widths
        How far past the last trip tau may run, in units of
        ``transition_width``. The default puts ``w`` below about 0.05 at the
        final trip when tau sits at the upper bound, so "no pivot" is cleanly
        all-pre rather than half-switched.
    """
    if policy not in TAU_POLICIES:
        raise ValueError(f"policy must be one of {TAU_POLICIES}")
    if policy == "interior":
        lo = np.full_like(T_p, 2.0)
        hi = np.maximum(T_p - 1.0, lo + 1.0)
        return lo, hi
    lo = 1.0 + min_frac * T_p
    hi = T_p + pad_widths * transition_width
    return lo, np.maximum(hi, lo + 1.0)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

SHIFT_MODES = ("none", "price", "alpha", "both")
TAU_MODES = ("individual", "common", "known")


def make_changepoint_model(
    panel: Panel,
    shift_mode: str = "both",
    tau_mode: str = "individual",
    transition_width: float = 0.75,
    y: np.ndarray | None = None,
    L: np.ndarray | None = None,
    tau_policy: str = "late_escape",
    tau_interior: tuple[float, float] = (0.2, 0.8),
    tau_min_frac: float = 0.25,
    tau_pad_widths: float = 3.0,
    tau_known: np.ndarray | None = None,
    mixture: bool = False,
    alpha_shift_mode: str = "common",
) -> pm.Model:
    """Build the change-point mixed logit.

    Parameters
    ----------
    shift_mode
        ``"none"`` recovers the chapter's panel mixed logit (no pivot) and is the
        null model. ``"price"`` lets only price sensitivity pivot, ``"alpha"``
        only the brand intercepts, ``"both"`` lets each move.
    tau_mode
        ``"individual"`` gives every household its own latent pivot, partially
        pooled on the logit scale — a change *detection* problem. ``"common"``
        forces one shared pivot fraction. ``"known"`` takes the pivot dates as
        given via ``tau_known`` and estimates only their consequences.

        The third is the one that matches the philosophy. Paul's transformative
        experiences are singular and self-known: you know the date you became a
        parent. What you cannot do beforehand is evaluate the utility function
        you will have afterwards. So the quantity her argument is about is Delta
        *given* tau, not the pair jointly — and conditioning on tau is worth a
        great deal of identification, which ``exp_known_tau`` quantifies.
    transition_width
        s, in trips. Smaller is a sharper step; 0.75 makes the pivot take
        roughly two trips to complete.
    L
        Optional (N, J) lagged-choice indicator. Passing it adds a state
        dependence (habit) term to utility. This matters more than it looks:
        without it, a household that switches brand mid-panel and then sticks
        is indistinguishable from a household whose utility function pivoted,
        so the change-point model will happily read habit as transformation.
    tau_policy
        ``"late_escape"`` (default) forbids an early pivot but lets tau run past
        the final trip, so that "no pivot for this household" has a single,
        readable home. ``"interior"`` confines tau strictly inside the window.
        See :func:`tau_bounds` for why the asymmetry matters.
    tau_interior
        Fractions of the window tau may occupy, used only when
        ``tau_policy="interior"``. ``(0.0, 1.0)`` reproduces the aliased,
        label-switching version deliberately.
    tau_min_frac, tau_pad_widths
        Used only when ``tau_policy="late_escape"``; see :func:`tau_bounds`.
    alpha_shift_mode
        ``"common"`` gives every household the same intercept shift; ``"hierarchical"``
        gives each its own ``delta_alpha_n``, partially pooled.

        Only the hierarchical form can represent an *individually* transformed
        utility function in the intercepts, which is the quantity of interest
        here. It is also far better identified than a shift in the price slope:
        an intercept shift enters utility directly, whereas a slope shift is
        multiplied by the price variation the household happens to face — about
        $0.26 in this panel, so a slope shift of 2.5 buys only ~0.64 logit units
        while an intercept shift of 2.5 buys 2.5. See :func:`information_budget`.
    mixture
        Give "this household never pivoted" its own state and marginalise over
        it, rather than hoping tau or Delta will encode the null implicitly.
        Adds ``pi_pivot`` (the population rate of transformation) and
        ``p_transformed`` (a per-household posterior probability).

        Worth understanding why this is needed. Reading a non-discovery off
        tau's position only works if the change magnitude cannot vanish: then
        tau has nowhere to hide a null except the ends of the series. Here
        Delta *can* shrink to zero, which is the cheaper route, so the
        likelihood goes flat in tau and tau reverts to its prior — mid-window,
        looking for all the world like a located pivot. The mixture makes the
        null an explicit hypothesis with its own probability instead.

        Note that this replaces the ``Categorical`` likelihood with a
        ``Potential``, so these fits are not LOO-comparable against the
        non-mixture ones.
    """
    lo_frac, hi_frac = tau_interior if tau_policy == "interior" else (0.0, 1.0)
    if not 0.0 <= lo_frac < hi_frac <= 1.0:
        raise ValueError("tau_interior must satisfy 0 <= low < high <= 1")
    if shift_mode not in SHIFT_MODES:
        raise ValueError(f"shift_mode must be one of {SHIFT_MODES}")
    if tau_mode not in TAU_MODES:
        raise ValueError(f"tau_mode must be one of {TAU_MODES}")

    y = panel.y if y is None else y
    lo, hi = tau_bounds(
        panel.T_p, policy=tau_policy, min_frac=tau_min_frac,
        pad_widths=tau_pad_widths, transition_width=transition_width,
    )

    with pm.Model(coords=panel.coords) as model:
        X = pm.Data("X", panel.X, dims=("obs", "alts", "covariates"))
        t = pm.Data("t", panel.t, dims="obs")
        grp = pm.Data("grp_idx", panel.grp_idx, dims="obs")
        tau_lo = pm.Data("tau_lo", lo, dims="person")
        tau_hi = pm.Data("tau_hi", hi, dims="person")
        observed = pm.Data("y", y, dims="obs")

        # --- pre-pivot utility ------------------------------------------------
        alphas_ = pm.Normal("alphas_", 0, 1.5, dims="alts")
        alphas = pm.Deterministic(
            "alphas", pt.set_subtensor(alphas_[-1], 0.0), dims="alts"
        )
        betas_non_random = pm.Normal(
            "betas_non_random", 0, 1.0, dims="normal_covariates"
        )

        mu_price = pm.Normal("mu_price", -2.0, 1.5)
        sigma_price = pm.HalfNormal("sigma_price", 1.0)
        z_price = pm.Normal("z_price", 0, 1, dims="person")
        beta_price_n = pm.Deterministic(
            "beta_price_n", mu_price + sigma_price * z_price, dims="person"
        )

        # --- the pivot --------------------------------------------------------
        pivots = shift_mode != "none"
        if pivots and tau_mode == "known":
            if tau_known is None:
                raise ValueError('tau_mode="known" requires tau_known')
            tau = pm.Data("tau", np.asarray(tau_known, dtype="float64"),
                          dims="person")
            w = pm.Deterministic(
                "w", pm.math.sigmoid((t - tau[grp]) / transition_width), dims="obs"
            )
        elif pivots:
            mu_tau = pm.Normal("mu_tau", 0.0, 1.0)
            if tau_mode == "individual":
                sigma_tau = pm.HalfNormal("sigma_tau", 1.0)
                z_tau = pm.Normal("z_tau", 0, 1, dims="person")
                raw = mu_tau + sigma_tau * z_tau
            else:
                raw = mu_tau * pt.ones(panel.P)
            tau_frac = pm.Deterministic(
                "tau_frac",
                lo_frac + pm.math.sigmoid(raw) * (hi_frac - lo_frac),
                dims="person",
            )
            tau = pm.Deterministic(
                "tau", tau_lo + tau_frac * (tau_hi - tau_lo), dims="person"
            )
            w = pm.Deterministic(
                "w", pm.math.sigmoid((t - tau[grp]) / transition_width), dims="obs"
            )
        else:
            w = pt.zeros(panel.N)

        if shift_mode in ("price", "both"):
            mu_delta_price = pm.Normal("mu_delta_price", 0.0, 1.5)
            sigma_delta_price = pm.HalfNormal("sigma_delta_price", 1.0)
            z_delta_price = pm.Normal("z_delta_price", 0, 1, dims="person")
            delta_price_n = pm.Deterministic(
                "delta_price_n",
                mu_delta_price + sigma_delta_price * z_delta_price,
                dims="person",
            )
        else:
            delta_price_n = pt.zeros(panel.P)

        delta_alpha_n = None
        if shift_mode in ("alpha", "both"):
            if alpha_shift_mode == "hierarchical":
                mu_delta_alpha = pm.Normal("mu_delta_alpha", 0.0, 1.0, dims="alts_free")
                sigma_delta_alpha = pm.HalfNormal(
                    "sigma_delta_alpha", 1.0, dims="alts_free"
                )
                z_delta_alpha = pm.Normal(
                    "z_delta_alpha", 0, 1, dims=("person", "alts_free")
                )
                free = mu_delta_alpha + sigma_delta_alpha * z_delta_alpha  # (P, J-1)
                delta_alpha_n = pm.Deterministic(
                    "delta_alpha_n",
                    pt.concatenate([free, pt.zeros((panel.P, 1))], axis=1),
                    dims=("person", "alts"),
                )
                delta_alpha = None
            elif alpha_shift_mode == "common":
                delta_alpha_ = pm.Normal("delta_alpha_", 0.0, 1.0, dims="alts_free")
                delta_alpha = pm.Deterministic(
                    "delta_alpha",
                    pt.concatenate([delta_alpha_, pt.zeros(1)]),
                    dims="alts",
                )
            else:
                raise ValueError('alpha_shift_mode must be "common" or "hierarchical"')
        else:
            delta_alpha = pt.zeros(panel.J)

        # (N, J) intercept contribution, whichever form it came in
        if delta_alpha_n is not None:
            alpha_shift = w[:, None] * delta_alpha_n[grp]
        else:
            alpha_shift = w[:, None] * delta_alpha

        # --- assemble utility -------------------------------------------------
        U_nr = pt.sum(X[:, :, :PRICE_IDX] * betas_non_random, axis=2)  # (N, J)
        beta_t = beta_price_n[grp] + w * delta_price_n[grp]  # (N,)
        U_price = beta_t[:, None] * X[:, :, PRICE_IDX]  # (N, J)

        if L is None:
            U_loyalty = pt.zeros((panel.N, panel.J))
        else:
            L_data = pm.Data("L", L, dims=("obs", "alts"))
            beta_loyalty = pm.Normal("beta_loyalty", 0.0, 1.0)
            U_loyalty = beta_loyalty * L_data

        U = pm.Deterministic(
            "U",
            alphas + alpha_shift + U_nr + U_price + U_loyalty,
            dims=("obs", "alts"),
        )
        p = pm.Deterministic("p", pm.math.softmax(U, axis=1), dims=("obs", "alts"))

        if not mixture or not pivots:
            pm.Categorical("y_cat", p=p, observed=observed, dims="obs")
            return model

        # --- explicit "no change" state ---------------------------------------
        # Asking tau to carry the null geometrically does not work: when Delta
        # is free to shrink to zero the likelihood goes flat in tau, so tau
        # simply reverts to its prior (mid-window) and tells us nothing. The
        # cure is to give "this household never pivoted" its own state and
        # marginalise over it, which yields a per-household posterior
        # probability of transformation instead of a guess read off geometry.
        U_null = alphas + U_nr + beta_price_n[grp][:, None] * X[:, :, PRICE_IDX]
        U_null = U_null + U_loyalty

        rows = pt.arange(panel.N)
        ll_pivot = pt.log(pm.math.softmax(U, axis=1))[rows, observed]  # (N,)
        ll_null = pt.log(pm.math.softmax(U_null, axis=1))[rows, observed]

        # Sum each household's log-likelihood. The pivot is a property of the
        # household, not of the individual trip, so the mixture has to be taken
        # at the household level.
        zeros_p = pt.zeros((panel.P,))
        LL_pivot = pt.inc_subtensor(zeros_p[grp], ll_pivot)
        LL_null = pt.inc_subtensor(zeros_p[grp], ll_null)

        mu_pi = pm.Normal("mu_pi", 0.0, 1.0)
        pi = pm.Deterministic("pi_pivot", pm.math.sigmoid(mu_pi))

        log_pi = pt.log(pi)
        log_1mpi = pt.log1p(-pi)
        pm.Potential(
            "mixture_ll",
            pt.logaddexp(log_pi + LL_pivot, log_1mpi + LL_null).sum(),
        )

        # Responsibility: P(this household pivoted | data, parameters). Its
        # posterior mean is the per-household verdict, and 1 - it is the
        # non-discovery signal reported as a probability.
        pm.Deterministic(
            "p_transformed",
            pm.math.sigmoid((log_pi + LL_pivot) - (log_1mpi + LL_null)),
            dims="person",
        )

    return model


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #

#: Interpretable quantities we score recovery on, by shift_mode.
SCALARS = {
    "none": ["mu_price", "sigma_price"],
    "price": ["mu_price", "sigma_price", "mu_delta_price", "sigma_delta_price",
              "mu_tau", "sigma_tau"],
    "alpha": ["mu_price", "sigma_price", "mu_tau", "sigma_tau"],
    "both": ["mu_price", "sigma_price", "mu_delta_price", "sigma_delta_price",
             "mu_tau", "sigma_tau"],
}
VECTORS = {
    "none": {"alphas": "alts", "betas_non_random": "normal_covariates",
             "beta_price_n": "person"},
    "price": {"alphas": "alts", "betas_non_random": "normal_covariates",
              "beta_price_n": "person", "delta_price_n": "person", "tau": "person"},
    "alpha": {"alphas": "alts", "betas_non_random": "normal_covariates",
              "beta_price_n": "person", "delta_alpha": "alts", "tau": "person"},
    "both": {"alphas": "alts", "betas_non_random": "normal_covariates",
             "beta_price_n": "person", "delta_price_n": "person",
             "delta_alpha": "alts", "tau": "person"},
}


def _tracked(
    shift_mode: str, tau_mode: str, alpha_shift_mode: str = "common"
) -> list[str]:
    names = list(SCALARS[shift_mode]) + list(VECTORS[shift_mode])
    if shift_mode in ("alpha", "both") and alpha_shift_mode == "hierarchical":
        names = [n for n in names if n != "delta_alpha"]
        names += ["delta_alpha_n", "mu_delta_alpha", "sigma_delta_alpha"]
    if tau_mode == "common":
        names = [n for n in names if n != "sigma_tau"]
    elif tau_mode == "known":
        # tau is data, not a parameter, so neither it nor its hyperparameters
        # are scored.
        names = [n for n in names if n not in ("mu_tau", "sigma_tau", "tau")]
    return names


def draw_truth_from_prior(
    panel: Panel,
    shift_mode: str = "both",
    tau_mode: str = "individual",
    transition_width: float = 0.75,
    tau_interior: tuple[float, float] = (0.2, 0.8),
    tau_policy: str = "late_escape",
    tau_min_frac: float = 0.25,
    alpha_shift_mode: str = "common",
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Take a single draw from the prior and return it as a truth dict."""
    model = make_changepoint_model(
        panel, shift_mode=shift_mode, tau_mode=tau_mode,
        transition_width=transition_width, tau_interior=tau_interior,
        tau_policy=tau_policy, tau_min_frac=tau_min_frac,
        alpha_shift_mode=alpha_shift_mode,
    )
    names = _tracked(shift_mode, tau_mode, alpha_shift_mode)
    with model:
        idata = pm.sample_prior_predictive(
            draws=1, var_names=names, random_seed=seed
        )
    prior = idata.prior
    return {n: np.asarray(prior[n].isel(chain=0, draw=0)) for n in names}


def simulate_choices(
    truth: dict,
    panel: Panel,
    transition_width: float = 0.75,
    seed: int = 0,
    gamma_loyalty: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Forward-simulate choices from a truth dict, in plain numpy.

    Deliberately an independent implementation of the utility algebra rather
    than a call back into the PyMC graph, so that a disagreement between this
    and the fitted model shows up as failed recovery instead of hiding.

    ``gamma_loyalty`` adds state dependence: a bonus to the alternative chosen
    on the household's previous trip. Because each choice then feeds the next,
    this branch has to simulate sequentially rather than all at once, and the
    lagged-choice matrix it returns is the one built from the *simulated*
    history, not the observed one.

    Returns ``(y, p, L)``, where ``L`` is ``None`` when ``gamma_loyalty == 0``.
    """
    rng = np.random.default_rng(seed)
    N, J = panel.X.shape[0], panel.X.shape[1]
    P = panel.P

    alphas = np.asarray(truth["alphas"], dtype=float)
    b_nr = np.asarray(truth["betas_non_random"], dtype=float)
    beta_price_n = np.asarray(truth["beta_price_n"], dtype=float)

    delta_price_n = np.asarray(truth.get("delta_price_n", np.zeros(P)), dtype=float)
    # Intercept shift may be common across households (J,) or individual (P, J).
    if "delta_alpha_n" in truth:
        delta_alpha_rows = np.asarray(truth["delta_alpha_n"], dtype=float)[
            panel.grp_idx
        ]  # (N, J)
    else:
        delta_alpha_rows = np.broadcast_to(
            np.asarray(truth.get("delta_alpha", np.zeros(J)), dtype=float), (N, J)
        )

    if "tau" in truth:
        tau = np.asarray(truth["tau"], dtype=float)
        w = 1.0 / (1.0 + np.exp(-(panel.t - tau[panel.grp_idx]) / transition_width))
    else:
        w = np.zeros(N)

    U = (
        alphas[None, :]
        + w[:, None] * delta_alpha_rows
        + np.einsum("njk,k->nj", panel.X[:, :, :PRICE_IDX], b_nr)
        + (beta_price_n[panel.grp_idx] + w * delta_price_n[panel.grp_idx])[:, None]
        * panel.X[:, :, PRICE_IDX]
    )

    def softmax_rows(V):
        V = V - V.max(axis=-1, keepdims=True)
        E = np.exp(V)
        return E / E.sum(axis=-1, keepdims=True)

    if gamma_loyalty == 0.0:
        p = softmax_rows(U)
        y = np.array([rng.choice(J, p=p[i]) for i in range(N)])
        return y, p, None

    # Sequential: each choice shifts the utility of the next one.
    y = np.empty(N, dtype=int)
    p = np.empty((N, J))
    L = np.zeros((N, J))
    prev: dict[int, int] = {}
    for i in range(N):
        g = int(panel.grp_idx[i])
        if panel.t[i] > 1 and g in prev:
            L[i, prev[g]] = 1.0
        p[i] = softmax_rows(U[i] + gamma_loyalty * L[i])
        y[i] = rng.choice(J, p=p[i])
        prev[g] = int(y[i])
    return y, p, L


def fixed_truth(
    panel: Panel,
    *,
    mu_price: float = -2.5,
    sigma_price: float = 0.8,
    mu_delta_price: float = 0.0,
    sigma_delta_price: float = 0.3,
    delta_alpha: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    delta_alpha_mu: tuple[float, ...] | None = None,
    sigma_delta_alpha: float = 0.3,
    alphas: tuple[float, ...] = (-1.0, -1.2, 0.6, 0.0),
    betas_non_random: tuple[float, ...] = (0.5, 0.8),
    tau_frac: float | None = None,
    tau_interior: tuple[float, float] = (0.2, 0.8),
    tau_policy: str = "late_escape",
    tau_min_frac: float = 0.25,
    pivot_rate: float = 1.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Hand-specified ground truth, for null and power tests.

    Defaults give a market with real price heterogeneity but **no** pivot
    (``mu_delta_price = 0``, ``delta_alpha = 0``): the null case. Raise
    ``mu_delta_price`` or ``delta_alpha`` to dial in a pivot of known size.

    Intercept shifts are identified only up to an additive constant, so the
    reference alternative's entry is always forced to zero. A "brand switch" must
    therefore be written relative to the reference: a shift of ``-1.5`` on one
    brand and ``+1.5`` on the reference is the same model as
    ``(-1.5, -1.5, -3.0, 0)``, and only the latter is a legal specification.

    ``delta_alpha_mu`` switches the intercept shift from common to individual:
    each household draws its own ``delta_alpha_n`` around that mean with spread
    ``sigma_delta_alpha``. A *brand switch* — one alternative down, another up,
    e.g. ``(0, 0, -1.5, 1.5)`` — is both the most detectable form of pivot in
    this panel and the most faithful to what a transformative event does to
    preferences. The reference alternative's entry is always forced to zero.

    ``pivot_rate`` below 1.0 makes only that share of households pivot at all;
    the rest get ``delta_price_n = 0`` and are recorded in ``truth["pivoted"]``.
    This is the realistic case, and the only one that distinguishes detecting
    transformation in a population from identifying who it happened to.

    ``tau_frac=None`` (the default) scatters pivots uniformly through each
    household's window. Passing a scalar puts every pivot at the same fraction
    of its own window, which is convenient but makes tau recovery look better
    than it is: at ``tau_frac=0.5`` the truth *is* the midpoint of the prior,
    so a model that learned nothing still scores a perfect hit. Compare against
    ``tau_mae_trips_midpoint_guess`` in :func:`tau_diagnostics` before believing
    any tau result.
    """
    rng = np.random.default_rng(seed)
    lo, hi = tau_bounds(panel.T_p, policy=tau_policy, min_frac=tau_min_frac)
    # Scatter inside the region the model is allowed to search, otherwise some
    # true taus sit outside its support and miscoverage is guaranteed. Under
    # late_escape the upper end of that region means "no pivot", so true pivots
    # are placed in the part of the window that is genuinely observed.
    a, b = tau_interior if tau_policy == "interior" else (0.0, 0.75)
    frac = rng.uniform(a, b, panel.P) if tau_frac is None else np.full(panel.P, tau_frac)

    delta_price_n = mu_delta_price + sigma_delta_price * rng.standard_normal(panel.P)
    pivoted = rng.random(panel.P) < pivot_rate
    delta_price_n = np.where(pivoted, delta_price_n, 0.0)

    extra = {}
    if delta_alpha_mu is not None:
        mu_a = np.asarray(delta_alpha_mu, dtype=float)
        d_alpha_n = mu_a[None, :] + sigma_delta_alpha * rng.standard_normal(
            (panel.P, len(mu_a))
        )
        d_alpha_n[:, -1] = 0.0  # reference alternative defines the scale
        d_alpha_n = np.where(pivoted[:, None], d_alpha_n, 0.0)
        extra["delta_alpha_n"] = d_alpha_n
        extra["mu_delta_alpha"] = mu_a[:-1]
        extra["sigma_delta_alpha"] = np.full(len(mu_a) - 1, sigma_delta_alpha)

    return {
        **extra,
        "pivoted": pivoted,
        "mu_price": np.array(mu_price),
        "sigma_price": np.array(sigma_price),
        "mu_delta_price": np.array(mu_delta_price),
        "sigma_delta_price": np.array(sigma_delta_price),
        "alphas": np.array(alphas, dtype=float),
        "betas_non_random": np.array(betas_non_random, dtype=float),
        "beta_price_n": mu_price + sigma_price * rng.standard_normal(panel.P),
        "delta_price_n": delta_price_n,
        "delta_alpha": np.array(delta_alpha, dtype=float),
        "tau": lo + frac * (hi - lo),
    }


# --------------------------------------------------------------------------- #
# recovery scoring
# --------------------------------------------------------------------------- #


def recovery_table(
    idata: az.InferenceData,
    truth: dict,
    shift_mode: str,
    tau_mode: str,
    hdi_prob: float = 0.94,
    alpha_shift_mode: str = "common",
) -> pd.DataFrame:
    """One row per scalar parameter / per element of each vector parameter.

    Columns
    -------
    truth, post_mean, post_sd
        The comparison itself.
    z
        ``(post_mean - truth) / post_sd``. Systematic departure from 0 is bias;
        spread much wider than 1 means the posterior is overconfident.
    covered
        Whether the truth falls inside the HDI. Across replicates this is the
        calibration check: it should hit ``hdi_prob``.
    contraction
        ``1 - var_post / var_prior``, using the prior draws carried in
        ``idata.prior``. Near 0 means the data said nothing about this
        parameter; near 1 means it was pinned down.
    """
    post = idata.posterior
    prior = idata.prior if "prior" in idata else None
    rows = []

    names = _tracked(shift_mode, tau_mode, alpha_shift_mode)
    for name in names:
        if name not in post or name not in truth:
            continue
        samples = post[name].stack(sample=("chain", "draw"))
        # Ravel, so that 2-D parameters such as delta_alpha_n (person x alt) line
        # up with the flattened posterior below. Both are person-major.
        truth_arr = np.atleast_1d(np.asarray(truth[name], dtype=float)).ravel()
        prior_sd = None
        if prior is not None and name in prior:
            pr = prior[name].stack(sample=("chain", "draw"))
            prior_sd = np.atleast_1d(np.asarray(pr.std(dim="sample"))).ravel()

        flat = samples.values.reshape(-1, samples.sizes["sample"])
        for i in range(flat.shape[0]):
            draws = flat[i]
            tv = truth_arr[i] if truth_arr.size > 1 else truth_arr[0]
            lo, hi = az.hdi(draws, hdi_prob=hdi_prob)
            sd = draws.std(ddof=1)
            psd = None
            if prior_sd is not None:
                psd = prior_sd[i] if prior_sd.size > 1 else prior_sd[0]
            rows.append({
                "parameter": name,
                "index": i if flat.shape[0] > 1 else None,
                "truth": tv,
                "post_mean": draws.mean(),
                "post_sd": sd,
                "hdi_lo": lo,
                "hdi_hi": hi,
                "z": (draws.mean() - tv) / sd if sd > 0 else np.nan,
                "covered": bool(lo <= tv <= hi),
                "contraction": (1 - (sd / psd) ** 2) if psd else np.nan,
            })
    return pd.DataFrame(rows)


def tau_diagnostics(idata: az.InferenceData, truth: dict, panel: Panel) -> dict:
    """How well was the *timing* of the pivot located?

    Only meaningful where a pivot actually exists: a household whose utility
    never moved has no true tau to find, and its posterior will simply return
    the prior. Reported alongside the size of the pivot so the two can be read
    together.
    """
    if "tau" not in idata.posterior or "tau" not in truth:
        return {}
    tau_hat = idata.posterior["tau"].mean(dim=("chain", "draw")).values
    tau_true = np.asarray(truth["tau"], dtype=float)
    err = tau_hat - tau_true
    out = {
        "tau_mae_trips": float(np.mean(np.abs(err))),
        "tau_median_abs_err_trips": float(np.median(np.abs(err))),
        "tau_bias_trips": float(np.mean(err)),
        "tau_corr": float(np.corrcoef(tau_hat, tau_true)[0, 1])
        if np.std(tau_hat) > 0 else np.nan,
    }
    # Same error, but restricted to households with a pivot big enough to see.
    if "delta_price_n" in truth:
        big = np.abs(np.asarray(truth["delta_price_n"])) > np.percentile(
            np.abs(np.asarray(truth["delta_price_n"])), 75
        )
        if big.sum() > 2:
            out["tau_mae_trips_largest_quartile_pivot"] = float(
                np.mean(np.abs(err[big]))
            )
    # Naive benchmark: guess the midpoint of every window.
    lo, hi = tau_bounds(panel.T_p)  # benchmark on the default policy
    out["tau_mae_trips_midpoint_guess"] = float(
        np.mean(np.abs((lo + hi) / 2 - tau_true))
    )
    return out


def discovery_summary(
    idata: az.InferenceData,
    panel: Panel,
    p_threshold: float = 0.5,
) -> dict:
    """Per-household verdicts: for whom did the model actually find a pivot?

    Only meaningful under ``tau_policy="late_escape"``, where tau may run past a
    household's final trip. That region is the model's way of saying "no pivot
    here", so the posterior probability that tau lands inside the observed window
    is a direct, per-household detection statistic — a *non-discovery* reported
    as a number rather than inferred from a failed diagnostic.

    Under ``tau_policy="interior"`` tau cannot leave the window, so these figures
    are meaningless by construction and the function says so.

    Returns
    -------
    p_pivot_mean
        Average over households of P(tau < last trip).
    frac_persons_pivot_found
        Share of households whose P(tau inside window) exceeds ``p_threshold``.
    frac_persons_no_pivot
        Its complement: the non-discovery rate. Under a true null this should be
        high; under a strong true pivot it should be low.
    """
    post = idata.posterior
    if "tau" not in post:
        return {}
    tau = post["tau"]  # (chain, draw, person)
    if tau.size == 0:
        return {}

    last = panel.T_p  # (P,)
    inside = (tau.values < last[None, None, :])  # broadcast over chain/draw
    p_pivot = inside.reshape(-1, len(last)).mean(axis=0)  # (P,)

    reachable = float(np.max(tau.values)) > float(np.min(last))
    return {
        "discovery_stat_valid": bool(reachable),
        "p_pivot_mean": float(p_pivot.mean()),
        "frac_persons_pivot_found": float(np.mean(p_pivot > p_threshold)),
        "frac_persons_no_pivot": float(np.mean(p_pivot <= p_threshold)),
        "p_pivot_q10": float(np.quantile(p_pivot, 0.10)),
        "p_pivot_q90": float(np.quantile(p_pivot, 0.90)),
    }


def information_budget(
    panel: Panel,
    truth: dict,
    transition_width: float = 0.75,
    confidence: float = 0.9,
) -> pd.DataFrame:
    """How much evidence about its own pivot does each household actually hold?

    Computes, per household, the expected log-likelihood ratio between the pivot
    model and the same model with the pivot removed — the KL divergence between
    the two predictive distributions, accumulated over that household's trips,
    in nats. This is the information available to decide whether *that* household
    transformed, and it is an upper bound on what any estimator can extract.

    Run this *before* fitting. It answers "is this detectable at all?" with
    arithmetic instead of a sampler, and it separates two questions that have
    very different answers from the same data:

    * "did anyone transform?" pools every household's evidence, so the budget is
      roughly P times the per-household figure — typically overwhelming;
    * "did *this* household transform?" gets the per-household figure alone.

    The gap between those is not interference between levels. It is the ratio of
    the denominators, and it is why population rates come out crisp while
    individual verdicts stay vague.

    Returns a one-row frame with the distribution of per-household evidence, the
    share clearing the threshold implied by ``confidence``, and the structural
    reasons behind it: how much price variation each household sees, and how many
    of its trips fall in the transition zone where tau is actually identified.
    """
    thresh = np.log(confidence / (1 - confidence))  # 2.20 nats at 90%

    g, X, t = panel.grp_idx, panel.X, panel.t
    w = 1.0 / (1.0 + np.exp(-(t - np.asarray(truth["tau"])[g]) / transition_width))
    base = np.asarray(truth["alphas"])[None, :] + np.einsum(
        "njk,k->nj", X[:, :, :PRICE_IDX], np.asarray(truth["betas_non_random"])
    )
    b_pre = np.asarray(truth["beta_price_n"])[g]
    b_post = b_pre + w * np.asarray(truth.get("delta_price_n", np.zeros(panel.P)))[g]
    price = X[:, :, PRICE_IDX]

    if "delta_alpha_n" in truth:
        d_alpha_rows = np.asarray(truth["delta_alpha_n"], dtype=float)[g]
    else:
        d_alpha_rows = np.broadcast_to(
            np.asarray(truth.get("delta_alpha", np.zeros(panel.J)), dtype=float),
            (panel.N, panel.J),
        )

    def softmax(V):
        V = V - V.max(-1, keepdims=True)
        E = np.exp(V)
        return E / E.sum(-1, keepdims=True)

    p_null = softmax(base + b_pre[:, None] * price)
    p_pivot = softmax(base + w[:, None] * d_alpha_rows + b_post[:, None] * price)
    kl = (p_pivot * (np.log(p_pivot) - np.log(p_null))).sum(-1)  # (N,)
    llr = np.bincount(g, weights=kl, minlength=panel.P)

    # Only trips where w is genuinely in transit carry information about *when*.
    in_transit = np.bincount(
        g, weights=((w > 0.05) & (w < 0.95)).astype(float), minlength=panel.P
    )
    per_house_price_range = np.array([
        (lambda q: q.max() - q.min())(price[g == p].mean(axis=1))
        for p in range(panel.P)
    ])

    out = {}
    if "pivoted" in truth:
        # Households with no true pivot have nothing to find, so averaging over
        # them understates what the identifiable cases actually carry.
        piv = np.asarray(truth["pivoted"]).astype(bool)
        if piv.any():
            out["llr_median_among_pivoters"] = float(np.median(llr[piv]))
            out["frac_identifiable_among_pivoters"] = float(np.mean(llr[piv] > thresh))

    return pd.DataFrame([{
        **out,
        "nats_threshold": thresh,
        "llr_median": float(np.median(llr)),
        "llr_q25": float(np.quantile(llr, 0.25)),
        "llr_q75": float(np.quantile(llr, 0.75)),
        "frac_households_identifiable": float(np.mean(llr > thresh)),
        "pooled_nats_population_question": float(llr.sum()),
        "trips_median": float(np.median(panel.T_p)),
        "trips_in_transition_median": float(np.median(in_transit)),
        "price_range_median": float(np.median(per_house_price_range)),
    }])


def individual_recovery(
    idata: az.InferenceData,
    truth: dict,
    panel: Panel,
    hdi_prob: float = 0.94,
) -> dict:
    """Did we recover each household's *own* transformed utility function?

    The headline question. Scored only on households that genuinely pivoted: a
    household with no true shift has nothing to recover, and its posterior
    correctly reverts to the population, so including it would flatter a
    correlation (the zeros line up with the shrunk estimates) while telling us
    nothing about identification.

    Reports, among true pivoters: the correlation and RMSE between estimated and
    true per-household intercept shifts, interval coverage, and how well the
    pivot date was located relative to a midpoint guess.
    """
    post = idata.posterior
    out = {}
    piv = np.asarray(truth.get("pivoted", np.ones(panel.P, bool))).astype(bool)
    if piv.sum() < 3:
        return out
    out["n_true_pivoters"] = int(piv.sum())

    if "delta_alpha_n" in post and "delta_alpha_n" in truth:
        est = post["delta_alpha_n"].mean(dim=("chain", "draw")).values  # (P, J)
        true = np.asarray(truth["delta_alpha_n"], dtype=float)
        h = az.hdi(post, var_names=["delta_alpha_n"], hdi_prob=hdi_prob)[
            "delta_alpha_n"
        ]
        lo = h.sel(hdi="lower").values
        hi = h.sel(hdi="higher").values
        # Reference alternative is pinned at zero in both, so exclude it.
        e, tv = est[piv, :-1].ravel(), true[piv, :-1].ravel()
        out["delta_alpha_n_corr"] = (
            float(np.corrcoef(e, tv)[0, 1]) if e.std() > 0 and tv.std() > 0 else np.nan
        )
        out["delta_alpha_n_rmse"] = float(np.sqrt(np.mean((e - tv) ** 2)))
        out["delta_alpha_n_coverage"] = float(
            np.mean((lo[piv, :-1] <= true[piv, :-1]) & (true[piv, :-1] <= hi[piv, :-1]))
        )

    if "delta_price_n" in post and "delta_price_n" in truth:
        est = post["delta_price_n"].mean(dim=("chain", "draw")).values
        true = np.asarray(truth["delta_price_n"], dtype=float)
        if true[piv].std() > 0 and est[piv].std() > 0:
            out["delta_price_n_corr"] = float(np.corrcoef(est[piv], true[piv])[0, 1])

    if "tau" in post and "tau" in truth:
        tau_hat = post["tau"].mean(dim=("chain", "draw")).values
        tau_true = np.asarray(truth["tau"], dtype=float)
        err = np.abs(tau_hat[piv] - tau_true[piv])
        lo_b, hi_b = tau_bounds(panel.T_p)
        mid_err = np.abs(((lo_b + hi_b) / 2)[piv] - tau_true[piv])
        out["tau_mae_pivoters"] = float(err.mean())
        out["tau_mae_pivoters_midpoint_guess"] = float(mid_err.mean())
        out["tau_beats_midpoint"] = bool(err.mean() < mid_err.mean())
        if tau_hat[piv].std() > 0:
            out["tau_corr_pivoters"] = float(
                np.corrcoef(tau_hat[piv], tau_true[piv])[0, 1]
            )
    return out


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC. Avoids pulling in a dependency for five lines of work."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties, so ties score 0.5 rather than being rewarded
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mixture_summary(
    idata: az.InferenceData,
    pivoted: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict:
    """Read the explicit "no change" state: who transformed, and how surely?

    Requires a fit built with ``mixture=True``. ``pivoted`` is the known
    per-household truth when one exists (simulation), which lets us separate two
    very different abilities: detecting that transformation is happening *in the
    population*, and identifying *which households* it happened to. The first is
    far easier than the second, and conflating them flatters the model.
    """
    post = idata.posterior
    if "p_transformed" not in post:
        return {}
    p_tr = post["p_transformed"].mean(dim=("chain", "draw")).values
    out = {
        "pi_pivot_mean": float(post["pi_pivot"].mean()) if "pi_pivot" in post else np.nan,
        "p_transformed_mean": float(p_tr.mean()),
        "frac_flagged_transformed": float(np.mean(p_tr > threshold)),
        "frac_flagged_not_transformed": float(np.mean(p_tr <= threshold)),
    }
    if pivoted is not None:
        pivoted = np.asarray(pivoted).astype(bool)
        out["true_pivot_rate"] = float(pivoted.mean())
        if pivoted.any():
            out["p_transformed_mean_true_pivoters"] = float(p_tr[pivoted].mean())
        if (~pivoted).any():
            out["p_transformed_mean_true_non_pivoters"] = float(p_tr[~pivoted].mean())
        out["auc_individual_discrimination"] = _auc(pivoted, p_tr)
    return out


def degeneracy_summary(idata: az.InferenceData, panel: Panel) -> dict:
    """Is the fitted "change point" actually a change point?

    A pivot is only a pivot if the transition weight ``w`` moves appreciably
    *within* a household's own observed window. When it does not — when ``w`` is
    pinned near 0 or near 1 across every one of that household's trips — the
    model has quietly stopped doing change-point detection and started doing
    two-class segmentation: ``w`` is a class label, ``delta`` is the gap between
    classes, and tau is unidentified. That fits better and means something
    entirely different, so it needs to be detected rather than celebrated.

    The measure that matters is how many of a household's *own* trips fall on
    each side of its pivot. The span of ``w`` is not that measure and must not
    be used as a proxy: a tau parked hard against the edge of the window still
    sweeps ``w`` from near 0 to near 1 across a long panel, so the span looks
    healthy while one regime holds a single observation.

    Reports, per fit: the share of households with one regime reduced to at most
    one trip, the share whose tau has been pushed against its allowed bounds,
    and whether ``sigma_tau`` has run away (the mechanism of the collapse).
    """
    out = {}
    post = idata.posterior
    if "w" not in post:
        return out

    w = post["w"].mean(dim=("chain", "draw")).values  # (N,)
    min_side = np.empty(panel.P)
    for p in range(panel.P):
        wp = w[panel.grp_idx == p]
        min_side[p] = min((wp < 0.5).sum(), (wp >= 0.5).sum())
    out["min_regime_trips_median"] = float(np.median(min_side))
    out["frac_persons_one_sided"] = float(np.mean(min_side <= 1))
    out["frac_obs_w_saturated"] = float(np.mean((w < 0.02) | (w > 0.98)))

    if "tau_frac" in post:
        tf = post["tau_frac"].mean(dim=("chain", "draw")).values
        # tau_frac is already mapped into [lo_frac, hi_frac]; being pinned to
        # either end of that range is the signature of the collapse.
        span = tf.max() - tf.min()
        edge = 0.02 * max(span, 1e-9)
        out["frac_persons_tau_at_bound"] = float(
            np.mean((tf <= tf.min() + edge) | (tf >= tf.max() - edge))
        )
    if "sigma_tau" in post:
        out["sigma_tau_mean"] = float(post["sigma_tau"].mean())
        # On the logit scale, much above ~3 turns the sigmoid into an indicator.
        out["sigma_tau_runaway"] = bool(out["sigma_tau_mean"] > 3.0)
    return out


def detection_summary(
    idata: az.InferenceData,
    hdi_prob: float = 0.94,
    negligible: float = 0.15,
) -> dict:
    """Does the fit *claim* a pivot? The false-positive / power read-out.

    ``mu_delta_price`` and ``delta_alpha`` are unbounded, so "HDI excludes zero"
    is a meaningful verdict for them. ``sigma_delta_price`` is not: it has
    positive support, so its HDI can never contain zero and an "excludes zero"
    test on it would return True for every fit ever run, including a pure null.
    For that one we report the posterior mean and the posterior probability that
    it is negligibly small instead.

    ``any_pivot_claimed`` is the headline: across null replicates its mean is
    the false-positive rate, across power replicates its mean is the power.
    """
    out = {}
    post = idata.posterior
    claims = []

    if "mu_delta_price" in post:
        d = post["mu_delta_price"].values.ravel()
        lo, hi = az.hdi(d, hdi_prob=hdi_prob)
        out["mu_delta_price_mean"] = float(d.mean())
        out["mu_delta_price_hdi_lo"] = float(lo)
        out["mu_delta_price_hdi_hi"] = float(hi)
        out["mu_delta_price_excludes_0"] = bool(lo > 0 or hi < 0)
        claims.append(out["mu_delta_price_excludes_0"])

    if "sigma_delta_price" in post:
        d = post["sigma_delta_price"].values.ravel()
        out["sigma_delta_price_mean"] = float(d.mean())
        out["sigma_delta_price_p_negligible"] = float(np.mean(d < negligible))

    if "delta_alpha" in post:
        d = post["delta_alpha"]
        for j, alt in enumerate(d.coords["alts"].values[:-1]):
            dj = d.isel(alts=j).values.ravel()
            lo, hi = az.hdi(dj, hdi_prob=hdi_prob)
            out[f"delta_alpha[{alt}]_mean"] = float(dj.mean())
            out[f"delta_alpha[{alt}]_excludes_0"] = bool(lo > 0 or hi < 0)
            claims.append(out[f"delta_alpha[{alt}]_excludes_0"])

    if "delta_price_n" in post:
        h = az.hdi(post, var_names=["delta_price_n"], hdi_prob=hdi_prob)
        lo = h["delta_price_n"].sel(hdi="lower").values
        hi = h["delta_price_n"].sel(hdi="higher").values
        frac = float(np.mean((lo > 0) | (hi < 0)))
        out["frac_persons_delta_excludes_0"] = frac
        # A handful of flagged households is what 94% intervals produce by
        # construction; a real individual-level signal has to clear that.
        claims.append(frac > 1 - hdi_prob)

    out["any_pivot_claimed"] = bool(any(claims))
    return out


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #


#: Derived (N, J)-sized quantities. Excluded from convergence summaries: they
#: are functions of the parameters, and their r_hat is dominated by observations
#: whose w sits on a knife edge, which tells you nothing the parameters don't.
_DERIVED = ("U", "p", "w")


def _agg_diag(ds, reducer, exclude: tuple[str, ...] = _DERIVED) -> float:
    """Reduce an ArviZ diagnostic dataset to one number, ignoring NaNs.

    The reference alternative's intercept is pinned to zero, so its r_hat and
    ess come back NaN. Without filtering, those NaNs would silently swallow the
    diagnostic for every other parameter.
    """
    vals = np.concatenate(
        [
            np.atleast_1d(v.values).ravel().astype(float)
            for k, v in ds.data_vars.items()
            if k not in exclude
        ]
        or [np.array([np.nan])]
    )
    vals = vals[np.isfinite(vals)]
    return float(reducer(vals)) if vals.size else float("nan")


def fit(
    panel: Panel,
    y: np.ndarray,
    shift_mode: str = "both",
    tau_mode: str = "individual",
    transition_width: float = 0.75,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 0,
    nuts_sampler: str = "numpyro",
    prior_draws: int = 2000,
    log_likelihood: bool = False,
    L: np.ndarray | None = None,
    tau_policy: str = "late_escape",
    tau_interior: tuple[float, float] = (0.2, 0.8),
    tau_min_frac: float = 0.25,
    tau_pad_widths: float = 3.0,
    tau_known: np.ndarray | None = None,
    mixture: bool = False,
    alpha_shift_mode: str = "common",
) -> az.InferenceData:
    """Fit the change-point model to a given choice vector."""
    model = make_changepoint_model(
        panel, shift_mode=shift_mode, tau_mode=tau_mode,
        transition_width=transition_width, y=y, L=L,
        tau_policy=tau_policy, tau_interior=tau_interior,
        tau_min_frac=tau_min_frac, tau_pad_widths=tau_pad_widths,
        tau_known=tau_known, mixture=mixture,
        alpha_shift_mode=alpha_shift_mode,
    )
    # Everything except the big (N, J) intermediates, which we never score and
    # which would blow up the prior-predictive object.
    skip = {"U", "p", "w", "U_null"}
    keep = sorted(
        {v.name for v in model.free_RVs + model.deterministics} - skip
    )
    with model:
        idata = pm.sample_prior_predictive(
            draws=prior_draws, var_names=keep, random_seed=seed
        )
        idata.extend(
            pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                nuts_sampler=nuts_sampler,
                random_seed=seed,
                progressbar=False,
                idata_kwargs={"log_likelihood": log_likelihood},
            )
        )
    return idata


def run_recovery(
    panel: Panel,
    n_reps: int = 4,
    source: str = "prior",
    shift_mode: str = "both",
    tau_mode: str = "individual",
    transition_width: float = 0.75,
    truth_kwargs: dict | None = None,
    fit_shift_mode: str | None = None,
    gamma_loyalty: float = 0.0,
    fit_loyalty: bool = False,
    tau_interior: tuple[float, float] = (0.2, 0.8),
    tau_policy: str = "late_escape",
    tau_min_frac: float = 0.25,
    fit_tau_mode: str | None = None,
    mixture: bool = False,
    alpha_shift_mode: str = "common",
    seed0: int = 100,
    verbose: bool = True,
    **fit_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate, refit, score. Repeat.

    Parameters
    ----------
    source
        ``"prior"`` draws each replicate's truth from the model's own prior —
        the calibration check, where coverage should hit the nominal level.
        ``"fixed"`` uses :func:`fixed_truth` so that the pivot's size is set by
        hand, which is what a null or power test needs.
    fit_shift_mode
        Fit under a different specification than the one that generated the
        data. Setting data-generating ``shift_mode="none"`` against
        ``fit_shift_mode="both"`` is the false-positive test.
    gamma_loyalty
        Strength of state dependence in the *simulated* data.
    fit_loyalty
        Whether the fitted model is allowed a habit term. Generating with
        ``gamma_loyalty > 0`` and fitting with ``fit_loyalty=False`` asks
        whether habit gets misread as a transformative pivot; flipping
        ``fit_loyalty`` to True on the same data asks whether controlling for
        habit makes the spurious pivot go away.

    Returns
    -------
    params, summary
        Per-parameter rows across replicates, and one row per replicate with
        sampler diagnostics, tau localisation, and the detection verdict.
    """
    truth_kwargs = truth_kwargs or {}
    fit_mode = fit_shift_mode or shift_mode
    fit_tau = fit_tau_mode or tau_mode

    param_frames, summaries = [], []
    for rep in range(n_reps):
        seed = seed0 + rep
        t0 = time.time()

        if source == "prior":
            truth = draw_truth_from_prior(
                panel, shift_mode=shift_mode,
                tau_mode="individual" if tau_mode == "known" else tau_mode,
                transition_width=transition_width, tau_interior=tau_interior,
                tau_policy=tau_policy, tau_min_frac=tau_min_frac,
                alpha_shift_mode=alpha_shift_mode, seed=seed,
            )
        elif source == "fixed":
            truth = fixed_truth(
                panel, seed=seed, tau_interior=tau_interior,
                tau_policy=tau_policy, tau_min_frac=tau_min_frac, **truth_kwargs
            )
        else:
            raise ValueError("source must be 'prior' or 'fixed'")

        y_sim, p_sim, L_sim = simulate_choices(
            truth, panel, transition_width=transition_width, seed=seed,
            gamma_loyalty=gamma_loyalty,
        )
        # Fit with the habit matrix implied by the simulated history, so the
        # control is the correct one rather than the observed panel's.
        L_fit = None
        if fit_loyalty:
            L_fit = L_sim if L_sim is not None else lagged_choice_matrix(
                y_sim, panel.grp_idx, panel.t
            )
        idata = fit(
            panel, y_sim, shift_mode=fit_mode, tau_mode=fit_tau,
            transition_width=transition_width, seed=seed, L=L_fit,
            tau_interior=tau_interior, tau_policy=tau_policy,
            tau_min_frac=tau_min_frac,
            # Conditioning on the event dates is only legitimate because they
            # are genuinely known in Paul's setting; here we hand the fit the
            # same tau the simulation used.
            tau_known=truth.get("tau") if fit_tau == "known" else None,
            mixture=mixture, alpha_shift_mode=alpha_shift_mode, **fit_kwargs,
        )

        tbl = recovery_table(
            idata, truth, fit_mode, fit_tau, alpha_shift_mode=alpha_shift_mode
        )
        tbl["rep"] = rep
        param_frames.append(tbl)

        row = {
            "rep": rep,
            "source": source,
            "gen_shift_mode": shift_mode,
            "fit_shift_mode": fit_mode,
            "tau_mode": tau_mode,
            "fit_tau_mode": fit_tau,
            "gamma_loyalty_true": gamma_loyalty,
            "fit_loyalty": fit_loyalty,
            "tau_policy": tau_policy,
            "mixture": mixture,
            "alpha_shift_mode": alpha_shift_mode,
            "tau_interior": str(tau_interior),
            "n_divergences": int(idata.sample_stats["diverging"].values.sum()),
            "max_rhat": _agg_diag(az.rhat(idata), np.max),
            "min_ess": _agg_diag(az.ess(idata), np.min),
            "sim_share_entropy": float(
                -np.sum(
                    (bc := np.bincount(y_sim, minlength=panel.J) / len(y_sim))
                    * np.log(bc + 1e-12)
                )
            ),
            "coverage": float(tbl["covered"].mean()),
            "mean_abs_z": float(tbl["z"].abs().mean()),
            "seconds": round(time.time() - t0, 1),
        }
        row.update(tau_diagnostics(idata, truth, panel))
        row.update(individual_recovery(idata, truth, panel))
        row.update(mixture_summary(idata, truth.get("pivoted")))
        row.update(discovery_summary(idata, panel))
        row.update(degeneracy_summary(idata, panel))
        row.update(detection_summary(idata))
        if "beta_loyalty" in idata.posterior:
            row["beta_loyalty_mean"] = float(idata.posterior["beta_loyalty"].mean())
        summaries.append(row)

        if verbose:
            print(
                f"[rep {rep}] {source}/{shift_mode}->{fit_mode} "
                f"cov={row['coverage']:.2f} |z|={row['mean_abs_z']:.2f} "
                f"div={row['n_divergences']} rhat={row['max_rhat']:.3f} "
                f"({row['seconds']}s)",
                flush=True,
            )

    return pd.concat(param_frames, ignore_index=True), pd.DataFrame(summaries)


def coverage_report(params: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-parameter rows into a per-parameter-family verdict."""
    return (
        params.groupby("parameter")
        .agg(
            n=("covered", "size"),
            coverage=("covered", "mean"),
            mean_z=("z", "mean"),
            sd_z=("z", "std"),
            mean_abs_z=("z", lambda s: s.abs().mean()),
            mean_contraction=("contraction", "mean"),
        )
        .sort_values("mean_contraction", ascending=False)
        .round(3)
    )
