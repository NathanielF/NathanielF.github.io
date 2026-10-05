"""Panel plumbing for the change-point choice post.

Deliberately holds no models. The change-point logit and the forward simulator
are written out in the post itself, because they are the argument; what lives
here is the data wrangling and the one arithmetic check that nobody needs to
watch being done.

    prepare_panel         cracker csv -> the (N, J, K) arrays PyMC wants
    lagged_choice_matrix  "did this household buy j last trip?"
    tau_bounds            where a pivot is allowed to fall, and why it matters
    nats_per_household    evidence about each household's own pivot, pre-fit
    auc                   rank-based discrimination, for scoring who pivoted
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Display names. Every choice in the post is simulated, so the alternatives are
# deliberately anonymous: what is borrowed from the source panel is the geometry
# (how much price moves, how unevenly people are observed), not its subject
# matter. _CSV_ALTS maps these onto the columns of the file the geometry is read
# from; nothing downstream should need it.
ALTS = ["A", "B", "C", "D"]
_CSV_ALTS = ["sunshine", "keebler", "nabisco", "private"]
COVARIATES = ["display", "feature", "price"]  # price last; the slope may pivot
_CSV_COVARIATES = ["disp", "feat", "price"]
PRICE_IDX = 2


@dataclass
class Panel:
    """A choice panel in the shape the model consumes."""

    X: np.ndarray            # (N, J, K) attributes, K ordered as COVARIATES
    y: np.ndarray            # (N,) chosen alternative, coded into ALTS
    t: np.ndarray            # (N,) within-household trip number, 1-based
    grp_idx: np.ndarray      # (N,) household index into 0..P-1
    T_p: np.ndarray          # (P,) trips observed per household
    person_id: np.ndarray    # (P,) original personId labels
    L: np.ndarray            # (N, J) lagged-choice indicator
    price_mean: float        # subtracted from price; kept to read coefficients back
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


def lagged_choice_matrix(
    y: np.ndarray, grp_idx: np.ndarray, t: np.ndarray, n_alts: int = len(ALTS)
) -> np.ndarray:
    """(N, J) indicator for "this household chose alternative j on its last trip".

    Rows at a household's first trip are all zero. Assumes rows sorted by
    (household, trip), which :func:`prepare_panel` guarantees.
    """
    L = np.zeros((len(y), n_alts))
    prev: dict[int, int] = {}
    for i in range(len(y)):
        p = int(grp_idx[i])
        if t[i] > 1 and p in prev:
            L[i, prev[p]] = 1.0
        prev[p] = int(y[i])
    return L


def prepare_panel(
    crackers_df: pd.DataFrame,
    min_trips: int = 8,
    center_price: bool = True,
) -> Panel:
    """Reshape the cracker panel into arrays plus index vectors.

    Households with fewer than ``min_trips`` observations are dropped: a pivot
    needs data on both sides of it. Three rows carry ``price_nabisco == 0``,
    which is a recording error rather than free crackers, and go too. Prices are
    centred, which leaves every slope unchanged but decorrelates the slope from
    the intercepts, and the pivot moves both.
    """
    df = crackers_df.copy()
    price_cols = [f"price_{a}" for a in _CSV_ALTS]

    df = df.loc[~(df[price_cols] <= 0).any(axis=1)]
    keep = df.groupby("personId")["personChoiceId"].transform("nunique") >= min_trips
    df = df.loc[keep].sort_values(["personId", "personChoiceId"]).reset_index(drop=True)
    df["t"] = df.groupby("personId").cumcount() + 1

    grp_idx, person_id = pd.factorize(df["personId"])
    T_p = df.groupby("personId")["t"].max().loc[person_id].to_numpy(dtype="float64")

    price = df[price_cols].to_numpy(dtype="float64")
    price_mean = float(price.mean()) if center_price else 0.0

    X = np.stack(
        [
            df[[f"disp_{a}" for a in _CSV_ALTS]].to_numpy(dtype="float64"),
            df[[f"feat_{a}" for a in _CSV_ALTS]].to_numpy(dtype="float64"),
            price - price_mean,
        ],
        axis=-1,
    )
    y = df["choice"].map({a: i for i, a in enumerate(_CSV_ALTS)}).to_numpy()
    grp_idx = grp_idx.astype("int32")
    t = df["t"].to_numpy(dtype="float64")

    return Panel(
        X=X,
        y=y,
        t=t,
        grp_idx=grp_idx,
        T_p=T_p,
        person_id=person_id,
        L=lagged_choice_matrix(y, grp_idx, t),
        price_mean=price_mean,
        coords={
            "alts": ALTS,
            "alts_free": ALTS[:-1],   # all but the reference; shifts live here
            "covariates": COVARIATES,
            "normal_covariates": COVARIATES[:PRICE_IDX],
            "person": person_id.astype(str),
            "obs": np.arange(len(df)),
        },
    )


def tau_bounds(
    T_p: np.ndarray,
    min_frac: float = 0.25,
    pad_widths: float = 3.0,
    s: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Where the pivot date is allowed to fall, per household.

    The window runs from ``min_frac`` of the way into a household's trips to
    ``pad_widths`` transition widths past its last one. Both ends are
    deliberate.

    The late end is open so that "nothing happened" has somewhere to live. A
    pivot placed after the final trip holds the transition weight at zero
    throughout, which is the hinged model reduced to the unhinged one. Without
    that exit the window spans only the observed trips, every household is
    necessarily assigned a pivot date somewhere inside its own sequence, and a
    panel in which nothing happened comes back looking like a panel of located
    events. A change-point model specified that way cannot report a null.

    The early end is closed because the symmetric exit is worth nothing. A pivot
    before the first trip holds the weight at one throughout, leaving utility at
    pre-pivot *plus the shift* — the same no-change model, relabelled so that
    the regime which actually held throughout gets called "after". Permitting it
    adds no hypothesis the late exit does not already cover, and costs: the two
    descriptions are indistinguishable in the likelihood, so the sampler is free
    to oscillate between them and the sign of the shift stops meaning anything.
    Closing it fixes the shift's meaning as movement away from whichever regime
    held at the start.
    """
    lo = 1.0 + min_frac * T_p
    hi = T_p + pad_widths * s
    return lo, np.maximum(hi, lo + 1.0)


def nats_per_household(
    panel: Panel,
    *,
    alphas: np.ndarray,
    betas_nr: np.ndarray,
    beta_price_n: np.ndarray,
    tau: np.ndarray,
    delta_alpha_n: np.ndarray | None = None,
    delta_price_n: np.ndarray | None = None,
    s: float = 0.75,
) -> np.ndarray:
    """Evidence each household holds about its *own* pivot, before any fitting.

    The Kullback-Leibler divergence between the choice probabilities a household
    generates with the pivot and the ones it would generate without, summed over
    its trips. Measured in nats, which converts straight into odds: about 2.2
    nats (log 9) takes a household from even odds to ninety-percent confidence
    that it pivoted. A household holding half a nat cannot be classified by any
    estimator, however good.

    Returns one value per household.
    """
    g = panel.grp_idx
    price = panel.X[:, :, PRICE_IDX]
    w = 1.0 / (1.0 + np.exp(-(panel.t - np.asarray(tau)[g]) / s))

    base = np.asarray(alphas)[None, :] + np.einsum(
        "njk,k->nj", panel.X[:, :, :PRICE_IDX], np.asarray(betas_nr)
    )
    b_pre = np.asarray(beta_price_n)[g]
    b_post = b_pre + w * (
        np.zeros(panel.P) if delta_price_n is None else np.asarray(delta_price_n)
    )[g]
    shift = (
        np.zeros((panel.N, panel.J))
        if delta_alpha_n is None
        else w[:, None] * np.asarray(delta_alpha_n)[g]
    )

    def softmax(V):
        V = V - V.max(-1, keepdims=True)
        E = np.exp(V)
        return E / E.sum(-1, keepdims=True)

    p_null = softmax(base + b_pre[:, None] * price)
    p_pivot = softmax(base + shift + b_post[:, None] * price)
    kl = (p_pivot * (np.log(p_pivot) - np.log(p_null))).sum(-1)
    return np.bincount(g, weights=kl, minlength=panel.P)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Probability a randomly chosen positive outranks a randomly chosen negative.

    0.5 is a coin toss, 1.0 a perfect sort. Ties score 0.5, via averaged ranks.
    """
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
