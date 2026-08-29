"""winprob.py — F6 train_winprob, F7 score_winprob, F8 explain_winprob.

Model: HistGradientBoostingClassifier (gradient-boosted trees, LightGBM-family).
Chosen over LightGBM to avoid native-build issues; handles NaN + categorical
natively. Swappable behind this module's interface.

Explainability (G5, hard requirement): true per-deal SHAP drivers. SHAP's
TreeExplainer cannot read pandas string categories, so categoricals are encoded
to integer codes using the categories captured at TRAIN time (kept in the
WinProbModel bundle) — the same codes the trees split on. A global-importance
fallback guarantees a score is never shown without a reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from . import schema as S

RANDOM_STATE = 42


@dataclass
class WinProbModel:
    """Trained bundle: estimator + category orderings + optional calibrator.

    `calibrator` is a prefit Platt (sigmoid) wrapper over `estimator`. Sigmoid is
    monotonic, so it preserves the estimator's ranking (and SHAP drivers explain
    `estimator` directly) while pulling the raw model's overconfident probabilities
    toward observed win-rates (verified: mean predicted 75%→66% vs ~58% actual).
    """
    estimator: HistGradientBoostingClassifier
    categories: dict[str, list]
    calibrator: object = None
    metrics: dict = field(default_factory=dict)


def _prepare(X: pd.DataFrame, categories: dict[str, list] | None = None) -> pd.DataFrame:
    """Set category dtype. If `categories` given, pin to TRAIN orderings so codes
    are stable between train/score/explain."""
    X = X.copy()
    for col in S.CATEGORICAL_FEATURES:
        if col not in X.columns:
            continue
        if categories is not None:
            X[col] = pd.Categorical(X[col], categories=categories[col])
        else:
            X[col] = X[col].astype("category")
    return X


def _capture_categories(X: pd.DataFrame) -> dict[str, list]:
    return {
        col: list(X[col].astype("category").cat.categories)
        for col in S.CATEGORICAL_FEATURES
        if col in X.columns
    }


def _to_codes(Xp: pd.DataFrame) -> pd.DataFrame:
    """Numeric matrix for SHAP: categoricals → integer codes (-1 → NaN)."""
    out = Xp.copy()
    for col in S.CATEGORICAL_FEATURES:
        if col in out.columns:
            codes = out[col].cat.codes.astype("float64")
            out[col] = codes.where(codes >= 0, np.nan)
    return out


def train_winprob(X: pd.DataFrame, y: pd.Series) -> WinProbModel:
    """F6 — train on CLOSED deals only (y in {0,1}). Reports cross-val AUC."""
    mask = y.notna()
    categories = _capture_categories(X[mask])
    Xc, yc = _prepare(X[mask], categories), y[mask].astype(int)
    if yc.nunique() < 2:
        raise ValueError("training needs both Won and Lost examples")

    # Regularized for the small mature training set (~360 deals): shallower trees,
    # stronger L2, fewer iters, larger leaves. Verified 2026-06-06 to fix the
    # overconfidence ("สูงเกินจริง") AND raise OOS AUC (0.757→0.796, mean prob
    # 82%→75%, deals ≥95% 39→15) — the overfit was costing accuracy too, so
    # regularizing beat both post-hoc calibration (which traded away AUC) and
    # feature pruning (G1/G3 were already inert).
    est = HistGradientBoostingClassifier(
        categorical_features=S.CATEGORICAL_FEATURES,
        learning_rate=0.05,
        max_iter=200,
        max_depth=3,
        l2_regularization=5.0,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    metrics: dict = {"n_train": int(len(yc)), "n_pos": int(yc.sum())}
    n_splits = min(5, int(yc.value_counts().min()))
    if n_splits >= 2:
        auc = cross_val_score(est, Xc, yc, cv=n_splits, scoring="roc_auc")
        metrics["auc_cv"] = float(np.mean(auc))
        metrics["auc_cv_std"] = float(np.std(auc))

    # Platt (sigmoid) calibration on a held-out slice so probabilities match
    # observed win-rates (the raw model is overconfident on the small training
    # set). Prefit on a single estimator → ranking + SHAP stay intact; sigmoid is
    # robust on little data. Needs both classes in the calibration slice.
    calibrator = None
    min_class = int(yc.value_counts().min())
    if min_class >= 12:
        from sklearn.model_selection import train_test_split

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            Xc, yc, test_size=0.25, stratify=yc, random_state=RANDOM_STATE
        )
        est.fit(X_fit, y_fit)
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.frozen import FrozenEstimator

            calibrator = CalibratedClassifierCV(
                FrozenEstimator(est), method="sigmoid"
            ).fit(X_cal, y_cal)
            metrics["calibrated"] = "sigmoid-prefit"
        except Exception:
            calibrator = None
            est.fit(Xc, yc)          # fall back to a full-data fit, uncalibrated
            metrics["calibrated"] = False
    else:
        est.fit(Xc, yc)
        metrics["calibrated"] = False

    return WinProbModel(
        estimator=est, categories=categories, calibrator=calibrator, metrics=metrics
    )


def score_winprob(wm: WinProbModel, X: pd.DataFrame) -> np.ndarray:
    """F7 — return calibrated P(win) per row (raw estimator if uncalibrated)."""
    Xp = _prepare(X, wm.categories)
    if getattr(wm, "calibrator", None) is not None:
        return wm.calibrator.predict_proba(Xp)[:, 1]
    return wm.estimator.predict_proba(Xp)[:, 1]


def backtest_winprob(
    X: pd.DataFrame,
    y: pd.Series,
    order: pd.Series | None = None,
    test_frac: float = 0.30,
    n_bins: int = 10,
    return_split: bool = False,
) -> dict:
    """Temporal holdout backtest on CLOSED deals only.

    Sorts closed deals by `order` (e.g. close date), trains on the earliest
    (1 - test_frac), evaluates on the most recent `test_frac` — a true
    point-in-time "predict the future from the past" test, no look-ahead.
    If `order` is None, the existing row order is assumed chronological.

    Returns out-of-sample metrics: AUC, Brier, accuracy@0.5, base rates, and a
    calibration table (predicted vs actual win-rate per probability bin).
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")

    mask = y.notna().to_numpy()
    Xc = X[mask]
    yc = y[mask].astype(int)
    n = len(yc)

    if order is not None:
        ordv = order[mask].to_numpy()
        idx = np.argsort(ordv, kind="stable")
    else:
        ordv = np.arange(n)
        idx = np.arange(n)

    Xc, yc, ordv = Xc.iloc[idx], yc.iloc[idx], ordv[idx]

    n_test = int(round(n * test_frac))
    if n_test < 1 or n - n_test < 2:
        raise ValueError(
            f"not enough closed deals to split: n={n}, test_frac={test_frac}"
        )
    n_train = n - n_test

    X_tr, y_tr = Xc.iloc[:n_train], yc.iloc[:n_train]
    X_te, y_te = Xc.iloc[n_train:], yc.iloc[n_train:]
    if y_tr.nunique() < 2:
        raise ValueError("train split is single-class — cannot fit")

    wm = train_winprob(X_tr, y_tr)
    p = score_winprob(wm, X_te)

    out = {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_base_rate": round(float(y_tr.mean()), 4),
        "test_base_rate": round(float(y_te.mean()), 4),
        "auc": _safe_auc(y_te.to_numpy(), p),
        "brier": _brier(y_te.to_numpy(), p),
        "accuracy": round(float(((p >= 0.5).astype(int) == y_te.to_numpy()).mean()), 4),
        "calibration": _calibration(y_te.to_numpy(), p, n_bins),
    }
    if return_split:
        out["_train_order_max"] = ordv[:n_train].max()
        out["_test_order_min"] = ordv[n_train:].min()
    return out


def _safe_auc(y_true, p) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:   # AUC undefined on a single-class test split
        return None
    return round(float(roc_auc_score(y_true, p)), 4)


def _brier(y_true, p) -> float:
    return round(float(np.mean((p - y_true) ** 2)), 4)


def _calibration(y_true, p, n_bins: int) -> list[dict]:
    """Bin predictions into [0,1] slices; compare mean predicted vs actual rate."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    binid = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = binid == b
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": cnt,
                "mean_pred": round(float(p[sel].mean()), 4),
                "actual_rate": round(float(y_true[sel].mean()), 4),
            }
        )
    return rows


def explain_winprob(wm: WinProbModel, X: pd.DataFrame, top_n: int = 3) -> list[list[dict]]:
    """F8 — per-row top drivers: {feature, value, impact}. impact>0 → toward Win.

    top_n=None returns ALL features (full SHAP vector). Default 3 is kept for
    backward compat; pipeline.py calls with top_n=None to persist all drivers.
    """
    Xp = _prepare(X, wm.categories)
    Xcodes = _to_codes(Xp)
    feat_names = list(Xp.columns)

    try:
        import shap

        explainer = shap.TreeExplainer(wm.estimator)
        sv = explainer.shap_values(Xcodes)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:           # (n, features, classes) → positive class
            sv = sv[:, :, -1]
        return _topn(sv, Xp, feat_names, top_n)
    except Exception:
        return _fallback_global(wm.estimator, Xcodes, Xp, feat_names, top_n)


def _topn(sv, Xp, feat_names, top_n):
    out: list[list[dict]] = []
    for i in range(sv.shape[0]):
        row = sv[i]
        order = np.argsort(np.abs(row))[::-1]
        if top_n is not None:
            order = order[:top_n]
        out.append(
            [
                {
                    "feature": feat_names[j],
                    "value": _clean(Xp.iloc[i, j]),
                    "impact": round(float(row[j]), 4),
                }
                for j in order
            ]
        )
    return out


def _fallback_global(est, Xcodes, Xp, feat_names, top_n):
    from sklearn.inspection import permutation_importance

    yhat = (est.predict_proba(Xcodes)[:, 1] >= 0.5).astype(int)
    imp = permutation_importance(
        est, Xcodes, yhat, n_repeats=3, random_state=RANDOM_STATE
    ).importances_mean
    order = np.argsort(np.abs(imp))[::-1]
    if top_n is not None:
        order = order[:top_n]
    base = [{"feature": feat_names[j], "impact": round(float(imp[j]), 4)} for j in order]
    return [
        [{**d, "value": _clean(Xp.iloc[i][d["feature"]])} for d in base]
        for i in range(len(Xp))
    ]


def _clean(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else round(float(v), 4)
    if pd.isna(v):
        return None
    return str(v)
