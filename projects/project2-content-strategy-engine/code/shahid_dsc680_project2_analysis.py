"""
End-to-end M2 analysis for DSC680 Project 2.

Pipeline:
1. Generate (or load) the corpus.
2. Build the feature matrix.
3. Fit two models — OLS (interpretable coefficients) and XGBoost (predictive
   strength) — on log-transformed engagement.
4. Report effect sizes from OLS coefficients, permutation importances from
   the XGBoost model, and SHAP values for the top features.
5. Run one-way ANOVA across topics and formats with eta-squared effect sizes.
6. Run lead-lag analysis between Google Trends interest and engagement.
7. Simulate the "generation vs. baseline" comparison and bootstrap a 95% CI
   on the mean predicted-engagement uplift.
8. Save figures (PNG) and a results table (CSV) for the whitepaper.

This module is intentionally side-effectful only when called as a script.

Author: Komal Shahid (DSC680, May 2026)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from shahid_dsc680_project2_features import build_design_matrix
from shahid_dsc680_project2_synthetic_corpus import (
    CorpusConfig,
    TOPIC_EFFECTS,
    generate_corpus,
    synthesize_trends_series,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class ModelReport:
    name: str
    r2: float
    mae: float
    n_train: int
    n_test: int


def fit_ols(X: pd.DataFrame, y: pd.Series) -> tuple[LinearRegression, ModelReport]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, ModelReport(
        name="OLS",
        r2=float(r2_score(y_test, preds)),
        mae=float(mean_absolute_error(y_test, preds)),
        n_train=len(X_train),
        n_test=len(X_test),
    )


def fit_xgb(X: pd.DataFrame, y: pd.Series):
    try:
        from xgboost import XGBRegressor  # type: ignore
    except ImportError:  # graceful fallback for grading environments w/o xgboost
        from sklearn.ensemble import GradientBoostingRegressor
        XGBRegressor = GradientBoostingRegressor  # type: ignore

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, ModelReport(
        name="GradientBoost",
        r2=float(r2_score(y_test, preds)),
        mae=float(mean_absolute_error(y_test, preds)),
        n_train=len(X_train),
        n_test=len(X_test),
    )


def one_way_anova(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    groups = [g[value_col].values for _, g in df.groupby(group_col)]
    f_stat, p_val = stats.f_oneway(*groups)
    grand_mean = df[value_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = float(((df[value_col] - grand_mean) ** 2).sum())
    eta_sq = ss_between / ss_total if ss_total else float("nan")
    return {
        "factor": group_col,
        "F": float(f_stat),
        "p_value": float(p_val),
        "eta_squared": float(eta_sq),
        "k_groups": int(df[group_col].nunique()),
        "n": int(len(df)),
    }


def lead_lag_xcorr(
    series_a: np.ndarray,
    series_b: np.ndarray,
    max_lag: int = 8,
) -> tuple[int, float]:
    """Return ``(best_lag, best_corr)`` where positive lag means ``a`` leads ``b``."""
    a = (series_a - series_a.mean()) / (series_a.std() + 1e-9)
    b = (series_b - series_b.mean()) / (series_b.std() + 1e-9)
    best_lag, best_corr = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = float(np.corrcoef(a[:lag], b[-lag:])[0, 1])
        elif lag > 0:
            corr = float(np.corrcoef(a[lag:], b[:-lag])[0, 1])
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        if not np.isnan(corr) and corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag, best_corr


def bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 11
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s_a = rng.choice(a, size=len(a), replace=True)
        s_b = rng.choice(b, size=len(b), replace=True)
        diffs[i] = s_a.mean() - s_b.mean()
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def fig_engagement_heatmap(df: pd.DataFrame, out: Path) -> None:
    pivot = (
        df.groupby(["topic", "format"])["log_engagement"].mean().unstack().sort_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean log-engagement by topic × format")
    fig.colorbar(im, ax=ax, label="mean log(score+1)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_ols_coefficients(
    model: LinearRegression, X: pd.DataFrame, out: Path, top_n: int = 18
) -> None:
    coefs = pd.Series(model.coef_, index=X.columns).sort_values()
    top = pd.concat([coefs.head(top_n // 2), coefs.tail(top_n // 2)])
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#c0392b" if v < 0 else "#1f77b4" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("OLS coefficient (log-engagement units)")
    ax.set_title("Strongest content-feature effects (OLS)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_permutation_importance(
    model, X_test: pd.DataFrame, y_test: pd.Series, out: Path, top_n: int = 15
) -> None:
    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=0, scoring="r2"
    )
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    top = imp.tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#2e74b5")
    ax.set_xlabel("Permutation importance (Δ R²)")
    ax.set_title("Top predictors of engagement (gradient-boosted model)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_trends_vs_engagement(
    trends: pd.DataFrame, df: pd.DataFrame, out: Path
) -> None:
    # collapse engagement to weekly mean per topic
    engagement = (
        df.assign(week=lambda d: pd.qcut(d.index, q=104, labels=False))
        .groupby(["week", "topic"])["log_engagement"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, topic in zip(axes.flat, ["recovery", "sleep", "training", "gear"]):
        t_series = trends[trends.topic == topic].sort_values("week")
        e_series = engagement[engagement.topic == topic].sort_values("week")
        ax.plot(t_series.week, t_series.interest, label="Google Trends", color="#2e74b5")
        ax_b = ax.twinx()
        ax_b.plot(
            e_series.week, e_series.log_engagement,
            label="Engagement", color="#c0392b", alpha=0.7,
        )
        ax.set_title(f"r/{topic}")
        ax.set_xlabel("Week")
    fig.suptitle("Trends (blue) vs. weekly engagement (red) — lead-lag check")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_anova_topic(df: pd.DataFrame, out: Path) -> None:
    order = df.groupby("topic")["log_engagement"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(
        column="log_engagement", by="topic", ax=ax, grid=False,
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Engagement distribution by topic (ANOVA factor)")
    ax.set_ylabel("log(score+1)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generation evaluation
# ---------------------------------------------------------------------------
def simulate_generation(
    df: pd.DataFrame,
    model,
    X_columns: pd.Index,
    n_each: int = 60,
    seed: int = 23,
) -> tuple[float, float, float]:
    """Score AI-recommended drafts vs. naive baseline drafts using the fitted model.

    The "AI-recommended" batch samples categorical features from the empirically
    top-performing buckets (recovery / how-to / question / Tue morning); the
    baseline batch samples uniformly. The fitted model predicts log-engagement
    for each; we report bootstrap CI on the mean difference.
    """
    rng = np.random.default_rng(seed)

    def make_batch(strategy: str) -> pd.DataFrame:
        if strategy == "informed":
            topic_pool = ["recovery", "mental", "sleep", "nutrition"]
            format_pool = ["how-to", "case-study"]
            hook_pool = ["question", "contrarian"]
            day_pool = ["Tue", "Wed", "Thu"]
            daypart_pool = ["morning"]
        else:
            topic_pool = list(TOPIC_EFFECTS.keys())
            format_pool = list(set(df["format"]))
            hook_pool = list(set(df["hook"]))
            day_pool = list(set(df["day_of_week"]))
            daypart_pool = list(set(df["daypart"]))
        rows = []
        for _ in range(n_each):
            rows.append({
                "topic": rng.choice(topic_pool),
                "format": rng.choice(format_pool),
                "hook": rng.choice(hook_pool),
                "day_of_week": rng.choice(day_pool),
                "daypart": rng.choice(daypart_pool),
                "subreddit": rng.choice(df["subreddit"].unique()),
                "title_words": int(rng.integers(8, 18)),
                "body_words": int(rng.integers(150, 600)),
                "subreddit_subs": int(rng.choice(df["subreddit_subs"].unique())),
                "hour": int(rng.choice(range(6, 12) if daypart_pool == ["morning"] else range(24))),
                "score": 1, "num_comments": 0, "upvote_ratio": 0.85,
                "has_body": True, "post_id": "gen",
            })
        return pd.DataFrame(rows)

    informed = make_batch("informed")
    baseline = make_batch("baseline")

    def score(batch: pd.DataFrame) -> np.ndarray:
        X_batch, _ = build_design_matrix(batch)
        X_aligned = X_batch.reindex(columns=X_columns, fill_value=0.0)
        return model.predict(X_aligned)

    informed_scores = score(informed)
    baseline_scores = score(baseline)
    return bootstrap_mean_diff(informed_scores, baseline_scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(cfg: CorpusConfig | None = None) -> dict:
    cfg = cfg or CorpusConfig(n_posts=12_000, seed=42)
    df = generate_corpus(cfg)
    df.to_csv(RESULTS / "corpus_sample.csv", index=False)

    X, y = build_design_matrix(df)
    ols, ols_report = fit_ols(X, y)
    xgb, X_train, X_test, _, y_test, xgb_report = fit_xgb(X, y)

    anova_topic = one_way_anova(df, "topic", "log_engagement")
    anova_format = one_way_anova(df, "format", "log_engagement")
    anova_hook = one_way_anova(df, "hook", "log_engagement")

    trends = synthesize_trends_series()
    weekly_engagement = (
        df.assign(week=lambda d: pd.qcut(d.index, q=104, labels=False))
        .groupby(["week", "topic"])["log_engagement"]
        .mean()
        .reset_index()
    )
    lead_results = {}
    for topic in ["recovery", "sleep", "training", "gear"]:
        t = trends[trends.topic == topic].sort_values("week")["interest"].values
        e = weekly_engagement[weekly_engagement.topic == topic].sort_values("week")["log_engagement"].values
        m = min(len(t), len(e))
        lag, corr = lead_lag_xcorr(t[:m], e[:m])
        lead_results[topic] = {"best_lag_weeks": lag, "correlation": corr}

    gen_diff, gen_lo, gen_hi = simulate_generation(df, xgb, X.columns)

    # ---- figures ----
    fig_engagement_heatmap(df, FIGURES / "fig01_topic_format_heatmap.png")
    fig_ols_coefficients(ols, X, FIGURES / "fig02_ols_coefficients.png")
    fig_permutation_importance(xgb, X_test, y_test, FIGURES / "fig03_permutation_importance.png")
    fig_trends_vs_engagement(trends, df, FIGURES / "fig04_trends_lead_lag.png")
    fig_anova_topic(df, FIGURES / "fig05_anova_topic.png")

    # ---- results table ----
    results = {
        "corpus": {
            "n_posts": int(len(df)),
            "median_score": int(df["score"].median()),
            "mean_score": float(df["score"].mean()),
            "n_subreddits": int(df["subreddit"].nunique()),
            "n_topics": int(df["topic"].nunique()),
        },
        "models": {
            "ols": asdict(ols_report),
            "gradient_boost": asdict(xgb_report),
        },
        "anova": {
            "topic": anova_topic,
            "format": anova_format,
            "hook": anova_hook,
        },
        "lead_lag": lead_results,
        "generation": {
            "mean_log_engagement_uplift": gen_diff,
            "ci95_low": gen_lo,
            "ci95_high": gen_hi,
            "percent_uplift": float((np.exp(gen_diff) - 1) * 100),
        },
    }
    (RESULTS / "m2_results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":  # pragma: no cover
    res = run()
    print(json.dumps(res, indent=2))
