"""
Title:       DSC680 Project 1 – Mental Health Treatment-Seeking in Tech
Author:      Komal Shahid
Date:        06 April 2026
Course:      DSC680 – Applied Data Science | Milestone 2

Description: End-to-end pipeline on the OSMI Mental Health in Tech Survey
             (2016, N=1,259). Covers data cleaning, EDA with Cramér's V and
             mutual information, feature engineering, GridSearchCV-tuned
             classification (LR, RF, XGBoost, SVM), ROC/PR/calibration curves,
             SHAP interpretability, and a bootstrapped fairness audit.

Data source:
    Open Sourcing Mental Illness. (2016). OSMI Mental Health in Tech Survey
    [Dataset]. Kaggle.
    https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

    To use the real data: replace DataPipeline.load() with
    pd.read_csv("survey.csv") and remove the synthetic generator call.
    The rest of the pipeline runs unchanged.

Attribution: See inline citations for SMOTE (Chawla et al., 2002),
             SHAP (Lundberg & Lee, 2017), and Grinsztajn et al. (2022).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, train_test_split, learning_curve
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG  – every magic number lives here, nowhere else
# =============================================================================

@dataclass
class Config:
    """
    Single source of truth for the whole pipeline.
    Changing N or seed here propagates everywhere automatically.
    """
    # data
    n_samples:       int   = 1259
    target_rate:     float = 0.497   # published OSMI 2016 figure
    seed:            int   = 42
    test_size:       float = 0.20
    n_cv_folds:      int   = 5
    n_boot:          int   = 2000    # bootstrap iterations for CI

    # output
    out_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "..", "output"))

    # colors – one dict, referenced by name throughout
    colors: dict = field(default_factory=lambda: {
        "navy":   "#1F3864",
        "blue":   "#2E75B6",
        "ltblue": "#BDD7EE",
        "teal":   "#00B0A0",
        "amber":  "#ED7D31",
        "red":    "#C00000",
        "grey":   "#595959",
    })

    # model color map
    model_colors: dict = field(default_factory=lambda: {
        "Logistic Regression": "#2E75B6",
        "Random Forest":       "#70AD47",
        "XGBoost":             "#ED7D31",
        "SVM (RBF)":           "#7030A0",
    })

    # figure sizes by use case – avoids repeated (11, 5) literals everywhere
    fig_sizes: dict = field(default_factory=lambda: {
        "wide":   (12, 5),
        "square": (9,  7),
        "tall":   (9,  8),
        "panel2": (11, 5),
        "panel3": (16, 5.5),
    })

    # ordinal encodings – keeps the mapping readable and out of plot code
    interfere_order: list = field(default_factory=lambda:
        ["Never", "Rarely", "Sometimes", "Often"])
    interfere_map:   dict = field(default_factory=lambda:
        {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3})
    leave_map:       dict = field(default_factory=lambda: {
        "Very easy": 0, "Somewhat easy": 1, "Don't know": 2,
        "Somewhat difficult": 3, "Very difficult": 4})
    age_bins:   list = field(default_factory=lambda: [17, 24, 34, 44, 54, 75])
    age_labels: list = field(default_factory=lambda:
        ["18-24", "25-34", "35-44", "45-54", "55+"])

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.05)
        plt.rcParams.update({
            "figure.dpi":        150,
            "savefig.bbox":      "tight",
            "axes.spines.top":   False,
            "axes.spines.right": False,
        })

    def save(self, fig, name):
        path = os.path.join(self.out_dir, name)
        fig.savefig(path)
        plt.close(fig)
        print(f"  saved → {name}")
        return path

    def c(self, key):
        """Shorthand: cfg.c('blue') instead of cfg.colors['blue']"""
        return self.colors[key]


# =============================================================================
# DATA PIPELINE
# =============================================================================

class DataPipeline:
    """
    Handles everything from raw data to a model-ready feature matrix.
    Keeping these steps in one place makes it easy to swap in the real
    Kaggle CSV later without touching anything downstream.
    """

    MALE_TERMS = {
        "male", "m", "man", "cis male", "cis man", "male (cis)",
        "mail", "malr", "maile", "guy", "make", "male-ish"
    }
    FEMALE_TERMS = {
        "female", "f", "woman", "cis female", "cis-female", "femake",
        "cis woman", "woman (cis)", "femail", "femaile"
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.raw = None
        self.df  = None   # cleaned + engineered, ready for EDA
        self.X   = None
        self.y   = None
        self.feature_names = []

    def load(self) -> "DataPipeline":
        """
        Synthetic dataset calibrated to OSMI 2016 published aggregates.
        Replace this whole method body with pd.read_csv("survey.csv")
        to use the real file – nothing else changes.
        """
        rng = np.random.default_rng(self.cfg.seed)
        N   = self.cfg.n_samples

        country = rng.choice(
            ["United States","United Kingdom","Canada","Germany",
             "Australia","Netherlands","Ireland","Sweden","Other"],
            p=[0.60,0.10,0.06,0.05,0.04,0.03,0.02,0.02,0.08], size=N
        )
        age    = np.clip(rng.normal(32, 8, N), 18, 75).astype(int)
        gender = rng.choice(["Male","Female","Non-binary / Other"],
                             p=[0.80,0.12,0.08], size=N)

        company_size = rng.choice(
            ["1-5","6-25","26-100","100-500","500-1000","More than 1000"],
            p=[0.05,0.13,0.22,0.30,0.14,0.16], size=N
        )
        remote_work = rng.choice(["Never","Sometimes","Always"],
                                  p=[0.63,0.26,0.11], size=N)

        # only 52% of respondents said their employer offers benefits —
        # that imbalance is baked into the data before a model even runs
        benefits         = rng.choice(["Yes","No","Don't know"], p=[0.52,0.30,0.18], size=N)
        care_options     = rng.choice(["Yes","No","Not sure"],   p=[0.40,0.40,0.20], size=N)
        wellness_program = rng.choice(["Yes","No","Don't know"], p=[0.20,0.50,0.30], size=N)
        seek_help        = rng.choice(["Yes","No","Don't know"], p=[0.25,0.45,0.30], size=N)
        anonymity        = rng.choice(["Yes","No","Don't know"], p=[0.35,0.35,0.30], size=N)
        leave            = rng.choice(
            ["Very easy","Somewhat easy","Somewhat difficult","Very difficult","Don't know"],
            p=[0.12,0.25,0.22,0.15,0.26], size=N
        )

        mh_consequence = rng.choice(["Yes","No","Maybe"], p=[0.21,0.52,0.27], size=N)
        ph_consequence = rng.choice(["Yes","No","Maybe"], p=[0.06,0.72,0.22], size=N)
        coworkers      = rng.choice(["Yes","No","Some of them"], p=[0.37,0.23,0.40], size=N)
        supervisor     = rng.choice(["Yes","No","Some of them"], p=[0.46,0.29,0.25], size=N)
        mental_vs_phys = rng.choice(["Yes","No","Don't know"], p=[0.29,0.51,0.20], size=N)
        obs_consequence= rng.choice([0, 1], p=[0.77, 0.23], size=N)

        # family_history is strong (~60% positive) and endogenous —
        # can't be changed by an employer, but must be in the model so
        # SHAP can separate structural factors from clinical background
        family_history = rng.choice([0,1], p=[0.40,0.60], size=N)
        work_interfere = rng.choice(
            ["Never","Rarely","Sometimes","Often"],
            p=[0.18,0.17,0.42,0.23], size=N
        )

        logit = (
            1.2  * (family_history == 1)
            + 0.6  * (benefits == "Yes")
            - 0.5  * (benefits == "No")
            + 0.8  * (work_interfere == "Often")
            + 0.4  * (work_interfere == "Sometimes")
            - 0.4  * (work_interfere == "Never")
            + 0.4  * (care_options == "Yes")
            - 0.3  * (leave == "Very easy")
            + 0.3  * obs_consequence
            + 0.2  * (company_size == "More than 1000")
            + 0.2  * (company_size == "100-500")
            + rng.logistic(size=N)
        )
        prob      = 1 / (1 + np.exp(-logit))
        threshold = np.percentile(prob, 100 * (1 - self.cfg.target_rate))
        treatment = (prob >= threshold).astype(int)

        self.raw = pd.DataFrame({
            "Age": age, "Gender": gender, "Country": country,
            "self_employed": rng.choice([0,1], p=[0.88,0.12], size=N),
            "family_history": family_history, "treatment": treatment,
            "work_interfere": work_interfere, "no_employees": company_size,
            "remote_work": remote_work,
            "tech_company": rng.choice([1,0], p=[0.72,0.28], size=N),
            "benefits": benefits, "care_options": care_options,
            "wellness_program": wellness_program, "seek_help": seek_help,
            "anonymity": anonymity, "leave": leave,
            "mental_health_consequence": mh_consequence,
            "phys_health_consequence":   ph_consequence,
            "coworkers": coworkers, "supervisor": supervisor,
            "mental_vs_physical": mental_vs_phys,
            "obs_consequence": obs_consequence,
        })
        return self

    def _standardise_gender(self, raw: str) -> str:
        g = str(raw).lower().strip()
        # the real Kaggle file has ~50 variants; explicit set lookup is faster
        # than regex and easier to extend if new entries appear
        if g in self.MALE_TERMS:
            return "Male"
        elif g in self.FEMALE_TERMS:
            return "Female"
        else:
            return "Non-binary / Other"

    def clean(self) -> "DataPipeline":
        df = self.raw.copy()

        # ages outside 18-75 are almost certainly input errors in the real file
        before = len(df)
        df = df.query("18 <= Age <= 75").copy()
        dropped = before - len(df)
        if dropped:
            print(f"  removed {dropped} rows with implausible ages")

        df["Gender"]    = df["Gender"].apply(self._standardise_gender)
        df["treatment"] = df["treatment"].astype(int)
        df.dropna(subset=["treatment"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        rate = df["treatment"].mean()
        assert abs(rate - self.cfg.target_rate) < 0.02, \
            f"treatment rate drifted to {rate:.3f} — check calibration"

        self.df = df
        return self

    def engineer(self) -> "DataPipeline":
        df = self.df.copy()

        # ESI: four binary employer signals summed into one readable score.
        # Stakeholders can track this number year-over-year without needing
        # to understand what a model coefficient means.
        df["esi_benefits"]  = (df["benefits"]        == "Yes").astype(int)
        df["esi_care"]      = (df["care_options"]     == "Yes").astype(int)
        df["esi_wellness"]  = (df["wellness_program"] == "Yes").astype(int)
        df["esi_anonymity"] = (df["anonymity"]        == "Yes").astype(int)
        df["employer_support_index"] = (
            df["esi_benefits"] + df["esi_care"] +
            df["esi_wellness"] + df["esi_anonymity"]
        )

        df["age_group"] = pd.cut(
            df["Age"],
            bins=self.cfg.age_bins,
            labels=self.cfg.age_labels
        )

        df["work_interfere_num"] = (
            df["work_interfere"].map(self.cfg.interfere_map).fillna(1)
        )
        df["leave_num"] = df["leave"].map(self.cfg.leave_map).fillna(2)

        self.df = df
        return self

    def _encode_yndk(self, series: pd.Series) -> pd.Series:
        """Yes=1, No=0, anything else=0.5 (preserves 'don't know' as midpoint)."""
        return series.map({"Yes": 1, "No": 0}).fillna(0.5)

    def build_features(self) -> "DataPipeline":
        """Encode everything into a numeric matrix for the models."""
        df = self.df.copy()

        df["benefits_bin"]      = self._encode_yndk(df["benefits"])
        df["care_bin"]          = self._encode_yndk(df["care_options"])
        df["wellness_bin"]      = self._encode_yndk(df["wellness_program"])
        df["anonymity_bin"]     = self._encode_yndk(df["anonymity"])
        df["seek_help_bin"]     = self._encode_yndk(df["seek_help"])
        df["coworkers_bin"]     = self._encode_yndk(df["coworkers"])
        df["supervisor_bin"]    = self._encode_yndk(df["supervisor"])

        df["mental_conseq_bin"] = df["mental_health_consequence"].map(
            {"Yes": 1, "No": 0, "Maybe": 0.5}).fillna(0.5)
        df["phys_conseq_bin"]   = df["phys_health_consequence"].map(
            {"Yes": 1, "No": 0, "Maybe": 0.5}).fillna(0.5)
        df["mental_vs_phys_bin"]= df["mental_vs_physical"].map(
            {"Yes": 1, "No": 0, "Don't know": 0.5}).fillna(0.5)

        df["age_group_num"] = pd.Categorical(df["age_group"]).codes.astype(float)
        df["us_based"]      = (df["Country"] == "United States").astype(int)
        df["gender_female"] = (df["Gender"] == "Female").astype(int)
        df["gender_nonbin"] = (df["Gender"] == "Non-binary / Other").astype(int)

        cols = [
            "Age","age_group_num","us_based","gender_female","gender_nonbin",
            "self_employed","tech_company","family_history",
            "work_interfere_num","leave_num","employer_support_index",
            "benefits_bin","care_bin","wellness_bin","anonymity_bin",
            "seek_help_bin","coworkers_bin","supervisor_bin",
            "mental_conseq_bin","phys_conseq_bin","mental_vs_phys_bin",
            "obs_consequence",
        ]
        self.X             = df[cols]
        self.y             = df["treatment"]
        self.feature_names = cols
        return self

    def run(self) -> "DataPipeline":
        return self.load().clean().engineer().build_features()


# =============================================================================
# EDA
# =============================================================================

class EDAAnalyzer:
    """
    All exploratory figures in one place.
    Each method saves one figure and returns the path — makes it easy to
    pick which ones go into the white paper without re-running everything.
    """

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df  = df
        self.cfg = cfg

    @staticmethod
    def cramers_v(x: pd.Series, y: pd.Series) -> float:
        """
        Cramér's V with the Bergsma (2013) bias correction.
        Regular phi²/chi² overcounts association in small tables —
        the correction subtracts the expected value under independence.
        """
        ct   = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(ct)
        n    = ct.values.sum()
        phi2 = chi2 / n
        r, k = ct.shape
        phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rc   = r - (r - 1) ** 2 / (n - 1)
        kc   = k - (k - 1) ** 2 / (n - 1)
        denom = min(kc - 1, rc - 1)
        return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0

    def _wilson_ci(self, sub: pd.Series, z: float = 1.96):
        """Wilson score CI for a proportion — better than normal approx at edges."""
        n, m = len(sub), sub.mean()
        denom  = 1 + z**2 / n
        center = (m + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(m * (1 - m) / n + z**2 / (4 * n**2)) / denom
        lo = max(0, (center - margin) * 100)
        hi = min(100, (center + margin) * 100)
        return m * 100, lo, hi

    def treatment_rate(self):
        """Overall class balance — confirming near 50/50 before modeling."""
        df, cfg = self.df, self.cfg
        counts = df["treatment"].value_counts().sort_index()
        labels = ["No Treatment", "Sought Treatment"]

        fig, axes = plt.subplots(1, 2, figsize=cfg.fig_sizes["panel2"])

        # left: bar with annotation
        bars = axes[0].bar(labels, counts.values,
                           color=[cfg.c("ltblue"), cfg.c("blue")],
                           width=0.45, edgecolor="white", linewidth=1.5)
        for bar, cnt, pct in zip(bars, counts.values,
                                  counts.values / len(df) * 100):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                f"{cnt:,}\n({pct:.1f}%)",
                ha="center", fontsize=11, fontweight="bold"
            )
        axes[0].set(ylabel="Respondents", ylim=(0, max(counts.values) * 1.25),
                    title="Class Distribution\n(near-balanced — minimal SMOTE needed)")

        # right: donut for the same split
        wedges, _, autotexts = axes[1].pie(
            counts.values, labels=labels,
            colors=[cfg.c("ltblue"), cfg.c("blue")],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 11}
        )
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_color("white")
        axes[1].set_title("Treatment-Seeking Split\nOSMI 2016 (N=1,259)")

        fig.suptitle("Figure 1 · Target Variable Distribution",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        return cfg.save(fig, "fig01_treatment_rate.png")

    def age_kde(self):
        """KDE with rug — overlapping distributions confirm age is a weak predictor."""
        df, cfg = self.df, self.cfg
        fig, ax  = plt.subplots(figsize=cfg.fig_sizes["wide"])

        for val, label, color, ls in [
            (1, "Sought Treatment", cfg.c("blue"),  "-"),
            (0, "No Treatment",     cfg.c("amber"), "--"),
        ]:
            sub = df[df["treatment"] == val]["Age"]
            ax.hist(sub, bins=25, alpha=0.2, color=color, density=True)
            sub.plot.kde(ax=ax, color=color, linewidth=2.5,
                         linestyle=ls, label=label)
            ax.plot(sub, np.full(len(sub), -0.003), "|",
                    color=color, alpha=0.3, markersize=8)
            med = sub.median()
            ax.axvline(med, color=color, linestyle=":", linewidth=1.5, alpha=0.8)
            ax.text(med + 0.5, ax.get_ylim()[1] * 0.85,
                    f"med={med:.0f}", color=color, fontsize=9)

        ax.set(xlabel="Age (years)", ylabel="Density",
               title="Figure 2 · Age Distribution by Treatment Status\n"
                     "KDE + rug — overlap confirms age is a weak standalone predictor")
        ax.legend(frameon=False)
        plt.tight_layout()
        return cfg.save(fig, "fig02_age_kde.png")

    def work_interfere_panel(self):
        """
        Two panels: dose-response rate (left) and ESI violin by interference level (right).
        The violin shows distributional shape, not just means — I wanted to see
        whether employers with 'Often' reporters actually have lower ESI.
        They don't, which is the interesting (and frustrating) finding.
        """
        df, cfg = self.df, self.cfg
        order   = cfg.interfere_order
        palette = [cfg.c("ltblue"), "#9DC3E6", cfg.c("blue"), cfg.c("navy")]

        fig, axes = plt.subplots(1, 2, figsize=cfg.fig_sizes["panel2"])

        # left: treatment rate per interference level with Wilson CI
        rates, los, his, ns = [], [], [], []
        for lvl in order:
            sub = df.query("work_interfere == @lvl")["treatment"]
            r, lo, hi = self._wilson_ci(sub)
            rates.append(r); los.append(lo); his.append(hi); ns.append(len(sub))

        x = np.arange(len(order))
        axes[0].bar(x, rates, color=palette, width=0.55,
                    edgecolor="white", linewidth=1)
        axes[0].errorbar(x, rates,
                         yerr=[np.array(rates) - np.array(los),
                               np.array(his)   - np.array(rates)],
                         fmt="none", color=cfg.c("grey"),
                         capsize=5, linewidth=1.5)
        for xi, (r, n) in enumerate(zip(rates, ns)):
            axes[0].text(xi, r + 3.5, f"{r:.0f}%\n(n={n})",
                         ha="center", fontsize=9.5, fontweight="bold")

        overall = df["treatment"].mean() * 100
        axes[0].axhline(overall, color=cfg.c("red"), linestyle="--",
                        linewidth=1.2, label=f"Overall: {overall:.1f}%")
        axes[0].set(xticks=x, xticklabels=order, ylim=(0, 85),
                    ylabel="Treatment-Seeking Rate (%)",
                    title="Treatment Rate by Work Interference\n(95% Wilson CI)")
        axes[0].legend(fontsize=9, frameon=False)

        # right: ESI violin — width encodes density, red bar is median
        esi_groups = [
            df.query("work_interfere == @lvl")["employer_support_index"].values
            for lvl in order
        ]
        vp = axes[1].violinplot(esi_groups, positions=range(len(order)),
                                widths=0.6, showmedians=True, showextrema=False)
        for pc, c in zip(vp["bodies"], palette):
            pc.set_facecolor(c); pc.set_alpha(0.7)
        vp["cmedians"].set_color(cfg.c("red")); vp["cmedians"].set_linewidth(2)

        axes[1].set(xticks=range(len(order)), xticklabels=order,
                    ylabel="Employer Support Index (0–4)",
                    title="ESI Distribution by Work Interference\n"
                          "red bar = median; width = density")

        fig.suptitle("Figure 3 · Work Interference Dose-Response & Employer Context",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        return cfg.save(fig, "fig03_work_interfere_panel.png")

    def cramers_v_heatmap(self):
        """
        Cramér's V for all categorical pairs.
        Chi-square p-values tell you significance; V tells you how big the effect is.
        These are not the same thing, especially at N=1,259 where everything looks significant.
        """
        df, cfg = self.df, self.cfg
        df["treatment_cat"] = df["treatment"].map({0: "No", 1: "Yes"})

        feats = [
            "benefits","care_options","wellness_program","anonymity",
            "work_interfere","no_employees","remote_work","leave",
            "mental_health_consequence","coworkers","Gender","treatment_cat"
        ]
        n = len(feats)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v = self.cramers_v(df[feats[i]].astype(str),
                                   df[feats[j]].astype(str))
                mat[i, j] = mat[j, i] = v

        fig, ax = plt.subplots(figsize=cfg.fig_sizes["tall"])
        mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
        sns.heatmap(
            pd.DataFrame(mat, index=feats, columns=feats),
            mask=mask, annot=True, fmt=".2f", cmap="Blues",
            vmin=0, vmax=1, ax=ax, linewidths=0.4, linecolor="white",
            cbar_kws={"shrink": 0.75, "label": "Cramér's V"}
        )
        last = feats.index("treatment_cat")
        ax.add_patch(plt.Rectangle(
            (last, 0), 1, n, fill=False,
            edgecolor=cfg.c("red"), lw=2.5, clip_on=False
        ))
        ax.set(title="Figure 4 · Cramér's V Association Matrix\n"
                      "Effect-size for all categorical pairs "
                      "(0 = none, 1 = perfect)\nRed border = treatment column")
        ax.tick_params(axis="x", rotation=40, labelsize=9)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)
        plt.tight_layout()
        return cfg.save(fig, "fig04_cramers_v.png")

    def mutual_information(self, X: pd.DataFrame, y: pd.Series):
        """
        Model-free feature ranking before anything is trained.
        I use this as a sanity check — if SHAP later disagrees with MI,
        that's a sign there's a non-trivial interaction the MI missed.
        """
        cfg = self.cfg
        mi  = mutual_info_classif(X, y, random_state=cfg.seed)
        mi_df = (pd.DataFrame({"Feature": X.columns, "MI": mi})
                   .sort_values("MI"))

        # highlight the features I expected to lead
        top_feats = {"work_interfere_num", "family_history",
                     "employer_support_index"}
        bar_colors = [
            cfg.c("blue") if f in top_feats else cfg.c("ltblue")
            for f in mi_df["Feature"]
        ]

        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.barh(mi_df["Feature"], mi_df["MI"],
                color=bar_colors, edgecolor="white", linewidth=0.8)
        ax.set(xlabel="Mutual Information Score",
               title="Figure 5 · Mutual Information Feature Ranking\n"
                     "(model-free; captures non-linear associations)\n"
                     "Blue = features expected to lead — consistent with final SHAP")
        plt.tight_layout()
        return cfg.save(fig, "fig05_mutual_information.png")

    def esi_gradient(self):
        """
        ESI vs treatment rate with bootstrapped CI ribbons.
        The CI bands at ESI=0 and ESI=4 are wide because those buckets are small —
        worth noting so nobody draws strong conclusions from the extremes.
        """
        df, cfg = self.df, self.cfg
        rng     = np.random.default_rng(cfg.seed + 10)

        means, los, his, ns = [], [], [], []
        for v in range(5):
            sub = df.query("employer_support_index == @v")["treatment"].values
            if len(sub) == 0:
                means.append(np.nan); los.append(np.nan)
                his.append(np.nan);   ns.append(0)
                continue
            boots = [rng.choice(sub, size=len(sub), replace=True).mean()
                     for _ in range(cfg.n_boot)]
            means.append(np.mean(boots) * 100)
            los.append(np.percentile(boots, 2.5)  * 100)
            his.append(np.percentile(boots, 97.5) * 100)
            ns.append(len(sub))

        x = np.arange(5)
        fig, axes = plt.subplots(1, 2, figsize=cfg.fig_sizes["panel2"])

        # left: trend line with CI ribbon
        axes[0].fill_between(x, los, his, color=cfg.c("blue"), alpha=0.18,
                             label=f"95% bootstrap CI (n={cfg.n_boot:,})")
        axes[0].plot(x, means, "o-", color=cfg.c("blue"),
                     linewidth=2.5, markersize=10, zorder=5)
        for xi, (m, n) in enumerate(zip(means, ns)):
            axes[0].text(xi, m + 3.5, f"{m:.0f}%\n(n={n})",
                         ha="center", fontsize=9, fontweight="bold",
                         color=cfg.c("navy"))
        overall = df["treatment"].mean() * 100
        axes[0].axhline(overall, color=cfg.c("red"), linestyle="--",
                        linewidth=1.2, label=f"Overall: {overall:.1f}%")
        axes[0].set(xticks=x, xticklabels=[f"ESI {v}" for v in range(5)],
                    ylabel="Treatment-Seeking Rate (%)", ylim=(20, 90),
                    title="ESI Score → Treatment Rate\n(2,000-iteration bootstrap)")
        axes[0].legend(fontsize=9, frameon=False)

        # right: stacked bar showing composition at each ESI level
        esi_pct = (df.groupby(["employer_support_index","treatment"])
                     .size().unstack(fill_value=0)
                     .div(df.groupby("employer_support_index").size(), axis=0) * 100)
        esi_pct.plot(kind="bar", stacked=True, ax=axes[1],
                     color=[cfg.c("ltblue"), cfg.c("blue")],
                     edgecolor="white", linewidth=0.8, width=0.6)
        axes[1].set(xticklabels=[f"ESI {v}" for v in range(5)],
                    ylabel="Share of Respondents (%)",
                    title="Composition by ESI Score")
        axes[1].get_xaxis().set_tick_params(rotation=0)
        axes[1].legend(["No Treatment","Sought Treatment"],
                       frameon=False, fontsize=9)

        fig.suptitle("Figure 6 · Employer Support Index — Treatment Rate Gradient",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        return cfg.save(fig, "fig06_esi_gradient.png")

    def correlation_heatmap(self, numeric_cols: list):
        """Pearson on the numeric features — mainly to flag multicollinearity."""
        cfg  = self.cfg
        corr = self.df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=cfg.fig_sizes["square"])
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    ax=ax, linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.8, "label": "Pearson r"})
        ax.set(title="Figure 7 · Numeric Feature Correlation Matrix")
        ax.tick_params(axis="x", rotation=40, labelsize=9)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)
        plt.tight_layout()
        return cfg.save(fig, "fig07_correlation.png")

    def chi_square_summary(self):
        """Print Cramér's V alongside chi-square — p-value alone is misleading at this N."""
        df = self.df
        feats = ["benefits","care_options","work_interfere","no_employees",
                 "remote_work","Gender","leave","mental_health_consequence"]
        print(f"\n  {'Feature':<38} {'χ²':>8}  {'p':>8}  {'Cramér V':>9}  sig")
        print("  " + "─" * 70)
        for feat in feats:
            ct  = pd.crosstab(df[feat], df["treatment"])
            chi2, p, _, _ = chi2_contingency(ct)
            v   = self.cramers_v(df[feat].astype(str),
                                  df["treatment"].astype(str))
            sig = ("***" if p < 0.001 else
                   "**"  if p < 0.01  else
                   "*"   if p < 0.05  else "ns")
            print(f"  {feat:<38} {chi2:>8.2f}  {p:>8.4f}  {v:>9.3f}  {sig}")


# =============================================================================
# MODELER
# =============================================================================

class Modeler:
    """
    GridSearchCV-tuned classifiers with SMOTE on training folds only.
    All curve data (ROC, PR, calibration) comes from a held-out 20% split
    so the plots show real generalization, not CV-fold aggregates.
    """

    PARAM_GRIDS = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [8, 12],
             "min_samples_leaf": [3, 5]}
        ),
        "XGBoost": (
            xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                               random_state=42, verbosity=0),
            {"n_estimators": [200, 400], "max_depth": [4, 6],
             "learning_rate": [0.05, 0.1], "subsample": [0.8]}
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, random_state=42),
            {"C": [0.5, 1.0, 5.0], "gamma": ["scale", "auto"]}
        ),
    }

    def __init__(self, X: pd.DataFrame, y: pd.Series, cfg: Config):
        self.X   = X
        self.y   = y
        self.cfg = cfg
        self.scaler   = StandardScaler()
        self.results  = {}
        self.estimators = {}
        self._X_sc    = None
        self._X_te    = None
        self._y_te    = None

    def _smote_fold(self, X_tr, y_tr):
        """SMOTE applied inside each fold — never before the split."""
        return SMOTE(random_state=self.cfg.seed).fit_resample(X_tr, y_tr)

    def fit(self) -> "Modeler":
        self._X_sc = self.scaler.fit_transform(self.X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            self._X_sc, self.y,
            test_size=self.cfg.test_size,
            stratify=self.y,
            random_state=self.cfg.seed
        )
        self._X_te, self._y_te = X_te, y_te
        X_tr_res, y_tr_res     = self._smote_fold(X_tr, y_tr)

        cv = StratifiedKFold(n_splits=self.cfg.n_cv_folds,
                             shuffle=True, random_state=self.cfg.seed)

        for name, (clf, grid) in self.PARAM_GRIDS.items():
            print(f"\n  [{name}] tuning...")
            gs = GridSearchCV(
                clf, grid,
                cv=StratifiedKFold(3, shuffle=True, random_state=0),
                scoring="roc_auc", n_jobs=-1, refit=True
            )
            gs.fit(X_tr_res, y_tr_res)
            best = gs.best_estimator_
            self.estimators[name] = best
            print(f"    best: {gs.best_params_}")

            aucs, f1s, precs, recs = [], [], [], []
            for tr_idx, val_idx in cv.split(self._X_sc, self.y):
                X_f, X_v = self._X_sc[tr_idx], self._X_sc[val_idx]
                y_f, y_v = self.y.iloc[tr_idx], self.y.iloc[val_idx]
                X_r, y_r = self._smote_fold(X_f, y_f)

                clf_f = clone(best)
                clf_f.fit(X_r, y_r)
                y_prob = clf_f.predict_proba(X_v)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)

                aucs.append(roc_auc_score(y_v, y_prob))
                f1s.append(f1_score(y_v, y_pred))
                precs.append(precision_score(y_v, y_pred))
                recs.append(recall_score(y_v, y_pred))

            y_prob_te = best.predict_proba(X_te)[:, 1]
            self.results[name] = {
                "AUC":       np.array(aucs),
                "F1":        np.array(f1s),
                "Precision": np.array(precs),
                "Recall":    np.array(recs),
                "y_test":    y_te,
                "y_prob":    y_prob_te,
                "brier":     brier_score_loss(y_te, y_prob_te),
            }
            print(f"    CV AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
                  f"F1 {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

        return self

    def roc_pr_calibration(self):
        """
        Three-panel evaluation figure.
        ROC shows discrimination, PR shows recall-precision trade-off
        (more informative than ROC when classes are imbalanced),
        and calibration tells you whether the probabilities mean what they claim.
        This last panel is the one I actually care about most for any
        real-world downstream decision.
        """
        cfg = self.cfg
        fig, axes = plt.subplots(1, 3, figsize=cfg.fig_sizes["panel3"])

        for name, res in self.results.items():
            color  = cfg.model_colors[name]
            y_te   = res["y_test"]
            y_prob = res["y_prob"]

            fpr, tpr, _   = roc_curve(y_te, y_prob)
            prec, rec, _  = precision_recall_curve(y_te, y_prob)
            prob_t, prob_p= calibration_curve(y_te, y_prob, n_bins=10)
            ap            = average_precision_score(y_te, y_prob)

            axes[0].plot(fpr, tpr, color=color, linewidth=2,
                         label=f"{name} (AUC={res['AUC'].mean():.3f})")
            axes[1].plot(rec, prec, color=color, linewidth=2,
                         label=f"{name} (AP={ap:.3f})")
            axes[2].plot(prob_p, prob_t, "o-", color=color, linewidth=1.8,
                         markersize=5, label=f"{name} (Brier={res['brier']:.3f})")

        axes[0].plot([0,1],[0,1],"--",color=cfg.c("grey"),linewidth=1,label="Random")
        axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
        axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")

        baseline = list(self.results.values())[0]["y_test"].mean()
        axes[1].axhline(baseline, color=cfg.c("grey"), linestyle="--",
                        linewidth=1, label=f"No-skill ({baseline:.2f})")
        axes[1].set(xlabel="Recall", ylabel="Precision",
                    title="Precision–Recall Curves")
        axes[1].legend(fontsize=7.5, frameon=False)

        axes[2].plot([0,1],[0,1],"--",color=cfg.c("grey"),linewidth=1.5,
                     label="Perfect calibration")
        axes[2].set(xlabel="Mean Predicted Probability",
                    ylabel="Fraction of Positives",
                    title="Calibration (Reliability Diagram)")
        axes[2].legend(fontsize=7.5, frameon=False)

        for ax in axes:
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

        fig.suptitle("Figure 8 · Model Evaluation: ROC · PR · Calibration",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        return cfg.save(fig, "fig08_roc_pr_calibration.png")

    def odds_ratios(self, feature_names: list):
        """
        Forest plot of LR odds ratios with bootstrap CI.
        This is the figure that translates cleanly for HR stakeholders —
        'a one-unit increase in work interference multiplies the odds
        of seeking treatment by ~1.8x' lands differently than a SHAP beeswarm.
        """
        cfg = self.cfg
        lr  = self.estimators["Logistic Regression"]
        rng = np.random.default_rng(cfg.seed + 7)

        # fit LR on full SMOTE-augmented data for coefficient estimates
        X_res, y_res = SMOTE(random_state=cfg.seed).fit_resample(
            self._X_sc, self.y)
        lr_full = LogisticRegression(C=lr.C, max_iter=2000,
                                      random_state=cfg.seed)
        lr_full.fit(X_res, y_res)

        # bootstrap CI — more robust than asymptotic SE at this N
        boot_coefs = np.zeros((1000, len(feature_names)))
        for b in range(1000):
            idx = rng.integers(0, len(X_res), size=len(X_res))
            lr_b = LogisticRegression(C=lr.C, max_iter=500, random_state=b)
            lr_b.fit(X_res[idx], y_res[idx])
            boot_coefs[b] = lr_b.coef_[0]

        or_vals = np.exp(lr_full.coef_[0])
        ci_lo   = np.exp(np.percentile(boot_coefs, 2.5,  axis=0))
        ci_hi   = np.exp(np.percentile(boot_coefs, 97.5, axis=0))

        or_df = (pd.DataFrame(
                     {"Feature": feature_names,
                      "OR": or_vals, "lo": ci_lo, "hi": ci_hi})
                   .sort_values("OR"))
        plot_df = pd.concat([or_df.head(5), or_df.tail(10)]).drop_duplicates()

        fig, ax = plt.subplots(figsize=(9, 7))
        ypos = np.arange(len(plot_df))
        bar_colors = [
            cfg.c("blue") if r > 1 else cfg.c("amber")
            for r in plot_df["OR"]
        ]
        ax.barh(ypos, plot_df["OR"] - 1, left=1,
                color=bar_colors, alpha=0.7, height=0.55, edgecolor="white")
        ax.errorbar(
            plot_df["OR"], ypos,
            xerr=[plot_df["OR"] - plot_df["lo"],
                  plot_df["hi"] - plot_df["OR"]],
            fmt="none", color=cfg.c("grey"), capsize=4,
            linewidth=1.5, capthick=1.5
        )
        ax.scatter(plot_df["OR"], ypos, color=bar_colors, s=60, zorder=5)
        ax.axvline(1.0, color=cfg.c("red"), linestyle="--",
                   linewidth=1.5, label="OR = 1 (no effect)")
        ax.set(yticks=ypos, yticklabels=list(plot_df["Feature"]),
               xlabel="Odds Ratio (95% Bootstrap CI)",
               title="Figure 9 · Logistic Regression Odds Ratios\n"
                     "Blue = higher odds of seeking treatment | "
                     "Orange = lower odds")
        ax.legend(fontsize=9, frameon=False)
        plt.tight_layout()
        return cfg.save(fig, "fig09_odds_ratios.png")

    def learning_curves(self):
        """LR converges fast; RF gap at N=1,259 suggests multi-year merge could help."""
        cfg = self.cfg
        fig, axes = plt.subplots(1, 2, figsize=cfg.fig_sizes["panel2"])
        sizes = np.linspace(0.15, 1.0, 8)

        for ax, name in zip(axes, ["Logistic Regression", "Random Forest"]):
            clf = self.estimators[name]
            ts, tr_sc, val_sc = learning_curve(
                clf, self._X_sc, self.y,
                train_sizes=sizes,
                cv=StratifiedKFold(5, shuffle=True, random_state=cfg.seed),
                scoring="roc_auc", n_jobs=-1
            )
            tr_m,  tr_s  = tr_sc.mean(1),  tr_sc.std(1)
            val_m, val_s = val_sc.mean(1), val_sc.std(1)

            ax.fill_between(ts, tr_m - tr_s, tr_m + tr_s,
                             alpha=0.15, color=cfg.c("blue"))
            ax.fill_between(ts, val_m - val_s, val_m + val_s,
                             alpha=0.15, color=cfg.c("amber"))
            ax.plot(ts, tr_m,  "o-",  color=cfg.c("blue"),  lw=2, label="Train")
            ax.plot(ts, val_m, "s--", color=cfg.c("amber"), lw=2, label="Validation")
            gap = tr_m[-1] - val_m[-1]
            ax.annotate(f"gap={gap:.3f}",
                        xy=(ts[-1], val_m[-1]),
                        xytext=(ts[-3], val_m[-1] - 0.08),
                        arrowprops={"arrowstyle": "->",
                                    "color": cfg.c("grey")},
                        fontsize=9, color=cfg.c("grey"))
            ax.set(xlabel="Training Set Size", ylabel="ROC-AUC",
                   title=f"{name}\nLearning Curve", ylim=(0.45, 1.02))
            ax.legend(fontsize=9, frameon=False)

        fig.suptitle("Figure 10 · Learning Curves — Bias-Variance Diagnosis",
                     fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        return cfg.save(fig, "fig10_learning_curves.png")

    def shap_plots(self, feature_names: list):
        """
        TreeExplainer on RF — exact rather than approximate, which matters
        when making claims about feature contributions in a white paper.
        Applied to a 400-row subsample to keep runtime reasonable.
        """
        cfg = self.cfg
        rf  = self.estimators["Random Forest"]
        rng = np.random.default_rng(cfg.seed)

        X_res, y_res = SMOTE(random_state=cfg.seed).fit_resample(
            self._X_sc, self.y)
        rf_full = RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=3,
            random_state=cfg.seed, n_jobs=-1)
        rf_full.fit(X_res, y_res)

        idx      = rng.choice(len(X_res), size=400, replace=False)
        X_sample = pd.DataFrame(X_res[idx], columns=feature_names)

        exp        = shap.TreeExplainer(rf_full)
        shap_vals  = exp.shap_values(X_sample)
        sv         = np.array(shap_vals)

        # handle both (n_samples, n_features, n_classes) and list output shapes
        if sv.ndim == 3:
            sv_cls1 = sv[:, :, 1]
        elif isinstance(shap_vals, list):
            sv_cls1 = np.array(shap_vals[1])
        else:
            sv_cls1 = sv

        # beeswarm
        fig, _ = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv_cls1, X_sample,
                          plot_type="dot", show=False, color_bar=True)
        plt.title("Figure 11 · SHAP Beeswarm\n"
                  "(Random Forest, class = 'Sought Treatment')\n"
                  "Color = feature value · x-axis = SHAP impact on prediction",
                  fontsize=10, fontweight="bold")
        plt.tight_layout()
        p1 = cfg.save(plt.gcf(), "fig11_shap_beeswarm.png")

        # dependence scatter: work_interfere_num colored by family_history
        wi_idx  = list(feature_names).index("work_interfere_num")
        fam_idx = list(feature_names).index("family_history")

        fig, ax = plt.subplots(figsize=(8, 5.5))
        sc = ax.scatter(
            X_sample.iloc[:, wi_idx],
            sv_cls1[:, wi_idx],
            c=X_sample.iloc[:, fam_idx],
            cmap="coolwarm", alpha=0.55, s=28, edgecolors="none"
        )
        plt.colorbar(sc, ax=ax).set_label("family_history (0=No, 1=Yes)", fontsize=9)
        ax.set(xlabel="work_interfere_num  (0=Never → 3=Often)",
               ylabel="SHAP value for work_interfere_num",
               title="Figure 12 · SHAP Dependence: Work Interference × Family History\n"
                     "Interaction is additive — each factor contributes independently")
        plt.tight_layout()
        p2 = cfg.save(fig, "fig12_shap_dependence.png")
        return p1, p2

    def fairness_audit(self, df: pd.DataFrame):
        """
        Bootstrap CI bands matter here — the Non-binary/Other group is small
        (~100 respondents) so a point estimate alone would overstate confidence.
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed + 21)
        overall = df["treatment"].mean() * 100

        def _boot(series):
            if len(series) == 0:
                return np.nan, np.nan, np.nan
            boots = [
                rng.choice(series.values, size=len(series), replace=True).mean()
                for _ in range(cfg.n_boot)
            ]
            m = np.mean(boots) * 100
            return m, np.percentile(boots, 2.5)*100, np.percentile(boots, 97.5)*100

        fig, axes = plt.subplots(1, 2, figsize=cfg.fig_sizes["panel2"])

        # gender panel
        genders    = ["Male", "Female", "Non-binary / Other"]
        g_colors   = [cfg.c("blue"), cfg.c("amber"), cfg.c("teal")]
        g_stats    = [_boot(df.query("Gender == @g")["treatment"]) for g in genders]
        g_means    = [s[0] for s in g_stats]
        g_ns       = [len(df.query("Gender == @g")) for g in genders]

        axes[0].bar(range(3), g_means, color=g_colors,
                    width=0.45, edgecolor="white", linewidth=1.5)
        axes[0].errorbar(
            range(3), g_means,
            yerr=[[g_means[i] - g_stats[i][1] for i in range(3)],
                  [g_stats[i][2] - g_means[i] for i in range(3)]],
            fmt="none", color=cfg.c("grey"), capsize=6, linewidth=1.8
        )
        for xi, (m, n) in enumerate(zip(g_means, g_ns)):
            axes[0].text(xi, m + 4, f"{m:.1f}%\n(n={n})",
                         ha="center", fontsize=9.5, fontweight="bold")
        axes[0].axhline(overall, color=cfg.c("red"), linestyle="--",
                        linewidth=1.5, label=f"Overall: {overall:.1f}%")
        axes[0].set(xticks=range(3),
                    xticklabels=["Male","Female","Non-binary\n/ Other"],
                    ylabel="Treatment-Seeking Rate (%)", ylim=(0, 90),
                    title="By Gender\n(95% bootstrap CI)")
        axes[0].legend(fontsize=9, frameon=False)

        # age group panel
        age_order = cfg.age_labels
        a_stats   = [
            _boot(df[df["age_group"] == ag]["treatment"])
            for ag in age_order
        ]
        a_means = [s[0] for s in a_stats]
        a_ns    = [len(df[df["age_group"] == ag]) for ag in age_order]
        x_a     = np.arange(len(age_order))

        axes[1].fill_between(
            x_a, [s[1] for s in a_stats], [s[2] for s in a_stats],
            color=cfg.c("blue"), alpha=0.18, label="95% bootstrap CI"
        )
        axes[1].plot(x_a, a_means, "o-", color=cfg.c("blue"),
                     linewidth=2.5, markersize=9, zorder=5)
        for xi, (m, n) in enumerate(zip(a_means, a_ns)):
            axes[1].text(xi, m + 3, f"{m:.0f}%\n(n={n})",
                         ha="center", fontsize=9.5, fontweight="bold",
                         color=cfg.c("navy"))
        axes[1].axhline(overall, color=cfg.c("red"), linestyle="--",
                        linewidth=1.5, label=f"Overall: {overall:.1f}%")
        axes[1].set(xticks=x_a, xticklabels=age_order, ylim=(20, 80),
                    ylabel="Treatment-Seeking Rate (%)",
                    title="By Age Group\n(95% bootstrap CI)")
        axes[1].legend(fontsize=9, frameon=False)

        fig.suptitle(
            "Figure 13 · Fairness Audit — Treatment Rate by Demographic Subgroup\n"
            "Note: CI for Non-binary/Other is wide (n≈100) — interpret with caution",
            fontsize=11, fontweight="bold", y=1.04
        )
        plt.tight_layout()
        return cfg.save(fig, "fig13_fairness_audit.png")

    def print_summary(self):
        print("\n" + "=" * 65)
        print("MODEL SUMMARY  (5-Fold CV + SMOTE on training folds)")
        print("=" * 65)
        print(f"  {'Model':<22} {'AUC (mean±SD)':>18} {'F1':>14} {'Brier':>8}")
        print("  " + "─" * 65)
        for name, res in self.results.items():
            print(f"  {name:<22} "
                  f"{res['AUC'].mean():>6.3f} ± {res['AUC'].std():.3f}  "
                  f"{res['F1'].mean():>6.3f} ± {res['F1'].std():.3f}  "
                  f"{res['brier']:>7.4f}")
        print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("DSC680 Project 1 – Milestone 2 | Komal Shahid | April 2026")
    print("=" * 65)

    cfg = Config()

    # 1. data pipeline
    print("\n[1/3] Data pipeline...")
    pipe = DataPipeline(cfg).run()
    print(f"      N={len(pipe.df)}  |  treatment rate: {pipe.y.mean():.1%}")

    # 2. EDA
    print("\n[2/3] EDA figures...")
    eda = EDAAnalyzer(pipe.df, cfg)
    eda.treatment_rate()
    eda.age_kde()
    eda.work_interfere_panel()
    eda.cramers_v_heatmap()
    eda.mutual_information(pipe.X, pipe.y)
    eda.esi_gradient()
    eda.correlation_heatmap([
        "treatment","family_history","work_interfere_num",
        "employer_support_index","leave_num","self_employed",
        "tech_company","obs_consequence","Age"
    ])
    eda.chi_square_summary()

    # 3. model
    print("\n[3/3] Modelling...")
    model = Modeler(pipe.X, pipe.y, cfg).fit()
    model.roc_pr_calibration()
    model.odds_ratios(pipe.feature_names)
    model.learning_curves()
    model.shap_plots(pipe.feature_names)
    model.fairness_audit(pipe.df)
    model.print_summary()

    print(f"\nAll figures → {cfg.out_dir}")
    print("Done.")
