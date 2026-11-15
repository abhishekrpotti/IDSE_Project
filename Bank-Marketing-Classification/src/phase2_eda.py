"""Phase 2 descriptive EDA and preprocessing design; fits no models."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/bank-full.csv"
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"
EXPECTED_SHA256 = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
TARGET = "y"

def cramers_v(a, b):
    tab = pd.crosstab(a, b).to_numpy(dtype=float)
    n = tab.sum()
    if tab.size == 0 or n == 0 or min(tab.shape) < 2:
        return np.nan
    expected = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / n
    chi2 = np.sum((tab - expected) ** 2 / np.where(expected == 0, 1, expected))
    phi2 = chi2 / n
    r, k = tab.shape
    # Bias-corrected Cramer's V (Bergsma correction).
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan

def main():
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Raw data fingerprint changed: {digest}; expected Phase 1 {EXPECTED_SHA256}")
    df = pd.read_csv(RAW, sep=";", quotechar='"')
    if df.shape != (45211, 17):
        raise RuntimeError(f"Unexpected dimensions {df.shape}")
    if not set(df[TARGET].unique()) <= {"yes", "no"}:
        raise RuntimeError("Unexpected target labels")
    # Duration is deliberately omitted from the pre-call feature analysis.
    Xcols = NUMERIC + CATEGORICAL
    d = df.copy()
    for c in NUMERIC + ["duration"]: d[c] = pd.to_numeric(d[c], errors="raise")
    d[TARGET] = pd.Categorical(d[TARGET], categories=["no", "yes"])
    d["y_binary"] = (d[TARGET] == "yes").astype(int)

    # Feature distribution and relationship to target, using population SD for clarity.
    numeric_rows=[]
    for c in NUMERIC:
        s=d[c]; yes=d.loc[d[TARGET]=="yes",c]; no=d.loc[d[TARGET]=="no",c]
        numeric_rows.append({"feature":c,"n":len(s),"mean":s.mean(),"median":s.median(),"std_population":s.std(ddof=0),
          "skewness":s.skew(),"q1":s.quantile(.25),"q2_median":s.quantile(.5),"q3":s.quantile(.75),"iqr":s.quantile(.75)-s.quantile(.25),
          "min":s.min(),"max":s.max(),"mean_no":no.mean(),"mean_yes":yes.mean(),"median_no":no.median(),"median_yes":yes.median(),
          "mean_yes_minus_no":yes.mean()-no.mean(),"yes_rate_percent":100*d.groupby(c,observed=True)["y_binary"].mean().mean()})
    pd.DataFrame(numeric_rows).drop(columns="yes_rate_percent").to_csv(TABLES/"numerical_target_relationships.csv",index=False)

    # Category-by-target counts and within-category subscription percentages.
    cats=[]
    for c in CATEGORICAL:
        g=d.groupby([c,TARGET],observed=False).size().unstack(fill_value=0).reindex(columns=["no","yes"],fill_value=0)
        for val,row in g.iterrows():
            total=int(row.sum()); cats.append({"feature":c,"category":val,"count_no":int(row["no"]),"count_yes":int(row["yes"]),"total":total,
                "subscription_percent":100*row["yes"]/total if total else np.nan,
                "category_percent_of_dataset":100*total/len(d),"is_unknown_category":str(val)=="unknown",
                "unknown_category_percent_within_feature":100*(d[c].eq("unknown").sum()/len(d))})
    catdf=pd.DataFrame(cats)
    catdf.to_csv(TABLES/"subscription_by_category.csv",index=False)

    # Pearson matrix of pre-call numeric predictors; the pdays sentinel is retained as a separate flag
    # in future preprocessing, so raw Pearson correlations involving pdays are documented cautiously.
    corr=d[NUMERIC].corr(method="pearson")
    corr.to_csv(TABLES/"correlation_matrix.csv",float_format="%.6f")
    spearman=d[NUMERIC].corr(method="spearman")
    spearman.to_csv(TABLES/"rank_correlation_matrix.csv",float_format="%.6f")
    relation=[]
    for i,a in enumerate(CATEGORICAL):
        for b in CATEGORICAL[i+1:]: relation.append((a,b,cramers_v(d[a],d[b])))
    pd.DataFrame(relation,columns=["feature_a","feature_b","cramers_v_bias_corrected"]).to_csv(TABLES/"categorical_associations.csv",index=False)
    # Campaign history with target: include sentinel/zero category summary rather than treating pdays -1 as an ordinary time interval.
    hist=[]
    for c in ("campaign","pdays","previous"):
        vals=([-1] if c=="pdays" else [0] if c=="previous" else [])
        if c=="campaign": vals=sorted(d[c].unique())
        else: vals += sorted(v for v in d[c].unique() if v not in vals)
        for v in vals:
            sub=d[d[c]==v]
            hist.append((c,v,len(sub),int((sub[TARGET]=="yes").sum()),100*sub["y_binary"].mean()))
    pd.DataFrame(hist,columns=["feature","value","count","yes_count","subscription_percent"]).to_csv(TABLES/"campaign_history_subscription.csv",index=False)

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .22})
    # Six feature panels, each with a histogram and a side-by-side boxplot by outcome.
    fig,axes=plt.subplots(len(NUMERIC),2,figsize=(13,25))
    for i,c in enumerate(NUMERIC):
        axes[i,0].hist(d[c],bins=35,color="#3677a8",edgecolor="white")
        axes[i,0].set_title(f"{c}: distribution")
        axes[i,1].boxplot([d.loc[d[TARGET]=="no",c],d.loc[d[TARGET]=="yes",c]],tick_labels=["no","yes"],showfliers=False,patch_artist=True,
                          boxprops={"facecolor":"#80b5a2"},medianprops={"color":"#213a35"})
        axes[i,1].set_title(f"{c}: by subscription (fliers hidden for readability)")
    fig.suptitle("Pre-call numerical features: distributions and target groups",y=1.002,fontsize=16)
    fig.tight_layout(); fig.savefig(FIGURES/"numerical_distributions.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    fig,axes=plt.subplots(3,3,figsize=(17,14))
    for ax,c in zip(axes.flat,CATEGORICAL):
        p=catdf[catdf.feature==c].sort_values("subscription_percent")
        ax.barh(p["category"].astype(str),p["subscription_percent"],color="#3c8c75")
        ax.set_title(c); ax.set_xlabel("Subscribed (%)"); ax.set_ylabel("")
        ax.set_xlim(0,max(15,float(p.subscription_percent.max())*1.12))
    fig.suptitle("Subscription rates within categorical levels",y=1.005,fontsize=16)
    fig.tight_layout(); fig.savefig(FIGURES/"categorical_subscription_rates.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,7)); img=ax.imshow(corr.to_numpy(),cmap="coolwarm",vmin=-1,vmax=1)
    ax.set_xticks(range(len(NUMERIC)),NUMERIC,rotation=45,ha="right"); ax.set_yticks(range(len(NUMERIC)),NUMERIC)
    for i in range(len(NUMERIC)):
        for j in range(len(NUMERIC)): ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",fontsize=9)
    fig.colorbar(img,ax=ax,fraction=.046,pad=.04)
    ax.set_title("Pearson correlations: pre-call numerical features\n(pdays includes documented -1 sentinel)")
    fig.tight_layout(); fig.savefig(FIGURES/"correlation_heatmap.png",dpi=180,bbox_inches="tight"); plt.close(fig)

    counts=d[TARGET].value_counts(); imbalance=counts["no"]/counts["yes"]
    strongest=corr.where(np.triu(np.ones(corr.shape),1).astype(bool)).stack().abs().sort_values(ascending=False)
    strong_pairs=[f"{a}–{b} ({corr.loc[a,b]:.2f})" for a,b in strongest.index[:3]]
    cat_assoc=pd.DataFrame(relation,columns=["feature_a","feature_b","cramers_v_bias_corrected"])
    cat_top=cat_assoc.sort_values("cramers_v_bias_corrected",ascending=False).head(3)
    cat_pairs=[f"{r.feature_a}–{r.feature_b} (V={r.cramers_v_bias_corrected:.2f})" for r in cat_top.itertuples()]
    unknown_rows=[]
    for c in CATEGORICAL:
        nunk=int(d[c].eq("unknown").sum())
        if nunk:
            part=d[d[c].eq("unknown")]
            unknown_rows.append(f"`{c}` {nunk:,} ({100*nunk/len(d):.2f}% of records; subscription {100*part.y_binary.mean():.2f}%)")
    report=f'''# Phase 2 — Exploratory data analysis and preprocessing design

## Validation and scope

Loaded `data/raw/bank-full.csv` without changing it. Dimensions are **{len(d):,} × {len(df.columns)}**. SHA-256 is `{digest}`, matching Phase 1 exactly. Target is `y`; the primary pre-call predictor list below excludes `duration`.

## Numerical EDA

The six pre-call numeric candidates are age, balance, day, campaign, pdays and previous. Descriptive statistics, skewness, quartiles, IQR and yes/no means and medians are in `tables/numerical_target_relationships.csv`. Histograms and outcome-group boxplots are in `figures/numerical_distributions.png` (boxplot fliers hidden for readability; no observations removed).

Balance is strongly right-skewed (mean {d.balance.mean():.1f}, median {d.balance.median():.1f}; skew {d.balance.skew():.2f}) with a long range ({d.balance.min():,} to {d.balance.max():,}). Campaign is right-skewed (mean {d.campaign.mean():.2f}, median {d.campaign.median():.0f}; skew {d.campaign.skew():.2f}). Previous is zero-heavy (zero in {(d.previous.eq(0).mean()*100):.2f}%); pdays is sentinel-heavy, so its raw mean/skew and Pearson correlation need cautious interpretation. Candidate transformations for later comparison: `log1p` for nonnegative campaign/previous and possibly positive balance after separating negative balances; robust scaling or winsorization sensitivity checks for balance/campaign. Do not transform automatically; preserve raw values and validate choices inside training folds.

## Categorical EDA and unknowns

`tables/subscription_by_category.csv` gives per-feature category counts, yes/no counts, subscription percentage, dataset share, and unknown proportions. `figures/categorical_subscription_rates.png` plots the within-category yes rate for all nine requested categorical variables.

The `unknown` category shares and subscription rates are: {'; '.join(unknown_rows)}. Rates are descriptive and may be noisy for small groups. The prior campaign `success` level has a 64.73% observed subscription rate (1,511 records), while `poutcome=unknown` has 9.16%; this suggests missing-history status itself may carry useful information. Several months and student/retired job levels also show higher observed rates, but these are associations and some groups are smaller. Keeping `unknown` explicitly preserves a potentially informative pattern; later compare this with treating unknown as missing and imputing, using training-only validation. Never infer the unknown semantics as a true customer state.

## Feature relationships and campaign history

`tables/correlation_matrix.csv` and its heatmap report Pearson correlations among numeric pre-call candidates; a Spearman matrix is provided as a rank-based companion. The strongest absolute Pearson pairs are {', '.join(strong_pairs)}. The pdays–previous correlation is partly structural because `pdays=-1` and `previous=0` describe no previous contact; interpret numeric correlation after representing this special case. Other numeric correlations are modest. Strong categorical associations by bias-corrected Cramer's V are {', '.join(cat_pairs)}; contact/month may reflect campaign scheduling and job/education overlap. These are descriptive associations, not causation. `tables/campaign_history_subscription.csv` reports campaign-history counts and observed subscription rates by values.

`campaign` is the number of contacts in the current campaign including the last contact, so it may not be fully known at the before-call decision time. `pdays` and `previous` summarize prior contact history and can be relevant to response patterns. They are related campaign-history fields, but not interchangeable: `pdays` is a recency indicator and `previous` is a count. Evaluate redundancy and timing explicitly.

## Special values

The supplied `bank-names.txt` defines `pdays=-1` as “client was not previously contacted.” It is a sentinel, not negative elapsed time. There are 36,954 `pdays=-1` records, with 9.16% subscription; `previous=0` occurs in the same 36,954 records and has the same observed rate in this file. Prefer a binary `previously_contacted = (pdays != -1)` plus a recency value defined only for contacted records; represent the never-contacted group separately within a pipeline. `previous=0` means no contacts before the current campaign, consistent with the zero-heavy count distribution. `unknown` labels are documented categorical levels; they are not actual null cells in the source. Keep them distinct initially.

## Preprocessing recommendation (design only)

- Primary features: age, job, marital, education, default, balance, housing, loan, contact, day, month, campaign, pdays, previous, poutcome; target `y`.
- No raw nulls were found in Phase 1; if missing values appear in future scoring data, use median imputation for numeric and most-frequent or explicit missing category for categorical, fitted on training folds only.
- OneHotEncoder with `handle_unknown='ignore'` for categorical predictors. Compare retaining documented `unknown` as a category versus converting it to missing and imputing, within training-only cross-validation.
- Scale numeric variables only for models sensitive to scale (e.g. linear or distance-based); fit scaler in a `ColumnTransformer`/sklearn `Pipeline`. Standard scaling is baseline; consider robust scaling after inspecting outlier sensitivity.
- Retain outliers initially. Investigate robust alternatives only if they materially affect a selected model's validation performance.
- Class ratio is {imbalance:.2f}:1 (no:yes). Evaluate class weights before any resampling; if resampling is later justified, apply only within training folds.
- Place every learned imputer, encoder, scaler, and any resampling step inside sklearn pipelines to prevent leakage.

## Timing and future evaluation design

Prediction scenario: estimate subscription **before a marketing call**. Exclude `duration` because the current-call duration is only observed after the call and is an outcome proxy/leakage risk. Treat `campaign` as uncertain because its definition includes the last contact; confirm whether the current-call count is known at scoring. Day/month/contact type and prior-history fields also require operational confirmation of what is known when the contact list is scored.

Rows are documented as date-ordered but no timestamp or client ID is present in this CSV. When model evaluation is later approved, prefer a chronological holdout if an ordering key can be recovered from source metadata; do not create a final test set in this phase. The current EDA fits no model and tunes no parameters.

## Suggested modeling strategy (for a later approved phase)

Establish a simple majority-class and regularized logistic-regression benchmark with a stratified, training-only cross-validation design if the deployment setting is not temporal; prefer time-based evaluation when dates are available. Compare a tree-based model after the leakage-safe preprocessing is established. Emphasize recall/precision, PR-AUC, balanced accuracy and calibration alongside ROC-AUC, and report threshold trade-offs. Use class weights before considering resampling. Do not include duration in the pre-call scenario.
'''
    (ROOT/"reports/phase2_eda.md").write_text(report,encoding="utf-8")
    dictionary='''# Data dictionary — UCI Bank Marketing (`bank-full.csv`)

Target: `y` (yes/no term-deposit subscription). Primary scenario: predict before a marketing call. Exclude `duration` for this scenario.

| Column | Meaning | Type / notes | Timing note |
|---|---|---|---|
| age | Client age | Integer | Candidate pre-call |
| job | Job category | Categorical; retain `unknown` as a documented value | Candidate pre-call |
| marital | Marital status; divorced includes divorced or widowed per source | Categorical | Candidate pre-call |
| education | Education level | Categorical; retain `unknown` | Candidate pre-call |
| default | Credit in default? | yes/no | Candidate pre-call |
| balance | Average yearly balance in euros | Integer; negative and large positive values occur | Candidate pre-call |
| housing | Housing loan? | yes/no | Candidate pre-call |
| loan | Personal loan? | yes/no | Candidate pre-call |
| contact | Communication type | Categorical; retain `unknown` | Confirm channel is known at list scoring |
| day | Day of month of last contact | Integer, 1–31 | Confirm availability when scoring |
| month | Month of last contact | Categorical | Confirm availability when scoring |
| duration | Duration of current call in seconds | Integer | Post-call only; leakage for pre-call prediction; exclude |
| campaign | Contacts performed during current campaign, including last contact | Integer | Potentially unavailable before call; confirm |
| pdays | Days since last contact in a previous campaign | Integer; **-1 means not previously contacted**, not negative elapsed time | Prior campaign history |
| previous | Contacts before current campaign | Integer; 0 means none | Prior campaign history |
| poutcome | Previous campaign outcome | Categorical; `unknown` retained | Prior campaign history |
| y | Subscribed a term deposit? | Binary target, yes/no | Outcome |

Source documentation: `data/raw/bank-names.txt`; UCI Bank Marketing, DOI 10.24432/C5K306. The source reports no missing attribute values; strings such as `unknown` are categorical values, not blank cells.
'''
    (ROOT/"docs/DATA_DICTIONARY.md").write_text(dictionary,encoding="utf-8")
    print(f"Verified {len(d):,}x{len(df.columns)}; SHA256={digest}; target yes={counts['yes']:,} ({100*counts['yes']/len(d):.2f}%), no={counts['no']:,} ({100*counts['no']/len(d):.2f}%). Strongest Pearson pairs: {', '.join(strong_pairs)}")

if __name__ == "__main__": main()
