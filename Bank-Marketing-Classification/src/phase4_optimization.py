"""Phase 4: modest RF search, validation threshold/model choice, one final test evaluation."""
from pathlib import Path
import hashlib, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data/raw/bank-full.csv"
TABLES=ROOT/"reports/tables"; FIGURES=ROOT/"reports/figures"
ARTIFACTS=ROOT/"models/candidates"; FINAL=ROOT/"models/final"
EXPECTED_SHA256="d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"
SEED=42
FEATURES=["age","balance","day","campaign","pdays","previous","job","marital","education","default","housing","loan","contact","month","poutcome"]
NUMERIC=["age","balance","day","campaign","pdays","previous"]
CATEGORICAL=[c for c in FEATURES if c not in NUMERIC]
THRESHOLDS=sorted(set(np.round(np.arange(.10,.901,.05),2).tolist()+[.50]))

def metric_set(y,prob,threshold):
    pred=(prob>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"threshold":float(threshold),"precision":precision_score(y,pred,zero_division=0),
      "recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),
      "accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),
      "roc_auc":roc_auc_score(y,prob),"average_precision":average_precision_score(y,prob),
      "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

def threshold_table(name,y,prob):
    rows=[{"candidate":name,**metric_set(y,prob,t)} for t in THRESHOLDS]
    return rows

def main():
    TABLES.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True); FINAL.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA256: raise RuntimeError(f"Raw SHA-256 mismatch: {digest}")
    df=pd.read_csv(RAW,sep=";",quotechar='"')
    if df.shape!=(45211,17): raise RuntimeError(f"Unexpected dimensions {df.shape}")
    cfg3=json.loads((ARTIFACTS/"experiment_config.json").read_text(encoding="utf-8"))
    if cfg3["raw_sha256"]!=digest or cfg3["seed"]!=SEED or cfg3["features"]!=FEATURES or "duration" not in cfg3["excluded_features"]:
        raise RuntimeError("Phase 3 experiment configuration is inconsistent")
    # Enforce saved Phase 3 partition and exact coverage; do not derive a new split.
    index_map=pd.read_csv(TABLES/"split_indices.csv")
    if len(index_map)!=len(df) or not np.array_equal(index_map.row_index.to_numpy(),np.arange(len(df))):
        raise RuntimeError("Saved split indices do not align with raw row order")
    split=index_map.split.to_numpy()
    if set(split)!={"train","validation","test"}: raise RuntimeError("Saved split labels invalid")
    y=(df.y=="yes").astype("int8")
    X=df[FEATURES].copy()
    if "duration" in X.columns or len(X.columns)!=15: raise RuntimeError("Duration exclusion failed")
    it=np.where(split=="train")[0]; iv=np.where(split=="validation")[0]; ite=np.where(split=="test")[0]
    Xtr,ytr=X.iloc[it],y.iloc[it]; Xv,yv=X.iloc[iv],y.iloc[iv]

    original=joblib.load(ARTIFACTS/"random_forest_none.joblib")
    original.set_params(classifier__n_jobs=4)
    if "duration" in original.feature_names_in_: raise RuntimeError("Original model unexpectedly includes duration")
    # Stage 1: threshold curves for original baseline on validation only.
    original_prob=original.predict_proba(Xv)[:,list(original.classes_).index(1)]
    threshold_rows=threshold_table("original_random_forest",yv,original_prob)

    # Stage 2: 20 random parameter combinations, 5-fold stratified CV, training partition only.
    base=clone(original).set_params(classifier__n_jobs=4)
    search_space={"classifier__n_estimators":[80,120,180],
      "classifier__max_depth":[None,10,18],"classifier__min_samples_split":[2,5,10,20],
      "classifier__min_samples_leaf":[1,2,4,8],"classifier__max_features":["sqrt",.5],
      "classifier__class_weight":[None,"balanced","balanced_subsample"]}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    search=RandomizedSearchCV(base,search_space,n_iter=20,scoring="average_precision",cv=cv,refit=True,
       random_state=SEED,n_jobs=1,verbose=1,return_train_score=False,error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); search.fit(Xtr,ytr)
    tuned=search.best_estimator_
    tuned_prob=tuned.predict_proba(Xv)[:,list(tuned.classes_).index(1)]
    threshold_rows.extend(threshold_table("tuned_random_forest",yv,tuned_prob))
    pd.DataFrame(threshold_rows).to_csv(TABLES/"threshold_analysis.csv",index=False)

    # Candidate metrics at default .50, then validation AP selection. This occurs before any test read.
    phase3_cv=pd.read_csv(TABLES/"cv_results.csv")
    orig_cv=phase3_cv.loc[phase3_cv.candidate=="random_forest_none"].iloc[0]
    orig_val=metric_set(yv,original_prob,.5)
    tuned_val=metric_set(yv,tuned_prob,.5)
    model_rows=[{"candidate":"original_random_forest","training_cv_average_precision_mean":orig_cv.average_precision_mean,
       "training_cv_average_precision_std":orig_cv.average_precision_std,"search_best_params":"Phase 3 baseline parameters",**orig_val},
       {"candidate":"tuned_random_forest","training_cv_average_precision_mean":search.best_score_,
       "training_cv_average_precision_std":np.nan,"search_best_params":json.dumps(search.best_params_,sort_keys=True),**tuned_val}]
    # Compare validation AP, prespecified primary selection metric; all remaining metrics are reported alongside.
    selected_name="tuned_random_forest" if tuned_val["average_precision"]>orig_val["average_precision"] else "original_random_forest"
    selected_train_pipe=tuned if selected_name=="tuned_random_forest" else original
    selected_val_prob=tuned_prob if selected_name=="tuned_random_forest" else original_prob
    selected_grid=pd.DataFrame([r for r in threshold_rows if r["candidate"]==selected_name])
    maxf1=selected_grid.loc[selected_grid.f1.idxmax()]
    # Practical alternatives defined by constraints: high precision with recall >= .20; high recall with precision >= .20.
    pset=selected_grid[selected_grid.recall>=.20]
    ppoint=pset.sort_values(["precision","recall"],ascending=False).iloc[0] if not pset.empty else selected_grid.sort_values("precision",ascending=False).iloc[0]
    rset=selected_grid[selected_grid.precision>=.20]
    rpoint=rset.sort_values(["recall","precision"],ascending=False).iloc[0] if not rset.empty else selected_grid.sort_values("recall",ascending=False).iloc[0]
    selected_threshold=float(maxf1.threshold)
    pd.DataFrame(model_rows).to_csv(TABLES/"tuned_model_results.csv",index=False)

    # Threshold plots across both original and tuned candidate curves.
    thdf=pd.DataFrame(threshold_rows)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for candidate,g in thdf.groupby("candidate"):
        axes[0].plot(g.threshold,g.precision,marker="o",label=f"{candidate}: precision")
        axes[0].plot(g.threshold,g.recall,marker="s",linestyle="--",label=f"{candidate}: recall")
        axes[1].plot(g.threshold,g.f1,marker="o",label=candidate)
    axes[0].axvline(selected_threshold,color="#333",linestyle=":",label=f"selected {selected_threshold:.2f}")
    axes[0].set(title="Precision and recall vs threshold",xlabel="Threshold",ylabel="Score",ylim=(0,1)); axes[0].legend(fontsize=8)
    axes[1].axvline(selected_threshold,color="#333",linestyle=":"); axes[1].set(title="F1 vs threshold",xlabel="Threshold",ylabel="F1",ylim=(0,1)); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES/"threshold_analysis.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    # Validation permutation importance, on original feature columns and Average Precision scoring.
    perm=permutation_importance(selected_train_pipe,Xv,yv,scoring="average_precision",n_repeats=5,random_state=SEED,n_jobs=1)
    imp=pd.DataFrame({"feature":FEATURES,"importance_mean":perm.importances_mean,"importance_std":perm.importances_std}).sort_values("importance_mean",ascending=False)
    imp.to_csv(TABLES/"feature_importance.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,7)); show=imp.sort_values("importance_mean")
    ax.barh(show.feature,show.importance_mean,xerr=show.importance_std,color="#3c7a65",capsize=2)
    ax.axvline(0,color="#333",linewidth=.8); ax.set_xlabel("Decrease in validation Average Precision after permutation"); ax.set_title("Validation permutation importance (mean ± SD, 5 repeats)")
    fig.tight_layout(); fig.savefig(FIGURES/"feature_importance.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    # Final fit on train+validation only after model/threshold decision; test remains untouched until below.
    fit_idx=np.concatenate([it,iv]); Xfit=X.iloc[fit_idx]; yfit=y.iloc[fit_idx]
    final_pipe=clone(selected_train_pipe)
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); final_pipe.fit(Xfit,yfit)
    if "duration" in getattr(final_pipe,"feature_names_in_",[]): raise RuntimeError("Final pipeline includes duration")

    # The sole test evaluation call: one predict_proba and one threshold application after selection is locked.
    Xt=X.iloc[ite]; yt=y.iloc[ite]
    test_prob=final_pipe.predict_proba(Xt)[:,list(final_pipe.classes_).index(1)]
    test=metric_set(yt,test_prob,selected_threshold)
    test["candidate"]=selected_name; test["threshold"]=selected_threshold
    test["n_test"]=len(yt); test["positive_class"]="yes"
    pd.DataFrame([test]).to_csv(TABLES/"final_test_metrics.csv",index=False)
    test_pred=(test_prob>=selected_threshold).astype(int)
    cm=confusion_matrix(yt,test_pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm,cmap="Blues")
    ax.set_xticks([0,1],["no","yes"]); ax.set_yticks([0,1],["no","yes"]); ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"Final test confusion matrix — threshold {selected_threshold:.2f}")
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,f"{v:,}",ha="center",va="center",color="white" if v>cm.max()*.5 else "#123")
    fig.tight_layout(); fig.savefig(FIGURES/"final_confusion_matrix.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    class_report=classification_report(yt,test_pred,labels=[0,1],target_names=["no","yes"],digits=4,zero_division=0)

    # Save final pipeline and decision/config metadata.
    joblib.dump(final_pipe,FINAL/"final_pipeline.joblib",compress=3)
    val_selected=metric_set(yv,selected_val_prob,selected_threshold)
    config={"raw_sha256":digest,"seed":SEED,"split":"existing seed-42 stratified 70/15/15 indices",
      "features":FEATURES,"excluded_features":["duration"],"target":"y","positive_class":"yes",
      "selected_model":selected_name,"selected_threshold":selected_threshold,"threshold_selection":"maximum validation F1",
      "selection_metric":"validation Average Precision; threshold by validation F1",
      "final_fit_rows":"training + validation","test_evaluation_count":1,
      "tuning":{"method":"RandomizedSearchCV","n_iter":20,"cv":"StratifiedKFold(5, shuffle=True, random_state=42)",
                 "scoring":"average_precision","best_params":search.best_params_},
      "validation_metrics_at_selected_threshold":val_selected,"test_metrics":test}
    (FINAL/"final_config.json").write_text(json.dumps(config,indent=2,default=lambda x:float(x)),encoding="utf-8")

    def fmt(r): return ", ".join(f"{k}={r[k]:.4f}" for k in ("average_precision","roc_auc","precision","recall","f1","balanced_accuracy","accuracy"))
    roles={r.feature:r.importance_mean for r in imp.itertuples()}
    top_imp=imp.iloc[:6]
    imp_lines="\n".join(f"- `{r.feature}`: {r.importance_mean:.5f} ± {r.importance_std:.5f}" for r in top_imp.itertuples())
    op_lines=f"Maximum F1: threshold {maxf1.threshold:.2f} (P={maxf1.precision:.3f}, R={maxf1.recall:.3f}, F1={maxf1.f1:.3f}); higher-precision option (recall ≥0.20): threshold {ppoint.threshold:.2f} (P={ppoint.precision:.3f}, R={ppoint.recall:.3f}); higher-recall option (precision ≥0.20): threshold {rpoint.threshold:.2f} (P={rpoint.precision:.3f}, R={rpoint.recall:.3f})."
    opt_report=f'''# Phase 4 — Model optimisation and threshold analysis

## Integrity and protocol

Raw SHA-256 `{digest}` matches Phase 1. The data has 45,211 rows × 17 columns. Raw data was preserved and `duration` is absent from every feature matrix and pipeline. The existing saved seed-42 split indices were reused: 31,647 train, 6,782 validation, 6,782 test. Positive class is `yes` (1). All preprocessing remains inside the original sklearn pipeline. Hyperparameter search used training only; model/threshold choice and permutation importance used validation. No test rows were transformed, predicted, or scored until decisions and final refitting were complete; the test split was evaluated once.

## Threshold trade-off

Thresholds 0.10–0.90 by 0.05 (including 0.50) were evaluated on validation for both the Phase 3 Random Forest and tuned Random Forest. Confusion counts and precision, recall, F1, accuracy, balanced accuracy, ROC-AUC and Average Precision are in `tables/threshold_analysis.csv`; plots are in `figures/threshold_analysis.png`.

For the selected candidate, {op_lines} Lowering a threshold generally predicts more positives, increasing recall while often reducing precision; raising it generally makes positive predictions more selective, improving precision at the cost of missed subscribers. No threshold is universally best; these are validation operating points, and the business cost of false positives versus missed subscribers should guide deployment.

## Targeted Random Forest search

RandomizedSearchCV evaluated 20 configurations, five-fold shuffled stratified CV (seed {SEED}), scoring Average Precision, on training rows only. Search space covered n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features and class_weight. Best parameters: `{json.dumps(search.best_params_,sort_keys=True)}`. Full comparison against the existing Phase 3 baseline is in `tables/tuned_model_results.csv`.

## Validation model selection

Validation at threshold 0.50 — original Random Forest: {fmt(orig_val)}. Tuned Random Forest: {fmt(tuned_val)}. Selected by validation Average Precision: **{selected_name}**. Its selected threshold is **{selected_threshold:.2f}**, the validation maximum-F1 point. At that threshold: {fmt(val_selected)}. This choice balances precision and recall according to F1 while retaining AP as the model-selection metric. Other operating points are available above; no test metric informed these decisions.

## Interpretation and limits

Validation permutation importance ranks original feature columns by AP decrease after shuffling:

{imp_lines}

The requested feature roles: `poutcome` carries strong prior campaign outcome information, especially a prior success; `contact` may reflect outreach channel and its availability must match scoring time; `campaign` counts current-campaign attempts including the last contact and remains a timing risk; `pdays=-1` is a never-contacted sentinel, not a negative duration; `previous` counts prior contacts and is closely related to that sentinel; `balance` is highly skewed with extreme values. Their validation permutation importances were poutcome={roles['poutcome']:.5f}, contact={roles['contact']:.5f}, campaign={roles['campaign']:.5f}, pdays={roles['pdays']:.5f}, previous={roles['previous']:.5f}, and balance={roles['balance']:.5f}. Permutation importance measures this fitted model's validation reliance under the observed feature distribution. Correlated features can share or mask importance; it is not causal evidence and may shift across samples. `duration` remains excluded because it is measured after the call and leaks information for pre-call prediction.

No further model or threshold changes were made after the one-time final test evaluation. See `reports/final_model_evaluation.md` for final test metrics. The final fitted pipeline and decision configuration are under `models/final/`.
'''
    (ROOT/"reports/phase4_optimization.md").write_text(opt_report,encoding="utf-8")
    final_report=f'''# Final model evaluation

## Locked decision

Selected model: **{selected_name}**. Selected threshold: **{selected_threshold:.2f}**, chosen as the maximum validation F1 point after model selection. Final pipeline was refitted on train + validation ({len(fit_idx):,} rows); the untouched test partition was evaluated exactly once afterward.

## Validation and test performance

| Metric | Validation | Test |
|---|---:|---:|
| Average Precision | {val_selected['average_precision']:.4f} | {test['average_precision']:.4f} |
| ROC-AUC | {val_selected['roc_auc']:.4f} | {test['roc_auc']:.4f} |
| Precision | {val_selected['precision']:.4f} | {test['precision']:.4f} |
| Recall | {val_selected['recall']:.4f} | {test['recall']:.4f} |
| F1 | {val_selected['f1']:.4f} | {test['f1']:.4f} |
| Balanced accuracy | {val_selected['balanced_accuracy']:.4f} | {test['balanced_accuracy']:.4f} |
| Accuracy | {val_selected['accuracy']:.4f} | {test['accuracy']:.4f} |

Test confusion matrix, rows=true [no, yes], columns=predicted [no, yes]: **[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]**.

## Classification report (test)

```text
{class_report}```

The validation/test gap is descriptive evidence about generalisation, not a basis for another model or threshold change. Validation predictions come from a training-only fitted candidate; final test predictions come from the selected pipeline refitted on train + validation. No tuning or threshold changes were made using test outcomes.

## Reproducibility and artifacts

Raw SHA-256: `{digest}`. Seed: {SEED}. Existing saved split indices reused. Features exclude `duration`. Metrics are in `reports/tables/final_test_metrics.csv`; final confusion plot is `reports/figures/final_confusion_matrix.png`. Final pipeline: `models/final/final_pipeline.joblib`; selected threshold and search configuration: `models/final/final_config.json`.
'''
    (ROOT/"reports/final_model_evaluation.md").write_text(final_report,encoding="utf-8")

    print(f"Selected {selected_name} threshold={selected_threshold:.2f}; validation AP={val_selected['average_precision']:.4f}, F1={val_selected['f1']:.4f}; test AP={test['average_precision']:.4f}, F1={test['f1']:.4f}; test evaluated once")

if __name__=="__main__": main()
