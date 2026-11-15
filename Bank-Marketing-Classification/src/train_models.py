"""Phase 3 baseline comparison. Test split is created and saved but never scored."""
from pathlib import Path
import hashlib
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data/raw/bank-full.csv"
TABLES=ROOT/"reports/tables"; FIGURES=ROOT/"reports/figures"
ARTIFACTS=ROOT/"models/candidates"
EXPECTED_SHA256="d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"
SEED=42
NUMERIC=["age","balance","day","campaign","pdays","previous"]
CATEGORICAL=["job","marital","education","default","housing","loan","contact","month","poutcome"]
FEATURES=NUMERIC+CATEGORICAL
METRICS={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1",
         "roc_auc":"roc_auc","average_precision":"average_precision","balanced_accuracy":"balanced_accuracy"}

def make_pipeline(kind, class_weight=None, unknown_as_missing=False):
    numeric_steps=[("imputer",SimpleImputer(strategy="median"))]
    if kind=="logistic": numeric_steps.append(("scaler",StandardScaler()))
    num_pipe=Pipeline(numeric_steps)
    cat_steps=[]
    # The original source has no actual nulls; only the unknown-as-missing experiment needs a categorical imputer.
    if unknown_as_missing: cat_steps.append(("imputer",SimpleImputer(strategy="most_frequent")))
    cat_steps.append(("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False if kind=="hist_gradient_boosting" else True,
                                               dtype=np.float32)))
    cat_pipe=Pipeline(cat_steps)
    transform=ColumnTransformer([("numeric",num_pipe,NUMERIC),("categorical",cat_pipe,CATEGORICAL)],remainder="drop")
    if kind=="dummy": clf=DummyClassifier(strategy="most_frequent")
    elif kind=="logistic": clf=LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000,class_weight=class_weight,random_state=SEED)
    elif kind=="random_forest": clf=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_features="sqrt",
              class_weight=class_weight,random_state=SEED,n_jobs=-1)
    elif kind=="hist_gradient_boosting": clf=HistGradientBoostingClassifier(max_iter=150,learning_rate=.1,max_leaf_nodes=31,
              l2_regularization=1.0,random_state=SEED,early_stopping=True)
    else: raise ValueError(kind)
    return Pipeline([("preprocess",transform),("classifier",clf)])

def prepare_variant(X, unknown_as_missing):
    if not unknown_as_missing: return X
    X=X.copy()
    for c in CATEGORICAL: X[c]=X[c].replace("unknown",np.nan)
    return X

def score_predictions(y, pred, prob):
    return {"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),
            "recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),
            "roc_auc":roc_auc_score(y,prob),"average_precision":average_precision_score(y,prob),
            "balanced_accuracy":balanced_accuracy_score(y,pred)}

def main():
    TABLES.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True); ARTIFACTS.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA256: raise RuntimeError(f"bank-full.csv hash mismatch: {digest}")
    df=pd.read_csv(RAW,sep=";",quotechar='"')
    if df.shape!=(45211,17): raise RuntimeError(f"Unexpected raw dimensions {df.shape}")
    if df[FEATURES+['duration','y']].isna().any().any(): raise RuntimeError("Raw dataset contains nulls unlike Phase 1")
    X=df[FEATURES].copy()
    y=(df["y"]=="yes").astype("int8")
    # Two-stage stratified partition: 70% train, then split remaining evenly into 15% validation/test.
    train_idx,rest_idx=train_test_split(np.arange(len(df)),test_size=.30,random_state=SEED,stratify=y)
    val_idx,test_idx=train_test_split(rest_idx,test_size=.50,random_state=SEED,stratify=y.iloc[rest_idx])
    split=np.full(len(df),"",dtype=object); split[train_idx]="train"; split[val_idx]="validation"; split[test_idx]="test"
    pd.DataFrame({"row_index":np.arange(len(df)),"split":split}).to_csv(TABLES/"split_indices.csv",index=False)
    # The test indices are saved only. No test X/y is constructed, transformed, predicted, or scored below.
    X_train=prepare_variant(X.iloc[train_idx],False); X_val=prepare_variant(X.iloc[val_idx],False)
    y_train=y.iloc[train_idx]; y_val=y.iloc[val_idx]

    candidates=[
      ("dummy_most_frequent","dummy",None,False),
      ("logistic_none","logistic",None,False),
      ("logistic_balanced","logistic","balanced",False),
      ("random_forest_none","random_forest",None,False),
      ("random_forest_balanced_subsample","random_forest","balanced_subsample",False),
      ("hist_gradient_boosting","hist_gradient_boosting",None,False),
      ("logistic_balanced_unknown_missing","logistic","balanced",True),
      ("random_forest_balanced_unknown_missing","random_forest","balanced_subsample",True),
    ]
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    cv_rows=[]; val_rows=[]; fitted={}
    scoring={key:value for key,value in METRICS.items()}
    for name,kind,class_weight,unknown_missing in candidates:
        Xtr=prepare_variant(X.iloc[train_idx],unknown_missing)
        Xva=prepare_variant(X.iloc[val_idx],unknown_missing)
        pipe=make_pipeline(kind,class_weight,unknown_missing)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result=cross_validate(pipe,Xtr,y_train,cv=cv,scoring=scoring,return_train_score=False,n_jobs=1,error_score="raise")
        row={"candidate":name,"model":kind,"unknown_treatment":"unknown_as_missing" if unknown_missing else "explicit_category",
             "class_weight":class_weight or "none"}
        for metric in METRICS:
            vals=result[f"test_{metric}"]; row[f"{metric}_mean"]=float(np.mean(vals)); row[f"{metric}_std"]=float(np.std(vals,ddof=1))
        cv_rows.append(row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr,y_train)
        pred=pipe.predict(Xva)
        proba=pipe.predict_proba(Xva)[:,list(pipe.classes_).index(1)]
        scores=score_predictions(y_val,pred,proba); tn,fp,fn,tp=confusion_matrix(y_val,pred,labels=[0,1]).ravel()
        val_rows.append({"candidate":name,"model":kind,"unknown_treatment":row["unknown_treatment"],"class_weight":row["class_weight"],
             "threshold":0.5,"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),**scores})
        fitted[name]=pipe
        joblib.dump(pipe,ARTIFACTS/f"{name}.joblib",compress=3)

    cvdf=pd.DataFrame(cv_rows).sort_values("average_precision_mean",ascending=False)
    valdf=pd.DataFrame(val_rows).sort_values("average_precision",ascending=False)
    cvdf.to_csv(TABLES/"cv_results.csv",index=False)
    valdf.to_csv(TABLES/"validation_results.csv",index=False)
    # Validation-only visual comparisons; no threshold search or test evaluation.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics=["accuracy","precision","recall","f1","roc_auc","average_precision","balanced_accuracy"]
    labels={"dummy_most_frequent":"Dummy","logistic_none":"LR","logistic_balanced":"LR balanced",
      "random_forest_none":"RF","random_forest_balanced_subsample":"RF balanced",
      "hist_gradient_boosting":"HistGB","logistic_balanced_unknown_missing":"LR balanced; unknown→missing",
      "random_forest_balanced_unknown_missing":"RF balanced; unknown→missing"}
    short=[labels[n] for n in cvdf.candidate]
    fig,axes=plt.subplots(4,2,figsize=(14,16))
    for ax,m in zip(axes.flat,metrics):
        ax.barh(np.arange(len(cvdf)),cvdf[f"{m}_mean"],xerr=cvdf[f"{m}_std"],color="#397ca5",capsize=2)
        ax.set_yticks(np.arange(len(cvdf)),short,fontsize=8); ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_title(f"Training CV: {m} (mean ± 1 SD)"); ax.set_xlabel("Score")
    ax=axes.flat[-1]; xv=np.arange(len(valdf))
    ax.barh(xv-.18,valdf["average_precision"],.36,label="Average precision",color="#397ca5")
    ax.barh(xv+.18,valdf["balanced_accuracy"],.36,label="Balanced accuracy",color="#d89038")
    ax.set_yticks(xv,[labels[n] for n in valdf.candidate],fontsize=8); ax.invert_yaxis(); ax.set_xlim(0,1); ax.legend(fontsize=8); ax.set_title("Validation at threshold 0.5"); ax.set_xlabel("Score")
    fig.suptitle("Initial model comparison — test split not evaluated",fontsize=15); fig.tight_layout(rect=[0,0,1,.985]); fig.savefig(FIGURES/"model_comparison.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    from sklearn.metrics import ConfusionMatrixDisplay
    for ax,(_,r) in zip(axes.flat,valdf.iterrows()):
        cm=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int)
        ConfusionMatrixDisplay(cm,display_labels=["no","yes"]).plot(ax=ax,colorbar=False,values_format="d",cmap="Blues")
        ax.set_title(str(r.candidate),fontsize=9)
    for ax in axes.flat[len(valdf):]: ax.axis("off")
    fig.suptitle("Validation confusion matrices at threshold 0.5",y=1.01); fig.tight_layout(); fig.savefig(FIGURES/"confusion_matrices.png",dpi=170,bbox_inches="tight"); plt.close(fig)

    best=valdf.iloc[0]
    split_counts={s:{"rows":int((split==s).sum()),"yes":int(y.iloc[np.where(split==s)[0]].sum())} for s in ("train","validation","test")}
    config={"raw_sha256":digest,"dimensions":list(df.shape),"seed":SEED,"split_rows":split_counts,
      "features":FEATURES,"excluded_features":["duration"],"target":"y","positive_class":"yes",
      "threshold":0.5,"cross_validation":"StratifiedKFold(n_splits=5, shuffle=True, random_state=42), training only",
      "candidate_specs":[{"name":n,"kind":k,"class_weight":w,"unknown_as_missing":u} for n,k,w,u in candidates],
      "test_set_scored":False}
    (ARTIFACTS/"experiment_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    report=f'''# Phase 3 — Initial model comparison

## Data integrity and protocol

Raw SHA-256: `{digest}` (matches Phase 1); dimensions: 45,211 × 17. Raw CSV was read only. Target `y`, positive class `yes` (encoded 1). `duration` was excluded from all primary model features. Stratified 70/15/15 partition with seed {SEED}; exact row indices are in `tables/split_indices.csv`. Split class counts are recorded in `models/candidates/experiment_config.json`. The test partition was not scored, used for fitting, or used for candidate selection.

All preprocessing is inside sklearn `Pipeline`/`ColumnTransformer`: numeric median imputation; standard scaling only for Logistic Regression; categorical one-hot encoding with `handle_unknown='ignore'`. The raw data has no actual nulls. In the unknown-as-missing experiments, only the `unknown` categorical labels are converted to null and most-frequent imputation is fitted inside each training fold. The explicit-category candidates preserve them as categories. No outlier filtering or resampling was used.

## Cross-validation on training partition

Five-fold shuffled stratified CV, seed {SEED}, was run on training data only. Means and sample standard deviations are in `tables/cv_results.csv`; models include Dummy, regularized Logistic Regression (`class_weight=None` and `balanced`), Random Forest (`None` and `balanced_subsample`), HistGradientBoosting, and unknown-as-missing sensitivity candidates for balanced Logistic Regression and Random Forest. This is a small initial comparison, not an extensive search.

## Validation results

Every candidate was fitted on training data only and scored once on validation at threshold 0.5. Full metrics and confusion counts are in `tables/validation_results.csv`; plots are `figures/model_comparison.png` and `figures/confusion_matrices.png`.

**Best validation candidate by Average Precision:** `{best.candidate}` ({best.average_precision:.4f}); balanced accuracy {best.balanced_accuracy:.4f}, precision {best.precision:.4f}, recall {best.recall:.4f}, F1 {best.f1:.4f}, ROC-AUC {best.roc_auc:.4f}, accuracy {best.accuracy:.4f}. Confusion matrix (TN, FP, FN, TP) = ({int(best.tn)}, {int(best.fp)}, {int(best.fn)}, {int(best.tp)}).

This identifies the current validation leader only. The unknown handling comparison uses training CV and validation results; no test results informed any choice. No threshold exploration or hyperparameter tuning was performed.

## Saved artifacts and next step

Fitted candidate pipelines and `experiment_config.json` are in `models/candidates/`. They are fitted on the training partition only. Proceed to threshold trade-offs and any further validation analysis only after review/approval. Test evaluation remains for a later approved phase.
'''
    (ROOT/"reports/model_comparison.md").write_text(report,encoding="utf-8")
    print(f"Best validation candidate by AP: {best.candidate} | AP={best.average_precision:.4f} | F1={best.f1:.4f} | precision={best.precision:.4f} | recall={best.recall:.4f}")

if __name__=="__main__": main()
