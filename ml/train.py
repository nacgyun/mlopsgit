# ml/train.py
# Telco Churn (MinIO/S3) + OHE + XGBoost (fallback: RandomForest)
import os, time, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow as mlmod
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models import infer_signature
from mlflow import sklearn as ml_sklearn

# 네가 쓰던 로깅 유틸
from logs import log_json_line, log_ghcr_metadata_to_mlflow

# =========================
# 환경변수
# =========================
TELCO_CSV_URI        = os.getenv("TELCO_CSV_URI", "s3://data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv")
EXP_NAME             = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn")
RUN_NAME             = (os.getenv("GIT_SHA", "")[:12] or "run")
REGISTER_MODEL_NAME  = os.getenv("REGISTER_MODEL_NAME", "ChurnModel")
MODEL_STAGE          = os.getenv("MODEL_STAGE", "Production")   # Production/Staging/None

RANDOM_STATE         = int(os.getenv("SEED", "42"))
TARGET_WALL_SEC      = float(os.getenv("TARGET_WALL_SEC", "0"))   # 데모용 시간 늘리기(0=비활성)
SPEND_CHUNK_SEC      = float(os.getenv("SPEND_CHUNK_SEC", "0.2"))

# XGBoost 기본 하이퍼 (조금 보수적)
XGB_N_ESTIMATORS     = int(os.getenv("XGB_N_ESTIMATORS", "400"))
XGB_MAX_DEPTH        = int(os.getenv("XGB_MAX_DEPTH", "6"))
XGB_LR               = float(os.getenv("XGB_LEARNING_RATE", "0.05"))
XGB_SUBSAMPLE        = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
XGB_COLSAMPLE        = float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8"))
XGB_REG_LAMBDA       = float(os.getenv("XGB_REG_LAMBDA", "1.0"))

# =========================
# 유틸
# =========================
def ensure_experiment_id(name: str, client: MlflowClient, retries: int = 20, sleep: float = 0.25) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        exp_id = None
        try:
            exp_id = client.create_experiment(name)
        except RestException as e:
            if "RESOURCE_ALREADY_EXISTS" in str(e) or "UNIQUE constraint failed" in str(e):
                exp = client.get_experiment_by_name(name)
                if exp is not None:
                    exp_id = exp.experiment_id
            else:
                raise
        if exp_id is None:
            for i in range(retries):
                exp = client.get_experiment_by_name(name)
                if exp is not None:
                    exp_id = exp.experiment_id
                    break
                time.sleep(sleep * (1.5 ** i))
            if exp_id is None:
                raise RuntimeError(f"Failed to ensure experiment '{name}' exists")
    else:
        exp_id = exp.experiment_id

    for i in range(retries):
        try:
            if client.get_experiment(exp_id) is not None:
                return exp_id
        except RestException:
            pass
        time.sleep(sleep * (1.5 ** i))

    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    raise RuntimeError(f"Experiment id for '{name}' could not be validated")

def maybe_import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except Exception:
        return None

def load_telco(csv_uri: str):
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    access   = os.getenv("AWS_ACCESS_KEY_ID")
    secret   = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not endpoint or not access or not secret:
        raise RuntimeError("Missing S3 creds: MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")

    storage_options = {
        "client_kwargs": {"endpoint_url": endpoint},
        "key": access,
        "secret": secret
    }

    df = pd.read_csv(csv_uri, storage_options=storage_options)

    # 표준 칼럼 정리
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" not in df.columns:
        raise ValueError("CSV에 'Churn' 칼럼이 없습니다.")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    y = df["Churn"].values
    X = df.drop(columns=["Churn"])

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if pd.api.types.is_object_dtype(X[c])]

    return X, y, cat_cols, num_cols

class OHE_XGB_Classifier(BaseEstimator, ClassifierMixin):
    """
    OneHotEncoder(범주) + StandardScaler(수치) + XGBClassifier 묶은 래퍼
    - 검증셋으로 임계값(best_threshold) 튜닝(F1 macro 최대화)
    """
    def __init__(self, xgb_params=None, random_state=42):
        self.xgb_params = xgb_params or {}
        self.random_state = random_state
        self.preproc_ = None
        self.model_ = None
        self.best_threshold_ = 0.5

    def fit(self, X, y, cat_cols, num_cols, X_val=None, y_val=None):
        # 전처리자
        preproc = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop"
        )
        X_tr = preproc.fit_transform(X)

        # 클래스 불균형 보정: scale_pos_weight = neg/pos
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        spw = float(neg) / max(1.0, float(pos))

        XGBClassifier = maybe_import_xgb()
        if XGBClassifier is not None:
            model = XGBClassifier(
                n_estimators=self.xgb_params.get("n_estimators", 400),
                max_depth=self.xgb_params.get("max_depth", 6),
                learning_rate=self.xgb_params.get("learning_rate", 0.05),
                subsample=self.xgb_params.get("subsample", 0.8),
                colsample_bytree=self.xgb_params.get("colsample_bytree", 0.8),
                reg_lambda=self.xgb_params.get("reg_lambda", 1.0),
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=self.random_state,
                scale_pos_weight=spw,
            )
        else:
            # 폴백: RandomForest
            model = RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=self.random_state,
                class_weight="balanced_subsample"
            )

        model.fit(X_tr, y)

        self.preproc_ = preproc
        self.model_ = model
        self.best_threshold_ = 0.5

        # 임계값 튜닝(F1 macro)
        if X_val is not None and y_val is not None:
            Xv = self.preproc_.transform(X_val)
            if hasattr(self.model_, "predict_proba"):
                pv = self.model_.predict_proba(Xv)[:, 1]
            else:
                # 일부 모델은 decision_function만 있을 수 있음
                if hasattr(self.model_, "decision_function"):
                    from scipy.special import expit
                    pv = expit(self.model_.decision_function(Xv))
                else:
                    pv = self.model_.predict(Xv).astype(float)

            best_f1, best_t = -1.0, 0.5
            # 간단한 0.05 step 스캔
            for t in np.linspace(0.2, 0.8, 25):
                y_hat = (pv >= t).astype(int)
                f1 = f1_score(y_val, y_hat, average="macro")
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            self.best_threshold_ = best_t

        return self

    def predict(self, X):
        X2 = self.preproc_.transform(X)
        if hasattr(self.model_, "predict_proba"):
            p = self.model_.predict_proba(X2)[:, 1]
            return (p >= self.best_threshold_).astype(int)
        return self.model_.predict(X2)

    def predict_proba(self, X):
        X2 = self.preproc_.transform(X)
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X2)
        # 폴백: 예측값을 0/1 확률로 변환
        y = self.model_.predict(X2).astype(int)
        return np.vstack([1 - y, y]).T

def maybe_spend_until(deadline_ts, fn_iter=None):
    # (선택) 데모용으로 벽시계 시간 맞추기
    if deadline_ts <= 0:
        return
    while time.perf_counter() < deadline_ts:
        if fn_iter:
            fn_iter()
        time.sleep(SPEND_CHUNK_SEC)

# =========================
# 메인
# =========================
def main():
    wall_start = time.perf_counter()

    # MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # 데이터
    X, y, cat_cols, num_cols = load_telco(TELCO_CSV_URI)

    # Train/Val/Test (70/15/15)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_tmp)  # 0.1765 ≈ 15% of original

    with mlmod.start_run(experiment_id=exp_id, run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        log_ghcr_metadata_to_mlflow()

        # 파라미터 로깅
        mlmod.log_params({
            "dataset": "Telco-Customer-Churn (MinIO/S3)",
            "TELCO_CSV_URI": TELCO_CSV_URI,
            "model_family": "XGBoost(fallback:RandomForest)",
            "xgb_n_estimators": XGB_N_ESTIMATORS,
            "xgb_max_depth": XGB_MAX_DEPTH,
            "xgb_learning_rate": XGB_LR,
            "xgb_subsample": XGB_SUBSAMPLE,
            "xgb_colsample_bytree": XGB_COLSAMPLE,
            "xgb_reg_lambda": XGB_REG_LAMBDA,
        })

        # 모델 생성/학습 + 임계값 튜닝
        xgb_params = dict(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LR,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
            reg_lambda=XGB_REG_LAMBDA,
        )
        clf = OHE_XGB_Classifier(xgb_params=xgb_params, random_state=RANDOM_STATE)

        t0 = time.perf_counter()
        clf.fit(X_train, y_train, cat_cols=cat_cols, num_cols=num_cols, X_val=X_val, y_val=y_val)
        train_time = time.perf_counter() - t0
        mlmod.log_metric("train_time_sec", float(train_time))
        mlmod.log_metric("best_threshold", float(clf.best_threshold_))

        # 검증/테스트 평가
        for split_name, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
            y_hat = clf.predict(Xs)
            acc = float(accuracy_score(ys, y_hat))
            f1  = float(f1_score(ys, y_hat, average="macro"))

            try:
                proba = clf.predict_proba(Xs)[:,1]
                ll = float(log_loss(ys, np.vstack([1-proba, proba]).T))
                auc = float(roc_auc_score(ys, proba))
            except Exception:
                ll, auc = float("nan"), float("nan")

            mlmod.log_metrics({
                f"{split_name}_accuracy": acc,
                f"{split_name}_f1_macro": f1,
                f"{split_name}_log_loss": ll,
                f"{split_name}_roc_auc": auc,
            })

            log_json_line({
                "event": f"eval_{split_name}",
                "accuracy": acc,
                "f1_macro": f1,
                "log_loss": ll,
                "roc_auc": auc,
                "threshold": float(clf.best_threshold_),
                "run_id": run_id,
                "experiment": EXP_NAME,
            })

            # 혼동행렬 아티팩트
            cm = confusion_matrix(ys, y_hat)
            plt.figure(figsize=(5.2,4.2))
            im = plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix ({split_name})")
            plt.colorbar(im)
            tick = np.arange(2)
            plt.xticks(tick, ["No","Yes"])
            plt.yticks(tick, ["No","Yes"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center")
            plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
            fn = f"confusion_matrix_{split_name}.png"
            plt.savefig(fn, bbox_inches="tight")
            mlmod.log_artifact(fn, artifact_path="plots")

        # 모델 등록 (파이썬 객체 자체를 기록: 전처리 포함)
        # 예시 input_example은 원본(raw) 피처 일부
        input_example = X_test.iloc[:2].copy()
        signature = infer_signature(X_test, clf.predict_proba(X_test))

        ml_sklearn.log_model(
            sk_model=clf,                  # 전처리+모델 일체형
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTER_MODEL_NAME
        )

        # 스테이지 전환
        try:
            latest = client.get_latest_versions(name=REGISTER_MODEL_NAME, stages=["None","Staging","Production"])
            if latest:
                latest = sorted(latest, key=lambda v: v.creation_timestamp, reverse=True)[0]
                if MODEL_STAGE and MODEL_STAGE.lower() != "none":
                    client.transition_model_version_stage(
                        name=REGISTER_MODEL_NAME,
                        version=latest.version,
                        stage=MODEL_STAGE,
                        archive_existing_versions=True
                    )
                    print(f"[mlflow] transitioned {REGISTER_MODEL_NAME} v{latest.version} -> {MODEL_STAGE}")
        except Exception as e:
            print(f"[mlflow] stage transition skipped: {e}")

        # 데모용: 목표 시간까지 맞추기(선택)
        if TARGET_WALL_SEC > 0:
            deadline = wall_start + TARGET_WALL_SEC
            sample = X_train.sample(min(1500, len(X_train)), random_state=RANDOM_STATE)
            def _iter(): 
                try: _ = clf.predict(sample)
                except: pass
            maybe_spend_until(deadline, _iter)

        total_time = time.perf_counter() - wall_start
        mlmod.log_metric("total_wall_time_sec", float(total_time))
        print("✅ Train done.", flush=True)

if __name__ == "__main__":
    main()

