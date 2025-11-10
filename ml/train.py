# ml/train.py (Telco Churn only)
# -*- coding: utf-8 -*-
"""
MinIO(S3) 경로의 Telco Churn CSV를 읽어 학습하고 MLflow에 기록/등록하는 전용 스크립트.

필수 ENV 예시
- MLFLOW_TRACKING_URI=http://<mlflow-host>:5000
- MLFLOW_EXPERIMENT_NAME=telco-churn
- TELCO_CSV_URI=s3://data/telco/Telco-Customer-Churn.csv
- (MinIO 접속)
  - MLFLOW_S3_ENDPOINT_URL=http://<nodeip>:<nodeport>
  - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  - AWS_S3_ADDRESSING_STYLE=path
  - AWS_S3_FORCE_PATH_STYLE=true

선택 ENV
- REGISTER_MODEL_NAME=ChurnModel
- MODEL_STAGE=Staging|Production

필수 패키지: pandas, scikit-learn, mlflow, s3fs, fsspec
"""

import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow as mlmod
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss

# 선택 로그 모듈 (없어도 동작)
try:
    from logs import log_json_line, log_ghcr_metadata_to_mlflow
except Exception:
    def log_json_line(obj): print(json.dumps(obj, ensure_ascii=False))
    def log_ghcr_metadata_to_mlflow(): pass

# ========= 설정 =========
EXP_NAME        = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn")
RUN_NAME        = (os.getenv("GIT_SHA", "")[:12] or "run")

# 학습 루프 파라미터
EPOCHS          = int(os.getenv("MLFLOW_EPOCHS", "20"))
BATCH_SIZE      = int(os.getenv("TRAIN_BATCH_SIZE", "4096"))
RANDOM_STATE    = int(os.getenv("SEED", "42"))
LR_ALPHA        = float(os.getenv("LR_ALPHA", "0.0005"))

TARGET_WALL_SEC = float(os.getenv("TARGET_WALL_SEC", "180"))
EMA_ALPHA       = float(os.getenv("ETA_EMA_ALPHA", "0.2"))

TELCO_CSV_URI   = os.getenv("TELCO_CSV_URI", "s3://data/telco/Telco-Customer-Churn.csv")


# ========= 유틸 =========
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


def start_run_with_retry(exp_id: str, run_name: str, retries: int = 12, delay: float = 0.3, backoff: float = 1.5):
    last = None
    for _ in range(retries):
        try:
            return mlmod.start_run(experiment_id=exp_id, run_name=run_name)
        except RestException as e:
            last = e
            if "No Experiment with id" not in str(e) and "RESOURCE_DOES_NOT_EXIST" not in str(e):
                raise
            time.sleep(delay); delay *= backoff
    if last:
        raise last


def _s3_storage_options_from_env():
    """pandas.read_csv(storage_options=...)에 전달할 MinIO/S3 옵션 구성 (호환성 위해 단순화)"""
    opts = {}
    key = os.getenv("AWS_ACCESS_KEY_ID")
    sec = os.getenv("AWS_SECRET_ACCESS_KEY")
    tok = os.getenv("AWS_SESSION_TOKEN")
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")

    if key: opts["key"] = key
    if sec: opts["secret"] = sec
    if tok: opts["token"] = tok
    if endpoint:
        # ✅ aiobotocore 조합 호환 위해 endpoint_url만 설정
        opts.setdefault("client_kwargs", {})["endpoint_url"] = endpoint
    return opts


def batch_iter(X, y, batch_size, shuffle=True, seed=None):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = idx[start:end]
        yield X[b], y[b]


def load_telco_churn(csv_uri: str):
    storage_options = _s3_storage_options_from_env() if csv_uri.startswith("s3://") else None
    df = pd.read_csv(csv_uri, storage_options=storage_options)

    # 기본 전처리
    for c in ("customerID", "CustomerID", "customerId"):
        if c in df.columns:
            df = df.drop(columns=[c]); break

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 타깃/결측 제거
    df = df.dropna(subset=[col for col in ["Churn", "TotalCharges"] if col in df.columns])
    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn"])

    # 컬럼 타입 분리
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 전처리자: 호환성 위해 OneHotEncoder(sparse=False) 사용
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return X, y, preproc, cat_cols, num_cols


# ========= 메인 =========
def main():
    wall_start = time.perf_counter()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 데이터 로드
    X_all, y_all, preproc, cat_cols, num_cols = load_telco_churn(TELCO_CSV_URI)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.30, random_state=0, stratify=y_all
    )

    # 전처리 적합 & 변환 -> ndarray
    preproc.fit(X_train_df)
    X_train = preproc.transform(X_train_df)
    X_test  = preproc.transform(X_test_df)

    # SGD 분류기 (partial_fit로 에폭 학습)
    classes = np.array([0, 1], dtype=int)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=LR_ALPHA,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
        fit_intercept=True,
        max_iter=1,
        warm_start=False,
        tol=None,
    )

    # 실험 보장
    exp_id = ensure_experiment_id(EXP_NAME, client)

    with start_run_with_retry(exp_id, RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}, exp_id={exp_id}")

        # (선택) GHCR 메타데이터 기록
        log_ghcr_metadata_to_mlflow()

        mlmod.log_params({
            "dataset": "telco",
            "model": "SGDClassifier(logistic)",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "alpha": LR_ALPHA,
            "target_wall_sec": TARGET_WALL_SEC,
        })

        # 첫 partial_fit(classes 지정)
        first_n = min(BATCH_SIZE, len(X_train))
        clf.partial_fit(X_train[:first_n], y_train[:first_n], classes=classes)

        f1_hist = []
        ema = None

        for epoch in range(1, EPOCHS + 1):
            t_epoch = time.perf_counter()

            for Xb, yb in batch_iter(X_train, y_train, BATCH_SIZE, shuffle=True,
                                     seed=RANDOM_STATE + epoch * 1000):
                clf.partial_fit(Xb, yb)

            compute_sec = time.perf_counter() - t_epoch
            y_pred = clf.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1  = float(f1_score(y_test, y_pred, average="macro"))
            try:
                y_proba = clf.predict_proba(X_test)
                ll = float(log_loss(y_test, y_proba))
            except Exception:
                ll = float("nan")

            ema = compute_sec if ema is None else (EMA_ALPHA * compute_sec + (1 - EMA_ALPHA) * ema)
            elapsed = time.perf_counter() - wall_start
            eta_sec = max(0.0, TARGET_WALL_SEC - elapsed)

            mlmod.log_metrics({
                "accuracy": acc,
                "f1_score": f1,
                "log_loss": ll,
                "epoch_compute_sec": compute_sec,
                "elapsed_sec": elapsed,
                "eta_sec": eta_sec,
                "progress_pct": min(99.9, 100.0 * epoch / EPOCHS),
            }, step=epoch)

            f1_hist.append(f1)
            print(f"[epoch {epoch:03d}] acc={acc:.4f} f1={f1:.4f} comp={compute_sec:.2f}s "
                  f"elapsed={elapsed:.1f}s ETA~{eta_sec:.1f}s")
            log_json_line({
                "event": "epoch_metric",
                "epoch": epoch,
                "accuracy": acc,
                "duration": round(compute_sec, 4),
                "remaining_sec": round(eta_sec, 1),
                "run_id": run_id,
                "experiment": EXP_NAME,
            })

            if time.perf_counter() - wall_start >= TARGET_WALL_SEC:
                print(f"[info] target wall time ({TARGET_WALL_SEC:.0f}s) reached. stopping early.")
                break

        total_time = time.perf_counter() - wall_start
        mlmod.log_metric("train_time_total_sec", total_time)

        # 혼동행렬 저장
        cm = confusion_matrix(y_test, clf.predict(X_test))
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (Telco)")
        plt.colorbar(im)
        tick = np.arange(2)
        plt.xticks(tick, ["stay", "churn"], rotation=0)
        plt.yticks(tick, ["stay", "churn"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig("confusion_matrix.png", bbox_inches="tight")
        mlmod.log_artifact("confusion_matrix.png", artifact_path="plots")

        # F1 히스토리 저장
        plt.figure(figsize=(6, 3.5))
        plt.plot(range(1, len(f1_hist)+1), f1_hist, marker="o")
        plt.title("F1 over Epochs (Telco)")
        plt.xlabel("Epoch"); plt.ylabel("F1 (macro)"); plt.grid(True, alpha=0.3)
        plt.savefig("learning_curve_f1.png", bbox_inches="tight")
        mlmod.log_artifact("learning_curve_f1.png", artifact_path="plots")

        # 파이프라인(전처리+분류기)로 저장
        from mlflow import sklearn as ml_sklearn
        final_pipe = Pipeline(steps=[("preproc", preproc), ("clf", clf)])
        sig = infer_signature(X_train_df, final_pipe.predict(X_train_df.head(2)))
        ml_sklearn.log_model(
            sk_model=final_pipe,
            artifact_path="model",
            signature=sig,
            input_example=X_train_df.head(2),
        )

        with open("input_example.json", "w") as f:
            json.dump(X_train_df.head(2).to_dict(orient="records"), f)
        mlmod.log_artifact("input_example.json")

        final_acc_num = float(accuracy_score(y_test, final_pipe.predict(X_test_df)))
        log_json_line({
            "event": "train_done",
            "accuracy": final_acc_num,
            "duration": round(total_time, 4),
            "remaining_sec": 0.0,
            "run_id": run_id,
            "experiment": EXP_NAME,
        })

        # (옵션) 레지스트리 등록/스테이지 전환
        reg_name = os.getenv("REGISTER_MODEL_NAME", "").strip()
        if reg_name:
            try:
                try:
                    client.get_registered_model(reg_name)
                except Exception:
                    client.create_registered_model(reg_name)
                mv = mlmod.register_model(model_uri=f"runs:/{run_id}/model", name=reg_name)
                stage = os.getenv("MODEL_STAGE", "").strip()
                if stage:
                    client.transition_model_version_stage(reg_name, mv.version, stage)
                    print(f"[mlflow] transitioned {reg_name} v{mv.version} -> {stage}")
            except Exception as e:
                print(f"[warn] model registry step failed: {e}")

        print(f"[PROMOTE] accuracy={final_acc_num:.5f}", flush=True)
        print("✅ Train done.", flush=True)


if __name__ == "__main__":
    main()

