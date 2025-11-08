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
  - AWS_S3_ADDRESSING_STYLE=path (권장)
  - AWS_S3_FORCE_PATH_STYLE=true (권장)

선택 ENV
- REGISTER_MODEL_NAME=ChurnModel
- MODEL_STAGE=Staging|Production

필수 패키지: pandas, scikit-learn, mlflow, s3fs
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

# 🔹 로그 함수 모듈(있으면 사용, 없어도 동작)
try:
    from logs import log_json_line, log_ghcr_metadata_to_mlflow
except Exception:
    def log_json_line(obj):
        print(json.dumps(obj, ensure_ascii=False))
    def log_ghcr_metadata_to_mlflow():
        pass

# ===== 파라미터 / 설정 =====
EXP_NAME        = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn")
RUN_NAME        = (os.getenv("GIT_SHA", "")[:12] or "run")

EPOCHS          = int(os.getenv("MLFLOW_EPOCHS", "20"))
BATCH_SIZE      = int(os.getenv("TRAIN_BATCH_SIZE", "2048"))  # 파이프라인 전체 fit이므로 크~게
RANDOM_STATE    = int(os.getenv("SEED", "42"))
LR_ALPHA        = float(os.getenv("LR_ALPHA", "0.0005"))
TARGET_WALL_SEC = float(os.getenv("TARGET_WALL_SEC", "180"))
EMA_ALPHA       = float(os.getenv("ETA_EMA_ALPHA", "0.2"))

TELCO_CSV_URI   = os.getenv("TELCO_CSV_URI", "s3://data/telco/Telco-Customer-Churn.csv")

# ========= 공용 유틸 =========

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
    """pandas.read_csv(storage_options=...)에 전달할 MinIO/S3 옵션 구성"""
    opts = {}
    key = os.getenv("AWS_ACCESS_KEY_ID")
    sec = os.getenv("AWS_SECRET_ACCESS_KEY")
    tok = os.getenv("AWS_SESSION_TOKEN")
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
    addressing = os.getenv("AWS_S3_ADDRESSING_STYLE", "path")
    force_path = os.getenv("AWS_S3_FORCE_PATH_STYLE", "true").lower() in ("1","true","yes")

    if key: opts["key"] = key
    if sec: opts["secret"] = sec
    if tok: opts["token"] = tok
    if endpoint:
        opts.setdefault("client_kwargs", {})["endpoint_url"] = endpoint
    if force_path or addressing == "path":
        opts.setdefault("client_kwargs", {})["config_kwargs"] = {"s3": {"addressing_style": "path"}}
    return opts


def load_telco_churn(csv_uri: str):
    storage_options = _s3_storage_options_from_env() if csv_uri.startswith("s3://") else None
    df = pd.read_csv(csv_uri, storage_options=storage_options)

    # 기본 정제/형 변환
    for c in ("customerID", "CustomerID", "customerId"):
        if c in df.columns:
            df = df.drop(columns=[c]); break
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=[col for col in ["Churn", "TotalCharges"] if col in df.columns])

    # 타깃
    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn"])

    # 타입 분리
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=LR_ALPHA,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
        fit_intercept=True,
        max_iter=1,   # fit을 여러 epoch 반복 호출
        warm_start=False,
        tol=None,
    )

    pipe = Pipeline(steps=[("preproc", preproc), ("clf", clf)])
    return X, y, pipe, cat_cols, num_cols


def main():
    wall_start = time.perf_counter()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 데이터 로드
    X_all, y_all, model, cat_cols, num_cols = load_telco_churn(TELCO_CSV_URI)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.30, random_state=0, stratify=y_all
    )

    # 실험 보장
    exp_id = ensure_experiment_id(EXP_NAME, client)

    with start_run_with_retry(exp_id, RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}, exp_id={exp_id}")

        # GHCR 메타데이터 기록(선택)
        log_ghcr_metadata_to_mlflow()

        mlmod.log_params({
            "dataset": "telco",
            "model": "SGDClassifier(logistic)",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "alpha": LR_ALPHA,
            "target_wall_sec": TARGET_WALL_SEC,
        })

        f1_hist = []
        ema = None

        # 여러 epoch 동안 전체 fit — SGD라 빠름
        for epoch in range(1, EPOCHS + 1):
            t_epoch = time.perf_counter()
            model.fit(X_train, y_train)

            compute_sec = time.perf_counter() - t_epoch
            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1  = float(f1_score(y_test, y_pred, average="macro"))
            try:
                y_proba = model.predict_proba(X_test)
                ll = float(log_loss(y_test, y_proba))
            except Exception:
                ll = float("nan")

            if ema is None:
                ema = compute_sec
            else:
                ema = EMA_ALPHA * compute_sec + (1 - EMA_ALPHA) * ema

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
            print(f"[epoch {epoch:03d}] acc={acc:.4f} f1={f1:.4f} comp={compute_sec:.2f}s elapsed={elapsed:.1f}s ETA~{eta_sec:.1f}s")
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

        # Confusion Matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (Telco)")
        plt.colorbar(im)
        tick = np.arange(2)
        plt.xticks(tick, ["stay","churn"], rotation=0)
        plt.yticks(tick, ["stay","churn"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig("confusion_matrix.png", bbox_inches="tight")
        mlmod.log_artifact("confusion_matrix.png", artifact_path="plots")

        # F1 History
        plt.figure(figsize=(6, 3.5))
        plt.plot(range(1, len(f1_hist)+1), f1_hist, marker="o")
        plt.title("F1 over Epochs (Telco)")
        plt.xlabel("Epoch"); plt.ylabel("F1 (macro)"); plt.grid(True, alpha=0.3)
        plt.savefig("learning_curve_f1.png", bbox_inches="tight")
        mlmod.log_artifact("learning_curve_f1.png", artifact_path="plots")

        # 모델 로깅
        from mlflow import sklearn as ml_sklearn
        sig = infer_signature(X_train, model.predict(X_train))
        ml_sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=sig,
            input_example=X_train.head(2) if hasattr(X_train, "head") else X_train[:2],
        )

        with open("input_example.json", "w") as f:
            if hasattr(X_train, "head"):
                json.dump(X_train.head(2).to_dict(orient="records"), f)
            else:
                json.dump(np.asarray(X_train[:2]).tolist(), f)
        mlmod.log_artifact("input_example.json")

        final_acc_num = float(accuracy_score(y_test, model.predict(X_test)))
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

