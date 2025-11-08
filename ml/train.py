# ml/train.py
# -*- coding: utf-8 -*-
"""
Telco Churn(MLflow + MinIO/S3)와 Iris를 ENV로 스위치하여 학습하는 학습 스크립트.

ENV 스위치 요약
- DATASET=telco  (기본: iris)
- TELCO_CSV_URI=s3://data/telco/Telco-Customer-Churn.csv  # MinIO/S3 경로
- MLFLOW_EXPERIMENT_NAME=telco-churn  # 선택
- REGISTER_MODEL_NAME=ChurnModel      # 선택: 레지스트리에 등록
- MODEL_STAGE=Staging|Production      # 선택: 등록 후 스테이지 전환

MinIO/S3 접속 ENV (이미 클러스터에 있음)
- MLFLOW_S3_ENDPOINT_URL=http://<minio-nodeip:nodeport>
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
- AWS_S3_ADDRESSING_STYLE=path, AWS_S3_FORCE_PATH_STYLE=true (권장)

필요 패키지: s3fs (pandas가 s3://를 읽기 위함)
"""
import os
import time
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow as mlmod
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models import infer_signature

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import load_iris

# 🔹 로그 함수 모듈
from logs import log_json_line, log_ghcr_metadata_to_mlflow

# ===== 파라미터 / 설정 =====
EXP_NAME        = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")
RUN_NAME        = (os.getenv("GIT_SHA", "")[:12] or "run")

# 기본 러닝타임 타깃(초)
TARGET_WALL_SEC = float(os.getenv("TARGET_WALL_SEC", "180"))

EPOCHS          = int(os.getenv("MLFLOW_EPOCHS", "40"))
BATCH_SIZE      = int(os.getenv("TRAIN_BATCH_SIZE", "64"))
SLEEP_SEC       = float(os.getenv("TRAIN_SLEEP_SEC", "0.0"))

# 🔸 학습 연산량 제어 파라미터 (Iris 증분학습 전용)
LOOPS_PER_EPOCH = int(os.getenv("LOOPS_PER_EPOCH", "3"))
AUGMENT_ENABLE  = os.getenv("AUGMENT_ENABLE", "1") == "1"
AUGMENT_COPIES  = int(os.getenv("AUGMENT_COPIES", "3"))
AUGMENT_NOISE   = float(os.getenv("AUGMENT_NOISE", "0.08"))

# 🔸 burn(추가 연산) — Iris 증분학습 전용
DEFAULT_BURN_PASSES = "1200"
BURN_PASSES     = int(os.getenv("BURN_PASSES", DEFAULT_BURN_PASSES))
BURN_NOISE      = float(os.getenv("BURN_NOISE", "0.08"))
BURN_ENABLE     = os.getenv("BURN_ENABLE", "1") == "1"
BURN_CHUNK_PASSES = int(os.getenv("BURN_CHUNK_PASSES", "256"))

# SGD 하이퍼파라미터
LR_ALPHA        = float(os.getenv("LR_ALPHA", "0.0005"))
LR_INITIAL      = float(os.getenv("LR_INITIAL", "0.01"))
RANDOM_STATE    = int(os.getenv("SEED", "42"))
EMA_ALPHA       = float(os.getenv("ETA_EMA_ALPHA", "0.2"))

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


def extra_training_burn(template_clf, X, y, passes, batch_size, noise, seed):
    if passes <= 0:
        return
    rng = np.random.default_rng(seed)
    shadow = clone(template_clf)
    uclasses = np.unique(y)
    if len(X) >= batch_size:
        shadow.partial_fit(X[:batch_size], y[:batch_size], classes=uclasses)
    else:
        shadow.partial_fit(X, y, classes=uclasses)
    n = len(X)
    for _ in range(passes):
        idx = rng.integers(0, n, size=min(batch_size, n))
        Xb = X[idx] + rng.normal(0.0, noise, size=X[idx].shape)
        shadow.partial_fit(Xb, y[idx])


def spend_time_to_target(template_clf, X, y, batch_size, noise, seed, target_deadline_sec):
    if not BURN_ENABLE:
        return
    start = time.perf_counter()
    rng_seed = seed
    while time.perf_counter() - start < target_deadline_sec:
        extra_training_burn(
            template_clf=template_clf,
            X=X, y=y,
            passes=BURN_CHUNK_PASSES,
            batch_size=batch_size,
            noise=noise,
            seed=rng_seed
        )
        rng_seed += 1

# ========= Telco Churn 로더 =========

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
    # path-style 강제
    if force_path or addressing == "path":
        opts.setdefault("client_kwargs", {})["config_kwargs"] = {"s3": {"addressing_style": "path"}}
    return opts


def load_telco_churn(csv_uri: str):
    """Telco Churn CSV를 읽고 (X,y), 파이프라인을 반환한다."""
    storage_options = _s3_storage_options_from_env() if csv_uri.startswith("s3://") else None
    df = pd.read_csv(csv_uri, storage_options=storage_options)

    # 기본 정제
    for c in ("customerID", "CustomerID", "customerId"):
        if c in df.columns:
            df = df.drop(columns=[c])
            break
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=[col for col in ["Churn", "TotalCharges"] if col in df.columns])

    # 타깃
    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn"])

    # 컬럼 타입 분리
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
        max_iter=1,  # fit를 여러 epoch 반복 호출
        warm_start=False,
        tol=None,
    )

    pipe = Pipeline(steps=[("preproc", preproc), ("clf", clf)])
    return X, y, pipe, cat_cols, num_cols


# ========= 메인 =========

def main():
    wall_start = time.perf_counter()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 데이터셋 스위치
    DATASET = os.getenv("DATASET", "iris").lower()

    if DATASET in ("telco", "telco-churn", "churn"):
        TELCO_CSV_URI = os.getenv("TELCO_CSV_URI", "s3://data/telco/Telco-Customer-Churn.csv")
        X_all, y_all, model, cat_cols, num_cols = load_telco_churn(TELCO_CSV_URI)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.30, random_state=0, stratify=y_all
        )
        classes = np.unique(y_train)
        target_names = ["stay", "churn"]
        is_telco = True
        exp_name_local = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn")
    else:
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.30, random_state=0
        )
        classes = np.unique(y_train)
        target_names = iris.target_names.tolist()
        is_telco = False
        # Iris용 모델(증분학습)
        model = SGDClassifier(
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
        exp_name_local = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")

    # EXP_NAME 전역 업데이트
    global EXP_NAME
    EXP_NAME = exp_name_local

    # 실험 보장
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # 증강(Iris 전용)
    if not is_telco and AUGMENT_ENABLE and AUGMENT_COPIES > 0:
        rng = np.random.default_rng(RANDOM_STATE + 777)
        X_aug_list = [X_train]; y_aug_list = [y_train]
        for _ in range(AUGMENT_COPIES):
            noise = rng.normal(0.0, AUGMENT_NOISE, size=X_train.shape)
            X_aug_list.append(X_train + noise)
            y_aug_list.append(y_train)
        X_train = np.vstack(X_aug_list)
        y_train = np.hstack(y_aug_list)

    with start_run_with_retry(exp_id, RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}, exp_id={exp_id}")

        # === GHCR 메타데이터 기록 ===
        log_ghcr_metadata_to_mlflow()

        mlmod.log_params({
            "dataset": DATASET,
            "model": "SGDClassifier(logistic)",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "sleep_sec": SLEEP_SEC,
            "loops_per_epoch": LOOPS_PER_EPOCH,
            "burn_passes": BURN_PASSES,
            "augment": AUGMENT_ENABLE if not is_telco else False,
            "augment_copies": AUGMENT_COPIES if not is_telco else 0,
            "augment_noise": AUGMENT_NOISE if not is_telco else 0.0,
            "target_wall_sec": TARGET_WALL_SEC,
            "burn_chunk_passes": BURN_CHUNK_PASSES,
        })

        f1_hist = []
        ema = None

        # Iris: 증분학습 초기 클래스 지정
        if not is_telco:
            model.partial_fit(X_train[:BATCH_SIZE], y_train[:BATCH_SIZE], classes=classes)

        for epoch in range(1, EPOCHS + 1):
            t_epoch = time.perf_counter()

            if is_telco:
                # 파이프라인(전처리+SGD) 전체 fit — 빠르므로 반복 가능
                model.fit(X_train, y_train)
            else:
                # 기존 Iris 증분 루프
                for loop in range(LOOPS_PER_EPOCH):
                    for Xb, yb in batch_iter(
                        X_train, y_train, BATCH_SIZE, shuffle=True,
                        seed=RANDOM_STATE + epoch * 1000 + loop
                    ):
                        model.partial_fit(Xb, yb)

                if BURN_ENABLE and BURN_PASSES > 0:
                    extra_training_burn(
                        template_clf=model,
                        X=X_train, y=y_train,
                        passes=BURN_PASSES,
                        batch_size=BATCH_SIZE,
                        noise=BURN_NOISE,
                        seed=RANDOM_STATE + 1000 + epoch,
                    )

            compute_sec = time.perf_counter() - t_epoch

            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1  = float(f1_score(y_test, y_pred, average="macro"))
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                else:
                    y_proba = None
                ll = float(log_loss(y_test, y_proba)) if y_proba is not None else float("nan")
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
                "epoch_sleep_sec": SLEEP_SEC,
                "epoch_time_sec": compute_sec + SLEEP_SEC,
                "eta_sec": eta_sec,
                "progress_pct": min(99.9, 100.0 * epoch / EPOCHS),
                "elapsed_sec": elapsed,
            }, step=epoch)

            f1_hist.append(f1)
            print(f"[epoch {epoch:03d}] acc={acc:.4f} f1={f1:.4f} comp={compute_sec:.2f}s "
                  f"sleep={SLEEP_SEC:.2f}s elapsed={elapsed:.1f}s ETA~{eta_sec:.1f}s")
            log_json_line({
                "event": "epoch_metric",
                "epoch": epoch,
                "accuracy": acc,
                "duration": round(compute_sec + SLEEP_SEC, 4),
                "remaining_sec": round(eta_sec, 1),
                "run_id": run_id,
                "experiment": EXP_NAME,
            })

            if SLEEP_SEC > 0:
                time.sleep(SLEEP_SEC)

            if not is_telco:
                remaining_to_target = TARGET_WALL_SEC - (time.perf_counter() - wall_start)
                if BURN_ENABLE and remaining_to_target > 5.0:
                    per_epoch_cap = float(os.getenv("PER_EPOCH_SPEND_CAP_SEC", "60"))
                    to_spend = min(per_epoch_cap, max(0.0, remaining_to_target * 0.4))
                    if to_spend > 1.0:
                        spend_time_to_target(
                            template_clf=model,
                            X=X_train, y=y_train,
                            batch_size=BATCH_SIZE,
                            noise=BURN_NOISE,
                            seed=RANDOM_STATE + 2000 + epoch,
                            target_deadline_sec=to_spend,
                        )

            if time.perf_counter() - wall_start >= TARGET_WALL_SEC:
                print(f"[info] target wall time ({TARGET_WALL_SEC:.0f}s) reached. stopping early.")
                break

        total_time = time.perf_counter() - wall_start
        mlmod.log_metric("train_time_total_sec", total_time)

        # Confusion Matrix (공용)
        cm = confusion_matrix(y_test, model.predict(X_test))
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar(im)
        tick = np.arange(len(target_names))
        plt.xticks(tick, target_names, rotation=45)
        plt.yticks(tick, target_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig("confusion_matrix.png", bbox_inches="tight")
        mlmod.log_artifact("confusion_matrix.png", artifact_path="plots")

        # F1 Learning Curve
        plt.figure(figsize=(6, 3.5))
        plt.plot(range(1, len(f1_hist)+1), f1_hist, marker="o")
        plt.title("F1 over Epochs")
        plt.xlabel("Epoch"); plt.ylabel("F1 (macro)"); plt.grid(True, alpha=0.3)
        plt.savefig("learning_curve_f1.png", bbox_inches="tight")
        mlmod.log_artifact("learning_curve_f1.png", artifact_path="plots")

        # 모델 로깅 (스키마/예시 포함)
        from mlflow import sklearn as ml_sklearn
        # Telco는 DataFrame, Iris는 ndarray → 예시를 통일 처리
        if hasattr(X_train, "head"):
            input_example = X_train.head(2)
            sig = infer_signature(X_train, model.predict(X_train))
        else:
            input_example = X_train[:2]
            sig = infer_signature(X_train, model.predict(X_train))

        ml_sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=sig,
            input_example=input_example,
        )

        with open("input_example.json", "w") as f:
            if hasattr(input_example, "to_dict"):
                json.dump(input_example.to_dict(orient="records"), f)
            else:
                json.dump(np.asarray(input_example).tolist(), f)
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

        # (옵션) 모델 레지스트리에 등록/스테이지 전환
        reg_name = os.getenv("REGISTER_MODEL_NAME", "").strip()
        if reg_name:
            try:
                # 등록 모델 없으면 생성 시도
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

