# ml/train.py
# Telco Churn (MinIO/S3) 전용 학습 스크립트
import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# --- ML/Sklearn ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.linear_model import SGDClassifier

# --- MLflow ---
import mlflow as mlmod
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models import infer_signature
from mlflow import sklearn as ml_sklearn

# --- 로깅 유틸(네가 이미 쓰던 모듈) ---
from logs import log_json_line, log_ghcr_metadata_to_mlflow

# =========================
# 환경변수 / 기본값
# =========================
TELCO_CSV_URI        = os.getenv("TELCO_CSV_URI", "s3://data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv")
EXP_NAME             = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn")
RUN_NAME             = (os.getenv("GIT_SHA", "")[:12] or "run")
REGISTER_MODEL_NAME  = os.getenv("REGISTER_MODEL_NAME", "ChurnModel")
MODEL_STAGE          = os.getenv("MODEL_STAGE", "Production")   # e.g., "Production" / "Staging" / "None"

TARGET_WALL_SEC      = float(os.getenv("TARGET_WALL_SEC", "180"))
RANDOM_STATE         = int(os.getenv("SEED", "42"))

# SGD 하이퍼파라미터(얘는 경량 + 빠른 수렴용)
LR_ALPHA             = float(os.getenv("LR_ALPHA", "0.0005"))
BATCH_SIZE           = int(os.getenv("TRAIN_BATCH_SIZE", "256"))
EPOCHS               = int(os.getenv("MLFLOW_EPOCHS", "1"))  # 파이프라인 학습이라 1 epoch면 충분
SLEEP_SEC            = float(os.getenv("TRAIN_SLEEP_SEC", "0.0"))

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

    # 확인 루프
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

# =========================
# 데이터 로드/전처리
# =========================
def load_telco_churn(csv_uri: str):
    """
    MinIO/S3 경로에서 Telco CSV 로드 + 전처리 컬럼 구분 반환
    - 타깃: 'Churn' (Yes/No) -> 1/0
    - 수치형: 'tenure','MonthlyCharges','TotalCharges'(문자→숫자)
    - 범주형: 나머지 object 타입(Churn 제외)
    """
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    access   = os.getenv("AWS_ACCESS_KEY_ID")
    secret   = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not endpoint or not access or not secret:
        raise RuntimeError("Missing S3 credentials/env (MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)")

    storage_options = {
        "client_kwargs": {"endpoint_url": endpoint},
        "key": access,
        "secret": secret
    }

    df = pd.read_csv(csv_uri, storage_options=storage_options)

    # 표준 Telco 칼럼 처리
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 타깃
    if "Churn" not in df.columns:
        raise ValueError("CSV에 'Churn' 칼럼이 없습니다.")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # TotalCharges 정리(공백 → NaN → 숫자)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    y = df["Churn"].values
    X = df.drop(columns=["Churn"])

    # 수치형/범주형 구분
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if pd.api.types.is_object_dtype(X[c])]

    return X, y, cat_cols, num_cols

def build_pipeline(cat_cols, num_cols):
    cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_tf = StandardScaler()

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ],
        remainder="drop"
    )

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=LR_ALPHA,
        random_state=RANDOM_STATE,
        max_iter=1000,
        tol=1e-3
    )

    pipe = Pipeline(steps=[
        ("preproc", preproc),
        ("clf", clf),
    ])
    return pipe

# =========================
# 메인
# =========================
def main():
    wall_start = time.perf_counter()

    # MLflow 설정
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # 데이터 로드
    X, y, cat_cols, num_cols = load_telco_churn(TELCO_CSV_URI)

    # 학습/검증 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 파이프라인
    pipe = build_pipeline(cat_cols, num_cols)

    with start_run_with_retry(exp_id, RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}, exp_id={exp_id}")

        # GHCR 메타데이터(이미지/sha 등) 기록
        log_ghcr_metadata_to_mlflow()

        # 파라미터 로깅
        mlmod.log_params({
            "dataset": "Telco-Customer-Churn (MinIO/S3)",
            "TELCO_CSV_URI": TELCO_CSV_URI,
            "model": "SGDClassifier(logistic)",
            "alpha": LR_ALPHA,
            "random_state": RANDOM_STATE,
            "batch_size_hint": BATCH_SIZE,
            "epochs": EPOCHS,
        })

        # 학습
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        mlmod.log_metric("train_time_sec", float(train_time))

        # 평가
        y_pred = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1  = float(f1_score(y_test, y_pred, average="macro"))
        try:
            proba = pipe.predict_proba(X_test)
            ll = float(log_loss(y_test, proba))
        except Exception:
            ll = float("nan")

        mlmod.log_metrics({
            "accuracy": acc,
            "f1_macro": f1,
            "log_loss": ll
        })

        log_json_line({
            "event": "eval",
            "accuracy": acc,
            "f1_macro": f1,
            "log_loss": ll,
            "run_id": run_id,
            "experiment": EXP_NAME,
        })

        # 혼동행렬 아티팩트
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5.2, 4.2))
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar(im)
        tick = np.arange(2)
        plt.xticks(tick, ["No","Yes"])
        plt.yticks(tick, ["No","Yes"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig("confusion_matrix.png", bbox_inches="tight")
        mlmod.log_artifact("confusion_matrix.png", artifact_path="plots")

        # 러닝 커브(여기서는 단일 fit라 간단한 점만)
        plt.figure(figsize=(6,3.5))
        plt.plot([1], [f1], marker="o")
        plt.title("F1 (macro)")
        plt.xlabel("epoch"); plt.ylabel("F1")
        plt.grid(True, alpha=0.3)
        plt.savefig("learning_curve_f1.png", bbox_inches="tight")
        mlmod.log_artifact("learning_curve_f1.png", artifact_path="plots")

        # 모델 로깅 + 시그니처
        # input_example: 전처리 전 raw feature의 일부
        input_example = X_test.iloc[:2].copy()
        # 예측용 시그니처는 전처리 뒤 모양이지만, raw기반 시그니처 추론을 위해 predict 결과 사용
        signature = infer_signature(X_test, pipe.predict(X_test))

        ml_sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTER_MODEL_NAME  # 바로 등록
        )

        # 최신 모델 버전 조회 후 스테이지 전환
        try:
            latest = client.get_latest_versions(name=REGISTER_MODEL_NAME, stages=["None", "Staging", "Production"])
            if latest:
                # 가장 최근 버전 선택(등록 시각 기준으로 정렬)
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

        total_time = time.perf_counter() - wall_start
        mlmod.log_metric("total_wall_time_sec", float(total_time))
        print(f"[PROMOTE] accuracy={acc:.5f}", flush=True)
        print("✅ Train done.", flush=True)

if __name__ == "__main__":
    main()

