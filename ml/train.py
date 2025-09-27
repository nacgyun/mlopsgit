# ml/train.py
import os, time, json, tempfile, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow as mlmod
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ===== 파라미터 / 설정 =====
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH    = int(os.environ.get("MAX_DEPTH", 5))
EXP_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")
RUN_NAME     = (os.getenv("GIT_SHA", "")[:12] or "run")

# ===== 공통 유틸 =====
def retry(fn, retries=12, delay=0.3, backoff=1.5):
    last = None
    for _ in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(delay)
            delay *= backoff
    if last:
        raise last

def ensure_experiment_id(name: str, client: MlflowClient, retries: int = 8, sleep: float = 0.25) -> str:
    """이름 조회 레이스를 최소화해서 experiment_id를 확보한다."""
    # 1) 이미 있으면 바로 사용
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id

    # 2) 없으면 생성 시도
    try:
        exp_id = client.create_experiment(name)
        return exp_id
    except RestException as e:
        # 동시 생성 경합 등
        msg = str(e)
        if "RESOURCE_ALREADY_EXISTS" not in msg and "UNIQUE constraint failed" not in msg:
            raise
    except Exception:
        # 기타 경합 가능 → 아래 재시도 루프에서 조회
        pass

    # 3) 짧게 조회 재시도하여 id 확보
    for i in range(retries):
        exp = client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        time.sleep(sleep * (1.5 ** i))

    raise RuntimeError(f"Failed to ensure experiment '{name}' exists")

def create_run_no_visibility_race(client: MlflowClient, exp_id: str, run_name: str):
    """이름 조회 대신 experiment_id로 바로 run 생성(가시성 지연 우회)."""
    return client.create_run(
        experiment_id=exp_id,
        tags={"mlflow.runName": run_name, "source": "k8s-train-job"},
    )

def log_params_safe(client: MlflowClient, run_id: str, params: dict):
    for k, v in params.items():
        retry(lambda: client.log_param(run_id, k, str(v)))

def log_metric_safe(client: MlflowClient, run_id: str, key: str, value: float, step: int | None = None):
    retry(lambda: client.log_metric(run_id, key, float(value), step=step if step is not None else 0))

def log_artifact_safe(client: MlflowClient, run_id: str, local_path: str, artifact_path: str | None = None):
    retry(lambda: client.log_artifact(run_id, local_path, artifact_path))

def log_artifacts_safe(client: MlflowClient, run_id: str, local_dir: str, artifact_path: str | None = None):
    retry(lambda: client.log_artifacts(run_id, local_dir, artifact_path))

def set_terminated_safe(client: MlflowClient, run_id: str, status="FINISHED"):
    retry(lambda: client.set_terminated(run_id, status))

def main():
    # ===== MLflow 연결 =====
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("TRACKING_URI")
        or mlmod.get_tracking_uri()
    )
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # ===== experiment_id 확보 (이름 조회 레이스 방지) =====
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # ===== 데이터 =====
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # ===== run 생성: experiment_id로 직접 생성 =====
    run = create_run_no_visibility_race(client, exp_id, RUN_NAME)
    run_id = run.info.run_id

    try:
        # 파라미터 기록
        log_params_safe(client, run_id, {
            "N_ESTIMATORS": N_ESTIMATORS,
            "MAX_DEPTH": MAX_DEPTH,
            "framework": "sklearn",
            "dataset": "iris",
        })

        # 학습
        start = time.time()
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        # 메트릭
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        log_metric_safe(client, run_id, "train_time_sec", train_time)
        log_metric_safe(client, run_id, "accuracy", acc)
        log_metric_safe(client, run_id, "f1_score", f1)

        # Confusion Matrix → artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar(im)
        tick = np.arange(len(iris.target_names))
        plt.xticks(tick, iris.target_names, rotation=45)
        plt.yticks(tick, iris.target_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        log_artifact_safe(client, run_id, cm_path)

        # 입력 예시
        input_example = X_test[:2].tolist()
        with open("input_example.json", "w") as f:
            json.dump(input_example, f)
        log_artifact_safe(client, run_id, "input_example.json")

        # === 모델 저장 → artifacts 업로드 ===
        tmpdir = tempfile.mkdtemp(prefix="model_")
        try:
            local_model_dir = os.path.join(tmpdir, "model")
            import mlflow.sklearn as ml_sklearn
            ml_sklearn.save_model(
                sk_model=clf,
                path=local_model_dir,
                input_example=X_test[:2],
            )
            log_artifacts_safe(client, run_id, local_model_dir, artifact_path="model")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        print(f"✅ Train done: acc={acc:.3f}, f1={f1:.3f}, time={train_time:.2f}s")

    finally:
        set_terminated_safe(client, run_id, "FINISHED")

if __name__ == "__main__":
    main()

