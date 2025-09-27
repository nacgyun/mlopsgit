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
from sklearn.linear_model import LogisticRegression  # ← 간단/가벼운 모델

# ===== 파라미터 / 설정 =====
EXP_NAME  = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-logreg")
RUN_NAME  = (os.getenv("GIT_SHA", "")[:12] or "run")
LR_C      = float(os.getenv("LR_C", "0.7"))
LR_MAX_IT = int(os.getenv("LR_MAX_ITER", "200"))

# ===== 공통 유틸 =====
def retry(fn, retries=12, delay=0.3, backoff=1.5, retry_on=lambda e: True):
    last = None
    for _ in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if not retry_on(e):
                raise
            time.sleep(delay); delay *= backoff
    if last:
        raise last

def _is_run_not_found(e: Exception) -> bool:
    return isinstance(e, RestException) and ("Run with id" in str(e) and "not found" in str(e))

def _is_exp_id_missing(e: Exception) -> bool:
    return isinstance(e, RestException) and ("No Experiment with id" in str(e) or "RESOURCE_DOES_NOT_EXIST" in str(e))

def ensure_experiment_id(name: str, client: MlflowClient, retries: int = 8, sleep: float = 0.25) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        try:
            client.create_experiment(name)
        except RestException as e:
            msg = str(e)
            if "RESOURCE_ALREADY_EXISTS" not in msg and "UNIQUE constraint failed" not in msg:
                raise
        # 재조회
        for i in range(retries):
            exp = client.get_experiment_by_name(name)
            if exp is not None:
                break
            time.sleep(sleep * (1.5 ** i))
        if exp is None:
            raise RuntimeError(f"Failed to ensure experiment '{name}' exists")
    exp_id = exp.experiment_id
    # id 유효성 검증
    for i in range(retries):
        try:
            ok = client.get_experiment(exp_id)
            if ok is not None:
                return exp_id
        except RestException:
            pass
        time.sleep(sleep * (1.5 ** i))
    # 경합/캐시 이슈 시 이름 재조회
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    raise RuntimeError(f"Experiment id for '{name}' could not be validated")

def create_run_no_visibility_race(client: MlflowClient, exp_name: str, exp_id: str, run_name: str):
    def _try_create(eid: str):
        return client.create_run(
            experiment_id=eid,
            tags={"mlflow.runName": run_name, "source": "k8s-train-job"},
        )
    try:
        return _try_create(exp_id)
    except RestException as e:
        if not _is_exp_id_missing(e):
            raise
    for i in range(6):
        fresh_id = ensure_experiment_id(exp_name, client)
        try:
            return _try_create(fresh_id)
        except RestException as e:
            if _is_exp_id_missing(e):
                time.sleep(0.25 * (2 ** i))
                continue
            raise
    raise RuntimeError("Failed to create run: experiment id kept missing")

def wait_run_visible(client: MlflowClient, run_id: str, retries: int = 40, sleep: float = 0.25):
    for i in range(retries):
        try:
            r = client.get_run(run_id)
            if r and r.info and r.info.run_id == run_id:
                return
        except RestException:
            pass
        time.sleep(sleep * (1.3 ** i))
    raise RuntimeError(f"Run {run_id} not visible after retries")

def log_params_safe(client: MlflowClient, run_id: str, params: dict):
    for k, v in params.items():
        retry(lambda: client.log_param(run_id, k, str(v)))

def log_metric_safe(client: MlflowClient, run_id: str, key: str, value: float, step: int | None = None):
    retry(
        lambda: client.log_metric(run_id, key, float(value), step=step if step is not None else 0),
        retries=50, delay=0.2, backoff=1.2,
        retry_on=lambda e: _is_run_not_found(e) or isinstance(e, RestException)
    )

def log_artifact_safe(client: MlflowClient, run_id: str, local_path: str, artifact_path: str | None = None):
    retry(lambda: client.log_artifact(run_id, local_path, artifact_path))

def log_artifacts_safe(client: MlflowClient, run_id: str, local_dir: str, artifact_path: str | None = None):
    retry(lambda: client.log_artifacts(run_id, local_dir, artifact_path))

def list_artifacts_with_retry(client, run_id, path="", retries=20, delay=0.3):
    last = None
    for _ in range(retries):
        try:
            return client.list_artifacts(run_id, path)
        except Exception as e:
            last = e
            time.sleep(delay); delay *= 1.3
    if last:
        raise last

def print_artifacts_tree(client, run_id, base_path=""):
    items = list_artifacts_with_retry(client, run_id, base_path)
    print("[artifacts]", base_path or ".", "->", [x.path for x in items])

def main():
    # ===== MLflow 연결 =====
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("TRACKING_URI")
        or mlmod.get_tracking_uri()
    )
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # ===== experiment_id 확보 =====
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # ===== 데이터 (간단) =====
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # ===== run 생성 + 가시성 대기 =====
    run = create_run_no_visibility_race(client, EXP_NAME, exp_id, RUN_NAME)
    run_id = run.info.run_id
    wait_run_visible(client, run_id)

    try:
        # 파라미터 기록
        log_params_safe(client, run_id, {
            "model": "LogisticRegression",
            "LR_C": LR_C,
            "LR_MAX_ITER": LR_MAX_IT,
            "dataset": "iris",
        })

        # ===== 학습 (간단/가벼움, 1.0 점수 방지) =====
        start = time.time()
        clf = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            C=LR_C, max_iter=LR_MAX_IT, random_state=42
        )
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        # 메트릭
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        log_metric_safe(client, run_id, "train_time_sec", train_time)
        log_metric_safe(client, run_id, "accuracy", acc)
        log_metric_safe(client, run_id, "f1_score", f1)

        # ===== Confusion Matrix → artifact =====
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
        plt.savefig(cm_path, bbox_inches="tight"); plt.close()
        log_artifact_safe(client, run_id, cm_path, artifact_path="plots")

        # ===== 모델 업로드: log_model (서버가 S3로 저장) =====
        import mlflow.sklearn as ml_sklearn
        ml_sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            input_example=X_test[:2]
        )

        # (선택) 입력 예시 JSON
        with open("input_example.json", "w") as f:
            json.dump(X_test[:2].tolist(), f)
        log_artifact_safe(client, run_id, "input_example.json")

        # 업로드 확인(리스트)
        print_artifacts_tree(client, run_id)           # 루트
        print_artifacts_tree(client, run_id, "plots")  # 이미지
        print_artifacts_tree(client, run_id, "model")  # 모델

        print(f"✅ Train done: acc={acc:.3f}, f1={f1:.3f}, time={train_time:.2f}s")

    finally:
        retry(lambda: client.set_terminated(run_id, "FINISHED"),
              retries=30, delay=0.2, backoff=1.2,
              retry_on=lambda e: _is_run_not_found(e) or isinstance(e, RestException))

if __name__ == "__main__":
    main()

