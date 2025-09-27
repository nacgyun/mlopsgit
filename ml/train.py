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
def retry(fn, retries=12, delay=0.3, backoff=1.5, retry_on=lambda e: True):
    last = None
    for _ in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if not retry_on(e):
                raise
            time.sleep(delay)
            delay *= backoff
    if last:
        raise last

def _is_run_not_found(e: Exception) -> bool:
    return isinstance(e, RestException) and ("Run with id" in str(e) and "not found" in str(e))

def _is_exp_id_missing(e: Exception) -> bool:
    return isinstance(e, RestException) and ("No Experiment with id" in str(e) or "RESOURCE_DOES_NOT_EXIST" in str(e))

def ensure_experiment_id(name: str, client: MlflowClient, retries: int = 8, sleep: float = 0.25) -> str:
    """이름→ID 확보 후, ID가 진짜 유효한지까지 검증."""
    # 1) 이름으로 조회
    exp = client.get_experiment_by_name(name)
    if exp is None:
        # 2) 없으면 생성 시도
        try:
            exp_id = client.create_experiment(name)
        except RestException as e:
            msg = str(e)
            if "RESOURCE_ALREADY_EXISTS" not in msg and "UNIQUE constraint failed" not in msg:
                raise
            exp_id = None
        # 3) 생성/경합 이후 재조회
        for i in range(retries):
            exp = client.get_experiment_by_name(name)
            if exp is not None:
                break
            time.sleep(sleep * (1.5 ** i))
        if exp is None:
            raise RuntimeError(f"Failed to ensure experiment '{name}' exists")
    # 여기서 exp는 존재
    exp_id = exp.experiment_id

    # 4) 최종적으로 ID가 유효한지 직접 검증
    for i in range(retries):
        try:
            ok = client.get_experiment(exp_id)  # id로 직접 조회
            if ok is not None:
                return exp_id
        except RestException:
            pass
        time.sleep(sleep * (1.5 ** i))

    # 유효하지 않으면 재생성 루트로 한 번 더 시도
    try:
        client.delete_experiment(exp_id)  # 혹시 반쯤 깨진 경우 정리(실패해도 무시)
    except Exception:
        pass
    try:
        new_id = client.create_experiment(name)
        # 생성 직후 가시성 대기
        for i in range(retries):
            try:
                ok = client.get_experiment(new_id)
                if ok is not None:
                    return new_id
            except RestException:
                pass
            time.sleep(sleep * (1.5 ** i))
    except RestException:
        # 경합이면 이름 재조회
        exp = client.get_experiment_by_name(name)
        if exp:
            return exp.experiment_id
    raise RuntimeError(f"Experiment id for '{name}' could not be validated")

def create_run_no_visibility_race(client: MlflowClient, exp_name: str, exp_id: str, run_name: str):
    """exp_id로 run 생성하되, 'No Experiment with id'면 exp_id 재확보 후 재시도."""
    def _try_create(eid: str):
        return client.create_run(
            experiment_id=eid,
            tags={"mlflow.runName": run_name, "source": "k8s-train-job"},
        )

    # 최초 시도
    try:
        return _try_create(exp_id)
    except RestException as e:
        if not _is_exp_id_missing(e):
            raise

    # exp_id 유효하지 않다면 이름으로 재확보하며 재시도
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
    """Run 생성 직후 캐시/가시성 지연이 끝날 때까지 대기."""
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

def set_terminated_safe(client: MlflowClient, run_id: str, status="FINISHED"):
    retry(
        lambda: client.set_terminated(run_id, status),
        retries=30, delay=0.2, backoff=1.2,
        retry_on=lambda e: _is_run_not_found(e) or isinstance(e, RestException)
    )

def main():
    # ===== MLflow 연결 =====
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("TRACKING_URI")
        or mlmod.get_tracking_uri()
    )
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # ===== experiment_id 확보 (유효성 검증 포함) =====
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # ===== 데이터 =====
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # ===== run 생성: exp_id로 생성, 필요 시 재확보/재시도 + 가시성 대기 =====
    run = create_run_no_visibility_race(client, EXP_NAME, exp_id, RUN_NAME)
    run_id = run.info.run_id
    wait_run_visible(client, run_id)

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

