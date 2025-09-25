import os, time, json, tempfile, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ===== 파라미터 / 설정 =====
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH    = int(os.environ.get("MAX_DEPTH", 5))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())
EXP_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")
RUN_NAME     = (os.getenv("GIT_SHA", "")[:12] or "run")

# ===== 공통 유틸 =====
def retry(fn, retries=12, delay=0.3, backoff=1.5, swallow=None):
    last = None
    for _ in range(retries):
        try:
            return fn()
        except Exception as e:
            if swallow and any(s in str(e) for s in swallow):
                # 삼켜야 할 에러면 성공 취급(바로 return)하지 말고 다음 단계에서 확인
                last = e
            else:
                last = e
            time.sleep(delay); delay *= backoff
    if last:
        raise last

def ensure_experiment_id(name: str, client: MlflowClient, retries: int = 18, sleep: float = 0.4) -> str:
    # 1) 이름으로 여러 번 조회
    for _ in range(retries):
        exp = client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        time.sleep(sleep)
    # 2) 생성 시도(이미 존재는 무시)
    try:
        client.create_experiment(name)
    except RestException as e:
        if "RESOURCE_ALREADY_EXISTS" not in str(e) and "UNIQUE constraint failed" not in str(e):
            raise
    except Exception:
        pass
    # 3) 다시 이름으로 조회될 때까지 대기
    for _ in range(retries):
        exp = client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        time.sleep(sleep)
    raise RuntimeError(f"Failed to ensure experiment '{name}' exists")

def create_run_with_retry(client: MlflowClient, exp_name: str, run_name: str, retries=18, sleep=0.4):
    """
    실험이 방금 생겨서 ID가 아직 인덱싱 안된 케이스까지 고려:
    - 매번 최신 exp_id를 이름으로 다시 가져와서 create_run 시도
    - RESOURCE_DOES_NOT_EXIST 오류면 잠깐 대기 후 재시도
    """
    for _ in range(retries):
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            time.sleep(sleep)
            continue
        try:
            run = client.create_run(
                experiment_id=exp.experiment_id,
                tags={
                    "mlflow.runName": run_name,
                    "source": "k8s-train-job",
                },
            )
            return run
        except RestException as e:
            # 실험 ID를 아직 서버가 못 알아볼 때
            if "RESOURCE_DOES_NOT_EXIST" in str(e) or "No Experiment with id" in str(e):
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError("Failed to create run due to experiment visibility delay")

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
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # ===== 실험 보장 (이름 기준) =====
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # ===== 데이터 =====
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # ===== 러닝 개체 직접 생성 (Fluent 미사용) =====
    run = create_run_with_retry(client, EXP_NAME, RUN_NAME)
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

        # === 모델 저장 → artifacts 업로드 (Fluent log_model 사용 안 함) ===
        tmpdir = tempfile.mkdtemp(prefix="model_")
        try:
            local_model_dir = os.path.join(tmpdir, "model")
            import mlflow.sklearn
            mlflow.sklearn.save_model(
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

