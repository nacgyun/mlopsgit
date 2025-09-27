# ml/train.py
import os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GitPython 경고 억제
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow as mlmod
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# ===== 파라미터 / 설정 =====
EXP_NAME  = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")  # env가 우선
RUN_NAME  = (os.getenv("GIT_SHA", "")[:12] or "run")
LR_C      = float(os.getenv("LR_C", "0.3"))     # 1.0 방지 위해 규제 강화
LR_MAX_IT = int(os.getenv("LR_MAX_ITER", "200"))

# ===== 공통 유틸 =====
def ensure_experiment_id(name: str, client: MlflowClient, retries: int = 20, sleep: float = 0.25) -> str:
    """
    1) 이름으로 조회해서 있으면 그 id 사용
    2) 없으면 생성 후, id 가시성(get_experiment) 확인까지 대기
    """
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
            # 이름 가시성 재시도
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

    # id 가시성 확인 (이름 가시성보다 신뢰)
    for i in range(retries):
        try:
            if client.get_experiment(exp_id) is not None:
                return exp_id
        except RestException:
            pass
        time.sleep(sleep * (1.5 ** i))

    # 마지막 안전망: 이름 재조회
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    raise RuntimeError(f"Experiment id for '{name}' could not be validated")

def start_run_with_retry(exp_id: str, run_name: str, retries: int = 12, delay: float = 0.3, backoff: float = 1.5):
    """
    가끔 REST 가시성 지연으로 'No Experiment with id'가 날 수 있어 짧게 재시도.
    """
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

def main():
    # ===== MLflow 연결 =====
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)

    # Client로 실험 ID 확보(가시성까지 확인)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # ===== 데이터 (1.0 방지: 테스트 30%, 시드 고정) =====
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.30, random_state=0
    )

    # ===== Run 시작: experiment_id를 명시하고 Fluent 컨텍스트 사용 =====
    with start_run_with_retry(exp_id, RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}, exp_id={exp_id}")

        # 파라미터
        mlmod.log_params({
            "model": "LogisticRegression",
            "LR_C": LR_C,
            "LR_MAX_ITER": LR_MAX_IT,
            "dataset": "iris",
            "test_size": 0.30,
            "random_state": 0,
        })

        # ===== 학습 =====
        t0 = time.time()
        clf = LogisticRegression(
            solver="lbfgs",
            C=LR_C,
            max_iter=LR_MAX_IT,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        # 메트릭
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1  = float(f1_score(y_test, y_pred, average="macro"))
        mlmod.log_metrics({
            "train_time_sec": train_time,
            "accuracy": acc,
            "f1_score": f1,
        })

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
        mlmod.log_artifact(cm_path, artifact_path="plots")

        # ===== 모델 업로드: Fluent log_model (동일 컨텍스트) =====
        from mlflow import sklearn as ml_sklearn
        signature = infer_signature(X_train, clf.predict(X_train))
        ml_sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:2],
        )

        # (선택) 입력 예시 JSON
        with open("input_example.json", "w") as f:
            json.dump(X_test[:2].tolist(), f)
        mlmod.log_artifact("input_example.json")

        print(f"✅ Train done: acc={acc:.3f}, f1={f1:.3f}, time={train_time:.2f}s")

if __name__ == "__main__":
    main()

