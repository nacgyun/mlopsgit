# ml/train.py
import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 컨테이너/헤드리스 환경에서 필수
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# ==== 파라미터 (환경변수) ====
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 5))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")
RUN_NAME = os.getenv("GIT_SHA", "")[:12] or "run"


def ensure_experiment_id(name: str, client: MlflowClient, retries: int = 10, sleep: float = 0.5) -> str:
    """
    MLflow 서버가 'create → 즉시 get'에서 가끔 지연을 보일 수 있어
    안전하게 생성/조회(retry)하는 헬퍼.
    """
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id

    exp_id = client.create_experiment(name)
    # id로 조회 재시도
    for _ in range(retries):
        try:
            got = client.get_experiment(exp_id)
            if got is not None:
                return exp_id
        except Exception:
            pass
        time.sleep(sleep)
    # 이름으로 재시도
    for _ in range(retries):
        exp = client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        time.sleep(sleep)
    raise RuntimeError(f"Failed to ensure experiment '{name}' exists")


def main():
    # ==== MLflow 설정 ====
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # ==== 데이터 준비 ====
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # ==== 학습 ====
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42
    )

    with mlflow.start_run(experiment_id=exp_id, run_name=RUN_NAME):
        # 파라미터 로깅
        mlflow.log_params({
            "N_ESTIMATORS": N_ESTIMATORS,
            "MAX_DEPTH": MAX_DEPTH,
            "framework": "sklearn",
            "dataset": "iris",
        })

        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        # 성능 평가
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # 메트릭 로깅
        mlflow.log_metric("train_time_sec", float(train_time))
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_score", float(f1))

        # Confusion Matrix 저장 → artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        # seaborn 없이 matplotlib로 그려도 충분 (외부 의존 최소화)
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar(im)
        tick_marks = np.arange(len(iris.target_names))
        plt.xticks(tick_marks, iris.target_names, rotation=45)
        plt.yticks(tick_marks, iris.target_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path)

        # 입력 예시/시그니처 보조 (선택)
        input_example = X_test[:2].tolist()
        with open("input_example.json", "w") as f:
            json.dump(input_example, f)
        mlflow.log_artifact("input_example.json")

        # 모델 로깅
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            input_example=X_test[:2],  # 시그니처 추론 보조
        )

        print(f"✅ Train done: acc={acc:.3f}, f1={f1:.3f}, time={train_time:.2f}s")


if __name__ == "__main__":
    main()

