# ml/train.py
import os, time, json, tempfile, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow as mlmod
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression  # 간단/가벼운 모델

# ===== 파라미터 / 설정 =====
EXP_NAME  = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")  # env가 우선
RUN_NAME  = (os.getenv("GIT_SHA", "")[:12] or "run")
LR_C      = float(os.getenv("LR_C", "0.3"))     # 1.0 방지 위해 규제 강화
LR_MAX_IT = int(os.getenv("LR_MAX_ITER", "200"))

def main():
    # ===== MLflow 연결 & 실험 설정(Fluent 전용) =====
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("TRACKING_URI")
        or mlmod.get_tracking_uri()
    )
    mlmod.set_tracking_uri(tracking_uri)
    mlmod.set_experiment(EXP_NAME)  # 없으면 생성, 있으면 선택

    # ===== 데이터 (1.0 방지: 테스트 30%, 시드 고정) =====
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.30, random_state=0
    )

    # ===== Run 시작 (컨텍스트 내부에서 모든 로깅 수행) =====
    with mlmod.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}")

        # 파라미터 기록
        mlmod.log_params({
            "model": "LogisticRegression",
            "LR_C": LR_C,
            "LR_MAX_ITER": LR_MAX_IT,
            "dataset": "iris",
            "test_size": 0.30,
            "random_state": 0,
        })

        # ===== 학습 =====
        start = time.time()
        clf = LogisticRegression(
            solver="lbfgs",
            C=LR_C,
            max_iter=LR_MAX_IT,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        # 메트릭
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        mlmod.log_metric("train_time_sec", float(train_time))
        mlmod.log_metric("accuracy", float(acc))
        mlmod.log_metric("f1_score", float(f1))

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

