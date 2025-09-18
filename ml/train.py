import os
import time
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ==== 파라미터 (환경변수 기반) ====
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 5))

# ==== 데이터 준비 ====
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# ==== MLflow Tracking 설정 ====
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri()))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf"))

with mlflow.start_run():
    # 파라미터 로깅
    mlflow.log_param("N_ESTIMATORS", N_ESTIMATORS)
    mlflow.log_param("MAX_DEPTH", MAX_DEPTH)

    # 모델 학습
    start = time.time()
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    # 성능 평가
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # 메트릭 로깅
    mlflow.log_metric("train_time_sec", train_time)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Confusion Matrix 시각화 → artifact 저장
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # 모델 로깅
    mlflow.sklearn.log_model(clf, "model")

    print(f"✅ Train done: acc={acc:.3f}, f1={f1:.3f}, time={train_time:.2f}s")

