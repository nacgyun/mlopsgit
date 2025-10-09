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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.linear_model import SGDClassifier

# ===== 파라미터 / 설정 =====
EXP_NAME        = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")
RUN_NAME        = (os.getenv("GIT_SHA", "")[:12] or "run")

# 학습 길이/속도 조절용(환경변수로 변경 가능)
EPOCHS          = int(os.getenv("MLFLOW_EPOCHS", "40"))
BATCH_SIZE      = int(os.getenv("TRAIN_BATCH_SIZE", "32"))
SLEEP_SEC       = float(os.getenv("TRAIN_SLEEP_SEC", "0.25"))

# SGD 하이퍼파라미터
LR_ALPHA        = float(os.getenv("LR_ALPHA", "0.0005"))  # L2 강도
LR_INITIAL      = float(os.getenv("LR_INITIAL", "0.01"))  # (힌트용) 초기 lr
RANDOM_STATE    = int(os.getenv("SEED", "42"))

# ETA 지수이동평균 파라미터
EMA_ALPHA       = float(os.getenv("ETA_EMA_ALPHA", "0.2"))  # 0.1~0.3 권장

# ===== 공통 유틸 =====
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

def main():
    # ===== MLflow 연결 =====
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # ===== 데이터 =====
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.30, random_state=0
    )
    classes = np.unique(y_train)

    # ===== Run 시작 =====
    with start_run_with_retry(exp_id, RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}, exp_id={exp_id}")

        mlmod.log_params({
            "model": "SGDClassifier(logistic)",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "sleep_sec": SLEEP_SEC,
            "alpha": LR_ALPHA,
            "eta0_hint": LR_INITIAL,
            "dataset": "iris",
            "test_size": 0.30,
            "random_state": RANDOM_STATE,
            "eta_ema_alpha": EMA_ALPHA,
        })

        # ===== 분류기 (로지스틱 손실) =====
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=LR_ALPHA,
            learning_rate="optimal",   # eta0는 힌트
            random_state=RANDOM_STATE,
            fit_intercept=True,
            max_iter=1,                # partial_fit 루프에서 1스텝씩
            warm_start=False,
            tol=None
        )

        # 첫 partial_fit에는 classes 전달 필요(내부 초기화)
        clf.partial_fit(X_train[:BATCH_SIZE], y_train[:BATCH_SIZE], classes=classes)

        # ===== 에폭 루프 =====
        t0 = time.time()
        f1_hist = []
        ema = None  # epoch compute time EMA

        for epoch in range(1, EPOCHS + 1):
            t_epoch = time.perf_counter()

            # 미니배치 학습
            for Xb, yb in batch_iter(X_train, y_train, BATCH_SIZE, shuffle=True, seed=RANDOM_STATE + epoch):
                clf.partial_fit(Xb, yb)

            # 실제 학습 시간(대기 제외)
            compute_sec = time.perf_counter() - t_epoch

            # 평가/로그
            y_pred = clf.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1  = float(f1_score(y_test, y_pred, average="macro"))
            try:
                y_proba = clf.predict_proba(X_test)
                ll = float(log_loss(y_test, y_proba))
            except Exception:
                ll = float("nan")

            # ETA 계산(지수이동평균)
            if ema is None:
                ema = compute_sec
            else:
                ema = EMA_ALPHA * compute_sec + (1 - EMA_ALPHA) * ema
            remaining_epochs = EPOCHS - epoch
            eta_sec = max(0.0, remaining_epochs * (ema + (SLEEP_SEC if SLEEP_SEC > 0 else 0.0)))

            # 실제 시간 로그: compute / sleep / total + ETA
            mlmod.log_metrics({
                "accuracy": acc,
                "f1_score": f1,
                "log_loss": ll,
                "epoch_compute_sec": compute_sec,              # 실제 학습시간
                "epoch_sleep_sec": SLEEP_SEC,                  # 대기시간
                "epoch_time_sec": compute_sec + SLEEP_SEC,     # 합계
                "eta_sec": eta_sec,                            # 남은 시간 추정
                "progress_pct": 100.0 * epoch / EPOCHS,        # 진행률(%)
            }, step=epoch)

            f1_hist.append(f1)
            print(f"[epoch {epoch:03d}] acc={acc:.4f} f1={f1:.4f} "
                  f"comp={compute_sec:.3f}s sleep={SLEEP_SEC:.2f}s ETA={eta_sec:.1f}s")

            if SLEEP_SEC > 0:
                time.sleep(SLEEP_SEC)

        train_time = time.time() - t0
        mlmod.log_metric("train_time_total_sec", train_time)

        # ===== Confusion Matrix → artifact =====
        cm = confusion_matrix(y_test, clf.predict(X_test))
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

        # ===== Learning Curve(에폭별 f1) → artifact =====
        plt.figure(figsize=(6, 3.5))
        plt.plot(range(1, len(f1_hist)+1), f1_hist, marker="o")
        plt.title("F1 over Epochs")
        plt.xlabel("Epoch"); plt.ylabel("F1 (macro)"); plt.grid(True, alpha=0.3)
        lc_path = "learning_curve_f1.png"
        plt.savefig(lc_path, bbox_inches="tight"); plt.close()
        mlmod.log_artifact(lc_path, artifact_path="plots")

        # ===== 모델 업로드 =====
        from mlflow import sklearn as ml_sklearn
        signature = infer_signature(X_train, clf.predict(X_train))
        ml_sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:2],
        )

        with open("input_example.json", "w") as f:
            json.dump(X_test[:2].tolist(), f)
        mlmod.log_artifact("input_example.json")

        print(f"✅ Train done: epochs={EPOCHS}, total_time={train_time:.2f}s")

if __name__ == "__main__":
    main()

