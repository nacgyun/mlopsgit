# ml/train.py
import os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow as mlmod
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models import infer_signature
from sklearn.base import clone

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.linear_model import SGDClassifier

# ===== 파라미터 / 설정 =====
EXP_NAME        = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")
RUN_NAME        = (os.getenv("GIT_SHA", "")[:12] or "run")

# 기본 러닝타임 타깃(초) — 대략 5분
TARGET_WALL_SEC = float(os.getenv("TARGET_WALL_SEC", "300"))

EPOCHS          = int(os.getenv("MLFLOW_EPOCHS", "40"))
BATCH_SIZE      = int(os.getenv("TRAIN_BATCH_SIZE", "64"))
SLEEP_SEC       = float(os.getenv("TRAIN_SLEEP_SEC", "0.0"))

# 🔸 학습 연산량 제어 파라미터
LOOPS_PER_EPOCH = int(os.getenv("LOOPS_PER_EPOCH", "3"))
AUGMENT_ENABLE  = os.getenv("AUGMENT_ENABLE", "1") == "1"
AUGMENT_COPIES  = int(os.getenv("AUGMENT_COPIES", "3"))
AUGMENT_NOISE   = float(os.getenv("AUGMENT_NOISE", "0.08"))

# 🔸 burn(추가 연산)
DEFAULT_BURN_PASSES = "1200"
BURN_PASSES     = int(os.getenv("BURN_PASSES", DEFAULT_BURN_PASSES))
BURN_NOISE      = float(os.getenv("BURN_NOISE", "0.08"))
BURN_ENABLE     = os.getenv("BURN_ENABLE", "1") == "1"
BURN_CHUNK_PASSES = int(os.getenv("BURN_CHUNK_PASSES", "256"))

# SGD 하이퍼파라미터
LR_ALPHA        = float(os.getenv("LR_ALPHA", "0.0005"))
LR_INITIAL      = float(os.getenv("LR_INITIAL", "0.01"))
RANDOM_STATE    = int(os.getenv("SEED", "42"))
EMA_ALPHA       = float(os.getenv("ETA_EMA_ALPHA", "0.2"))

# 🔸 로그 설정: Promtail/Loki용 JSON 로그 출력 (기본 ON)
LOG_JSON        = os.getenv("LOG_JSON", "1") == "1"

def log_json_line(payload: dict):
    """한 줄 JSON 로그 출력 (stdout). Promtail이 수집해 Loki로 보냄."""
    if not LOG_JSON:
        return
    try:
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    except Exception:
        # JSON 직렬화 실패해도 학습은 계속
        pass

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

def extra_training_burn(template_clf, X, y, passes, batch_size, noise, seed):
    """추가 학습(로그 반영 X, 연산량만 증가)."""
    if passes <= 0:
        return
    rng = np.random.default_rng(seed)
    shadow = clone(template_clf)
    uclasses = np.unique(y)
    if len(X) >= batch_size:
        shadow.partial_fit(X[:batch_size], y[:batch_size], classes=uclasses)
    else:
        shadow.partial_fit(X, y, classes=uclasses)
    n = len(X)
    for _ in range(passes):
        idx = rng.integers(0, n, size=min(batch_size, n))
        Xb = X[idx] + rng.normal(0.0, noise, size=X[idx].shape)
        shadow.partial_fit(Xb, y[idx])

def spend_time_to_target(template_clf, X, y, batch_size, noise, seed, target_deadline_sec):
    """목표 벽시계 시간에 맞추기 위한 burn 반복."""
    if not BURN_ENABLE:
        return
    start = time.perf_counter()
    rng_seed = seed
    while time.perf_counter() - start < target_deadline_sec:
        extra_training_burn(
            template_clf=template_clf,
            X=X, y=y,
            passes=BURN_CHUNK_PASSES,
            batch_size=batch_size,
            noise=noise,
            seed=rng_seed
        )
        rng_seed += 1

def main():
    wall_start = time.perf_counter()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = ensure_experiment_id(EXP_NAME, client)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.30, random_state=0
    )

    # ==== 데이터 증강 ====
    if AUGMENT_ENABLE and AUGMENT_COPIES > 0:
        rng = np.random.default_rng(RANDOM_STATE + 777)
        X_aug_list = [X_train]; y_aug_list = [y_train]
        for _ in range(AUGMENT_COPIES):
            noise = rng.normal(0.0, AUGMENT_NOISE, size=X_train.shape)
            X_aug_list.append(X_train + noise)
            y_aug_list.append(y_train)
        X_train = np.vstack(X_aug_list)
        y_train = np.hstack(y_aug_list)

    classes = np.unique(y_train)

    with start_run_with_retry(exp_id, RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}, exp_id={exp_id}")

        mlmod.log_params({
            "model": "SGDClassifier(logistic)",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "sleep_sec": SLEEP_SEC,
            "loops_per_epoch": LOOPS_PER_EPOCH,
            "burn_passes": BURN_PASSES,
            "augment": AUGMENT_ENABLE,
            "augment_copies": AUGMENT_COPIES,
            "augment_noise": AUGMENT_NOISE,
            "target_wall_sec": TARGET_WALL_SEC,
            "burn_chunk_passes": BURN_CHUNK_PASSES
        })

        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=LR_ALPHA,
            learning_rate="optimal",
            random_state=RANDOM_STATE,
            fit_intercept=True,
            max_iter=1,
            warm_start=False,
            tol=None
        )

        # 초기 partial_fit (classes 세팅)
        clf.partial_fit(X_train[:BATCH_SIZE], y_train[:BATCH_SIZE], classes=classes)

        f1_hist = []
        ema = None

        for epoch in range(1, EPOCHS + 1):
            t_epoch = time.perf_counter()

            # 본 학습
            for loop in range(LOOPS_PER_EPOCH):
                for Xb, yb in batch_iter(
                    X_train, y_train, BATCH_SIZE, shuffle=True,
                    seed=RANDOM_STATE + epoch * 1000 + loop
                ):
                    clf.partial_fit(Xb, yb)

            # 고정 burn
            if BURN_ENABLE and BURN_PASSES > 0:
                extra_training_burn(
                    template_clf=clf,
                    X=X_train, y=y_train,
                    passes=BURN_PASSES,
                    batch_size=BATCH_SIZE,
                    noise=BURN_NOISE,
                    seed=RANDOM_STATE + 1000 + epoch
                )

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

            if ema is None:
                ema = compute_sec
            else:
                ema = EMA_ALPHA * compute_sec + (1 - EMA_ALPHA) * ema

            elapsed = time.perf_counter() - wall_start
            remaining_epochs = EPOCHS - epoch
            eta_sec = max(0.0, TARGET_WALL_SEC - elapsed)

            mlmod.log_metrics({
                "accuracy": acc,
                "f1_score": f1,
                "log_loss": ll,
                "epoch_compute_sec": compute_sec,
                "epoch_sleep_sec": SLEEP_SEC,
                "epoch_time_sec": compute_sec + SLEEP_SEC,
                "eta_sec": eta_sec,
                "progress_pct": min(99.9, 100.0 * epoch / EPOCHS),
                "elapsed_sec": elapsed
            }, step=epoch)

            f1_hist.append(f1)
            # ── 사람이 읽는 로그
            print(f"[epoch {epoch:03d}] acc={acc:.4f} f1={f1:.4f} comp={compute_sec:.2f}s "
                  f"sleep={SLEEP_SEC:.2f}s elapsed={elapsed:.1f}s ETA~{eta_sec:.1f}s")
            # ── Loki용 JSON 로그 (대시보드 패널이 이 값을 unwarp)
            log_json_line({
                "event":"epoch_metric",
                "epoch": epoch,
                "accuracy": acc,
                # duration = 에폭 총 시간(학습+sleep). 패널은 이 키를 사용합니다.
                "duration": round(compute_sec + SLEEP_SEC, 4),
                "run_id": run_id,
                "experiment": EXP_NAME,
            })

            if SLEEP_SEC > 0:
                time.sleep(SLEEP_SEC)

            # 타임 타깃 보정
            remaining_to_target = TARGET_WALL_SEC - (time.perf_counter() - wall_start)
            if BURN_ENABLE and remaining_to_target > 5.0:
                per_epoch_cap = float(os.getenv("PER_EPOCH_SPEND_CAP_SEC", "60"))
                to_spend = min(per_epoch_cap, max(0.0, remaining_to_target * 0.4))
                if to_spend > 1.0:
                    spend_time_to_target(
                        template_clf=clf,
                        X=X_train, y=y_train,
                        batch_size=BATCH_SIZE,
                        noise=BURN_NOISE,
                        seed=RANDOM_STATE + 2000 + epoch,
                        target_deadline_sec=to_spend
                    )

            if time.perf_counter() - wall_start >= TARGET_WALL_SEC:
                print(f"[info] target wall time ({TARGET_WALL_SEC:.0f}s) reached. stopping early.")
                break

        # 총 학습 시간(더미 키 유지용)
        total_time = time.perf_counter() - wall_start
        mlmod.log_metric("train_time_total_sec", total_time)

        # ===== 아티팩트 =====
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
        plt.savefig("confusion_matrix.png", bbox_inches="tight")
        mlmod.log_artifact("confusion_matrix.png", artifact_path="plots")

        plt.figure(figsize=(6, 3.5))
        plt.plot(range(1, len(f1_hist)+1), f1_hist, marker="o")
        plt.title("F1 over Epochs")
        plt.xlabel("Epoch"); plt.ylabel("F1 (macro)"); plt.grid(True, alpha=0.3)
        plt.savefig("learning_curve_f1.png", bbox_inches="tight")
        mlmod.log_artifact("learning_curve_f1.png", artifact_path="plots")

        from mlflow import sklearn as ml_sklearn
        signature = infer_signature(X_train, clf.predict(X_train))
        ml_sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:2]
        )

        with open("input_example.json", "w") as f:
            json.dump(X_test[:2].tolist(), f)
        mlmod.log_artifact("input_example.json")

        # ── 최종 JSON 한 줄 (마지막 포인트로 쓰기 좋음)
        final_acc = float(f1_hist[-1]) if f1_hist else float("nan")
        log_json_line({
            "event":"train_done",
            "accuracy": float(accuracy_score(y_test, clf.predict(X_test))),
            "duration": round(total_time, 4),
            "run_id": run_id,
            "experiment": EXP_NAME,
        })

        print("✅ Train done.")

if __name__ == "__main__":
    main()
