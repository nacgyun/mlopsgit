# ml/train.py (heavy MLP version, final)
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
from sklearn.neural_network import MLPClassifier
from pathlib import Path

# ===== 파라미터 / 설정 =====
EXP_NAME        = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-rf")
RUN_NAME        = (os.getenv("GIT_SHA", "")[:12] or "run")

# 기본 러닝타임 타깃(초) — 대략 5분 유지
TARGET_WALL_SEC = float(os.getenv("TARGET_WALL_SEC", "300"))

EPOCHS          = int(os.getenv("MLFLOW_EPOCHS", "40"))
BATCH_SIZE      = int(os.getenv("TRAIN_BATCH_SIZE", "64"))
SLEEP_SEC       = float(os.getenv("TRAIN_SLEEP_SEC", "0.0"))

# 🔸 학습 연산량 제어 파라미터
LOOPS_PER_EPOCH = int(os.getenv("LOOPS_PER_EPOCH", "3"))
AUGMENT_ENABLE  = os.getenv("AUGMENT_ENABLE", "1") == "1"
AUGMENT_COPIES  = int(os.getenv("AUGMENT_COPIES", "3"))
AUGMENT_NOISE   = float(os.getenv("AUGMENT_NOISE", "0.08"))

# 🔸 burn(추가 연산) - pure compute로 시간 더 쓰기
DEFAULT_BURN_PASSES = "1200"
BURN_PASSES     = int(os.getenv("BURN_PASSES", DEFAULT_BURN_PASSES))
BURN_NOISE      = float(os.getenv("BURN_NOISE", "0.08"))
BURN_ENABLE     = os.getenv("BURN_ENABLE", "1") == "1"
BURN_CHUNK_PASSES = int(os.getenv("BURN_CHUNK_PASSES", "256"))

# 🔸 MLP 하이퍼파라미터 (무겁게 만들기 위한 포인트)
HIDDEN_WIDTH    = int(os.getenv("MLP_HIDDEN_WIDTH", "256"))
HIDDEN_LAYERS   = int(os.getenv("MLP_HIDDEN_LAYERS", "3"))
LR_INITIAL      = float(os.getenv("LR_INITIAL", "0.01"))
RANDOM_STATE    = int(os.getenv("SEED", "42"))
EMA_ALPHA       = float(os.getenv("ETA_EMA_ALPHA", "0.2"))

# 🔸 로그 설정: Promtail/Loki용 JSON stdout 로그 (그대로 유지)
LOG_JSON        = os.getenv("LOG_JSON", "1") == "1"

def log_json_line(payload: dict):
    """한 줄 JSON 로그 출력 (stdout). Promtail이 수집해서 Loki로 보냄."""
    if not LOG_JSON:
        return
    try:
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    except Exception:
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
    """
    추가 연산(로그 X). template_clf를 복제한 shadow MLP를 만들어서
    랜덤 노이즈를 섞은 배치를 계속 partial_fit해서 CPU 태움.
    """
    if passes <= 0:
        return
    rng = np.random.default_rng(seed)
    shadow = clone(template_clf)
    classes = np.unique(y)

    # 첫 partial_fit으로 클래스 세팅
    if len(X) >= batch_size:
        xb = X[:batch_size]; yb = y[:batch_size]
    else:
        xb = X; yb = y
    shadow.partial_fit(xb, yb, classes=classes)

    n = len(X)
    for _ in range(passes):
        idx = rng.integers(0, n, size=min(batch_size, n))
        Xb = X[idx] + rng.normal(0.0, noise, size=X[idx].shape)
        shadow.partial_fit(Xb, y[idx])

def spend_time_to_target(template_clf, X, y, batch_size, noise, seed, target_deadline_sec):
    """
    벽시계 타깃 시간(TARGET_WALL_SEC)에 맞추기 위해 남은 시간을 burn한다.
    """
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

# === GHCR 메타데이터 MLflow 태그/아티팩트 기록 (기존 그대로 유지)
def log_ghcr_metadata_to_mlflow():
    ghcr_image   = os.getenv("GHCR_IMAGE", "")
    ghcr_tag     = os.getenv("GHCR_TAG", "")
    ghcr_digest  = os.getenv("GHCR_DIGEST", "")
    ghcr_ref     = os.getenv("GHCR_IMAGE_REF", "")
    run_id_ci    = os.getenv("GITHUB_RUN_ID", "")
    git_sha      = os.getenv("GIT_SHA", "")

    # 태그 기록
    mlmod.set_tag("ci.run_id", run_id_ci)
    mlmod.set_tag("git.sha",   git_sha)
    mlmod.set_tag("ghcr.image",  ghcr_image)
    mlmod.set_tag("ghcr.tag",    ghcr_tag)
    mlmod.set_tag("ghcr.digest", ghcr_digest)
    mlmod.set_tag("ghcr.ref",    ghcr_ref)

    # 아티팩트 JSON 기록
    try:
        meta = {
            "image": ghcr_image,
            "tag": ghcr_tag,
            "digest": ghcr_digest,
            "ref": ghcr_ref,
            "run_id": run_id_ci,
            "git_sha": git_sha
        }
        Path("build").mkdir(exist_ok=True)
        with open("build/image.json", "w") as f:
            json.dump(meta, f, indent=2)
        mlmod.log_artifact("build/image.json", artifact_path="build")
    except Exception:
        pass

def main():
    wall_start = time.perf_counter()

    # MLflow 세팅
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = ensure_experiment_id(EXP_NAME, client)

    # 데이터 준비
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.30, random_state=0
    )

    # 데이터 증강 (원래 로직 유지)
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

    # ===== 무거운 MLP 모델 설정 =====
    hidden_layers = tuple([HIDDEN_WIDTH] * HIDDEN_LAYERS)

    base_clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="sgd",
        learning_rate_init=LR_INITIAL,
        momentum=0.9,
        n_iter_no_change=200,      # 조기 종료 안 걸리게 크게
        max_iter=1,                # 한 partial_fit마다 한 스텝만
        random_state=RANDOM_STATE,
        warm_start=False,
        tol=1e-12,                 # ← 여기 수정 (원래 None이어서 에러 났던 부분)
        alpha=0.0001,
        shuffle=False,
        verbose=False
    )

    with start_run_with_retry(exp_id, RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}, exp_id={exp_id}")

        # GHCR 메타데이터 MLflow에 남기기
        log_ghcr_metadata_to_mlflow()

        # MLflow 파라미터 기록 (형식은 유지, 모델 설명만 변경)
        mlmod.log_params({
            "model": f"MLPClassifier(hidden_layers={hidden_layers}, sgd)",
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

        # MLPClassifier는 partial_fit 첫 호출 시 classes 필요
        clf = clone(base_clf)
        init_batch_X = X_train[:min(BATCH_SIZE, len(X_train))]
        init_batch_y = y_train[:min(BATCH_SIZE, len(y_train))]
        clf.partial_fit(init_batch_X, init_batch_y, classes=classes)

        f1_hist = []
        ema = None  # epoch compute time EMA

        for epoch in range(1, EPOCHS + 1):
            t_epoch = time.perf_counter()

            # 본 학습 여러 loop/batch로 partial_fit 반복 → 무겁게
            for loop in range(LOOPS_PER_EPOCH):
                for Xb, yb in batch_iter(
                    X_train, y_train,
                    BATCH_SIZE,
                    shuffle=True,
                    seed=RANDOM_STATE + epoch * 1000 + loop
                ):
                    clf.partial_fit(Xb, yb)

            # burn 단계: 추가 연산으로 더 시간 소모
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

            # === 평가 및 로깅 (형식 그대로) ===
            y_pred = clf.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1  = float(f1_score(y_test, y_pred, average="macro"))
            try:
                y_proba = clf.predict_proba(X_test)
                ll = float(log_loss(y_test, y_proba))
            except Exception:
                ll = float("nan")

            # ETA 추정용 EMA
            if ema is None:
                ema = compute_sec
            else:
                ema = EMA_ALPHA * compute_sec + (1 - EMA_ALPHA) * ema

            elapsed = time.perf_counter() - wall_start
            eta_sec = max(0.0, TARGET_WALL_SEC - elapsed)

            # MLflow metrics 기록 (키 이름 동일)
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

            # stdout 텍스트 로그 + Loki용 JSON 로그 (형식 완전 동일)
            print(
                f"[epoch {epoch:03d}] acc={acc:.4f} f1={f1:.4f} comp={compute_sec:.2f}s "
                f"sleep={SLEEP_SEC:.2f}s elapsed={elapsed:.1f}s ETA~{eta_sec:.1f}s"
            )
            log_json_line({
                "event": "epoch_metric",
                "epoch": epoch,
                "accuracy": acc,
                "duration": round(compute_sec + SLEEP_SEC, 4),
                "remaining_sec": round(eta_sec, 1),
                "run_id": run_id,
                "experiment": EXP_NAME,
            })

            # epoch sleep (같은 형식 유지)
            if SLEEP_SEC > 0:
                time.sleep(SLEEP_SEC)

            # 남은 시간 태우기 (벽시계 5분 맞춤)
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

            # 목표 시간 채우면 조기 종료 (원래 로직 유지)
            if time.perf_counter() - wall_start >= TARGET_WALL_SEC:
                print(f"[info] target wall time ({TARGET_WALL_SEC:.0f}s) reached. stopping early.")
                break

        # 총 학습 시간 MLflow에 기록 (키는 그대로)
        total_time = time.perf_counter() - wall_start
        mlmod.log_metric("train_time_total_sec", total_time)

        # ===== 아티팩트 기록 (같은 구조 유지) =====
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

        # 모델 저장 (artifact_path="model" 유지 → promote 파이프라인에 그대로 먹힘)
        from mlflow import sklearn as ml_sklearn
        signature = infer_signature(X_train, clf.predict(X_train))
        ml_sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:2]
        )

        # 예제 입력도 동일하게 업로드
        with open("input_example.json", "w") as f:
            json.dump(X_test[:2].tolist(), f)
        mlmod.log_artifact("input_example.json")

        # 마지막 Loki용 JSON 로그 (형식 유지)
        log_json_line({
            "event": "train_done",
            "accuracy": float(accuracy_score(y_test, clf.predict(X_test))),
            "duration": round(total_time, 4),
            "remaining_sec": 0.0,
            "run_id": run_id,
            "experiment": EXP_NAME,
        })

        print("✅ Train done.")

if __name__ == "__main__":
    main()

