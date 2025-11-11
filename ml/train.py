# ml/train.py
import os, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import mlflow as mlmod
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss

try:
    from logs import log_json_line, log_ghcr_metadata_to_mlflow
except Exception:
    def log_json_line(obj): print(json.dumps(obj, ensure_ascii=False))
    def log_ghcr_metadata_to_mlflow(): pass

EXP_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn")
RUN_NAME = (os.getenv("GIT_SHA", "")[:12] or "run")
EPOCHS_CAP = int(os.getenv("MLFLOW_EPOCHS", "10000"))  # 안전 상한
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "4096"))
RANDOM_STATE = int(os.getenv("SEED", "42"))
LR_ALPHA = float(os.getenv("LR_ALPHA", "0.0005"))
TARGET_WALL_SEC = float(os.getenv("TARGET_WALL_SEC", "600"))
EMA_ALPHA = float(os.getenv("ETA_EMA_ALPHA", "0.2"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "gb").lower()  # sgd|gb|rf
TREES_PER_EPOCH = int(os.getenv("TREES_PER_EPOCH", "100"))

DEFAULT_CSV_CANDIDATES = [
    "s3://data/telco/Telco-Customer-Churn.csv",
    "s3://data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
TELCO_CSV_URI_ENV = os.getenv("TELCO_CSV_URI", "").strip()

def ensure_experiment_id(name, client):
    exp = client.get_experiment_by_name(name)
    if exp is None:
        try: client.create_experiment(name)
        except RestException: pass
        exp = client.get_experiment_by_name(name)
    return exp.experiment_id

def start_run_with_retry(exp_id, run_name):
    for _ in range(12):
        try: return mlmod.start_run(experiment_id=exp_id, run_name=run_name)
        except RestException: time.sleep(0.3)
    raise RuntimeError("failed to start run")

def _s3_storage_options_from_env():
    opts = {}
    key, sec, tok = os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"), os.getenv("AWS_SESSION_TOKEN")
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
    if key: opts["key"] = key
    if sec: opts["secret"] = sec
    if tok: opts["token"] = tok
    if endpoint: opts.setdefault("client_kwargs", {})["endpoint_url"] = endpoint
    return opts

def _exists_s3(path, storage_options):
    import fsspec
    fs, _, paths = fsspec.get_fs_token_paths(path, storage_options=storage_options)
    return fs.exists(paths[0])

def _resolve_telco_csv_uri():
    so = _s3_storage_options_from_env()
    if TELCO_CSV_URI_ENV:
        if TELCO_CSV_URI_ENV.startswith("s3://"):
            if _exists_s3(TELCO_CSV_URI_ENV, so):
                print(f"[data] Using TELCO_CSV_URI from ENV: {TELCO_CSV_URI_ENV}", flush=True); return TELCO_CSV_URI_ENV
            else:
                print(f"[warn] TELCO_CSV_URI not found: {TELCO_CSV_URI_ENV}", flush=True)
        elif Path(TELCO_CSV_URI_ENV).exists():
            print(f"[data] Using local TELCO_CSV_URI from ENV: {TELCO_CSV_URI_ENV}", flush=True); return TELCO_CSV_URI_ENV
    for cand in DEFAULT_CSV_CANDIDATES:
        if cand.startswith("s3://") and _exists_s3(cand, so):
            print(f"[data] Using autodetected CSV: {cand}", flush=True); return cand
        elif Path(cand).exists():
            print(f"[data] Using local CSV: {cand}", flush=True); return cand
    raise FileNotFoundError(f"No CSV found: {DEFAULT_CSV_CANDIDATES}")

def batch_iter(X, y, batch_size, shuffle=True, seed=None):
    X = np.asarray(X); y = np.asarray(y)
    n = len(X); idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed); rng.shuffle(idx)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n); b = idx[s:e]
        yield X[b], y[b]

def load_telco_churn(csv_uri):
    so = _s3_storage_options_from_env() if csv_uri.startswith("s3://") else None
    df = pd.read_csv(csv_uri, storage_options=so)
    for c in ("customerID", "CustomerID", "customerId"):
        if c in df.columns: df = df.drop(columns=[c]); break
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=[c for c in ["Churn", "TotalCharges"] if c in df.columns])
    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn"])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return X, y, preproc, cat_cols, num_cols

def _eval_and_log(clf, X_test, y_test, epoch, wall_start, comp, ema):
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1  = float(f1_score(y_test, y_pred, average="macro"))
    try:
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test); ll = float(log_loss(y_test, y_proba))
        else:
            ll = float("nan")
    except Exception:
        ll = float("nan")
    elapsed = time.perf_counter() - wall_start
    eta = max(0.0, TARGET_WALL_SEC - elapsed)
    mlmod.log_metrics({"accuracy":acc,"f1_score":f1,"log_loss":ll,
                       "epoch_compute_sec":comp,"elapsed_sec":elapsed,"eta_sec":eta}, step=epoch)
    return acc, f1, ll, elapsed, eta

def main():
    wall_start = time.perf_counter()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or mlmod.get_tracking_uri()
    mlmod.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    csv_uri = _resolve_telco_csv_uri()
    X_all, y_all, preproc, cat_cols, num_cols = load_telco_churn(csv_uri)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.30, random_state=0, stratify=y_all
    )
    y_train = y_train.reset_index(drop=True).to_numpy()
    y_test  = y_test.reset_index(drop=True).to_numpy()

    preproc.fit(X_train_df)
    X_train = preproc.transform(X_train_df)
    X_test  = preproc.transform(X_test_df)

    if MODEL_TYPE == "sgd":
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=LR_ALPHA,
                            learning_rate="optimal", random_state=RANDOM_STATE,
                            fit_intercept=True, max_iter=1, warm_start=False, tol=None)
    elif MODEL_TYPE == "gb":
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE, warm_start=True,
                                         learning_rate=0.05, subsample=0.8, max_depth=3,
                                         n_estimators=0)
    elif MODEL_TYPE == "rf":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, warm_start=True,
                                     n_estimators=0, n_jobs=-1, max_features="sqrt")
    else:
        raise ValueError(f"unknown MODEL_TYPE={MODEL_TYPE}")

    exp_id = ensure_experiment_id(EXP_NAME, client)

    with start_run_with_retry(exp_id, RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}, exp_id={exp_id}")
        log_ghcr_metadata_to_mlflow()
        mlmod.log_params({
            "dataset":"telco","model":MODEL_TYPE,"epochs_cap":EPOCHS_CAP,
            "batch_size":BATCH_SIZE,"alpha":LR_ALPHA,
            "trees_per_epoch":(TREES_PER_EPOCH if MODEL_TYPE in ("gb","rf") else 0),
            "target_wall_sec":TARGET_WALL_SEC
        })

        f1_hist, ema = [], None
        epoch = 0

        if MODEL_TYPE == "sgd":
            first_n = min(BATCH_SIZE, len(X_train))
            clf.partial_fit(X_train[:first_n], y_train[:first_n], classes=np.array([0,1], int))
            while (time.perf_counter() - wall_start) < TARGET_WALL_SEC and epoch < EPOCHS_CAP:
                epoch += 1
                t0 = time.perf_counter()
                for Xb, yb in batch_iter(X_train, y_train, BATCH_SIZE, shuffle=True,
                                         seed=RANDOM_STATE + epoch * 1000):
                    clf.partial_fit(Xb, yb)
                comp = time.perf_counter() - t0
                ema = comp if ema is None else (EMA_ALPHA*comp + (1-EMA_ALPHA)*ema)
                acc, f1, ll, elapsed, eta = _eval_and_log(clf, X_test, y_test, epoch, wall_start, comp, ema)
                f1_hist.append(f1)
                print(f"[epoch {epoch:03d}] acc={acc:.4f} f1={f1:.4f} comp={comp:.2f}s elapsed={elapsed:.1f}s ETA~{eta:.1f}s")
        else:
            while (time.perf_counter() - wall_start) < TARGET_WALL_SEC and epoch < EPOCHS_CAP:
                epoch += 1
                t0 = time.perf_counter()
                clf.n_estimators += TREES_PER_EPOCH
                clf.fit(X_train, y_train)
                comp = time.perf_counter() - t0
                ema = comp if ema is None else (EMA_ALPHA*comp + (1-EMA_ALPHA)*ema)
                acc, f1, ll, elapsed, eta = _eval_and_log(clf, X_test, y_test, epoch, wall_start, comp, ema)
                f1_hist.append(f1)
                print(f"[trees+{TREES_PER_EPOCH:03d} | {epoch:03d}] acc={acc:.4f} f1={f1:.4f} trees={getattr(clf,'n_estimators',0)} comp={comp:.2f}s elapsed={elapsed:.1f}s ETA~{eta:.1f}s")

        total = time.perf_counter() - wall_start
        mlmod.log_metric("train_time_total_sec", total)

        cm = confusion_matrix(y_test, clf.predict(X_test))
        plt.figure(figsize=(5,4)); im = plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix (Telco)")
        plt.colorbar(im); tick = np.arange(2); plt.xticks(tick, ["stay","churn"]); plt.yticks(tick, ["stay","churn"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]): plt.text(j,i,cm[i,j],ha="center",va="center")
        plt.tight_layout(); plt.savefig("confusion_matrix.png", bbox_inches="tight")
        mlmod.log_artifact("confusion_matrix.png", artifact_path="plots")

        plt.figure(figsize=(6,3.5)); plt.plot(range(1,len(f1_hist)+1), f1_hist, marker="o")
        plt.title("F1 over Epochs (Telco)"); plt.xlabel("Epoch"); plt.ylabel("F1 (macro)"); plt.grid(True, alpha=0.3)
        plt.savefig("learning_curve_f1.png", bbox_inches="tight")
        mlmod.log_artifact("learning_curve_f1.png", artifact_path="plots")

        from mlflow import sklearn as ml_sklearn
        final_pipe = Pipeline(steps=[("preproc", preproc), ("clf", clf)])
        sig = infer_signature(X_train_df, final_pipe.predict(X_train_df.head(2)))
        ml_sklearn.log_model(sk_model=final_pipe, artifact_path="model", signature=sig, input_example=X_train_df.head(2))
        with open("input_example.json", "w") as f: json.dump(X_train_df.head(2).to_dict(orient="records"), f)
        mlmod.log_artifact("input_example.json")

        final_acc = float(accuracy_score(y_test, final_pipe.predict(X_test_df)))
        print(f"[PROMOTE] accuracy={final_acc:.5f}")
        print("✅ Train done.", flush=True)

if __name__ == "__main__":
    main()

