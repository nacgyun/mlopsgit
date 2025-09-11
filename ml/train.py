import os, mlflow, mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
s3_endpoint  = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
print(f"[INFO] MLFLOW_TRACKING_URI={tracking_uri}")
print(f"[INFO] MLFLOW_S3_ENDPOINT_URL={s3_endpoint}")

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

n_estimators = int(os.environ.get("N_ESTIMATORS", "100"))
max_depth    = int(os.environ.get("MAX_DEPTH", "5"))

with mlflow.start_run(run_name="iris-rf"):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

print(f"[OK] accuracy={acc:.4f}")
