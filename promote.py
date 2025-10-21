#!/usr/bin/env python3
import os
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME   = os.environ["MODEL_NAME"]          # 예: iris
RUN_ID       = os.environ["RUN_ID"]              # 예: 1234abcd...

client = MlflowClient(TRACKING_URI)

def ensure_registered_model(name):
    try:
        client.get_registered_model(name)
    except RestException:
        client.create_registered_model(name)

def find_or_create_version(name, run_id):
    src = f"runs:/{run_id}/artifacts/model"   # 아티팩트 경로는 train 시 log_model(artifact_path='model') 기준
    for mv in client.search_model_versions(f"name='{name}'"):
        if mv.run_id == run_id:
            return mv
    return client.create_model_version(name=name, source=src, run_id=run_id)

def promote(name, version, stage="Production", archive=True):
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=stage,
        archive_existing_versions=archive
    )

def main():
    if not MODEL_NAME or not RUN_ID:
        raise SystemExit("MODEL_NAME / RUN_ID env required")
    ensure_registered_model(MODEL_NAME)
    mv = find_or_create_version(MODEL_NAME, RUN_ID)
    promote(MODEL_NAME, mv.version, stage="Production", archive=True)
    print(f"✅ Promoted run {RUN_ID} -> {MODEL_NAME} v{mv.version} (Production)")

if __name__ == "__main__":
    main()
