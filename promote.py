#!/usr/bin/env python3
import os
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME   = os.environ["MODEL_NAME"]          # ex) light-logreg
RUN_ID       = os.environ["RUN_ID"]              # ex) 7e2a0b9f1c7e4c2f8d0a...

client = MlflowClient(tracking_uri=TRACKING_URI)

def ensure_registered_model(name: str):
    try:
        client.get_registered_model(name)
    except RestException:
        client.create_registered_model(name)

def find_existing_version(name: str, run_id: str):
    for mv in client.search_model_versions(f"name='{name}'"):
        if mv.run_id == run_id:
            return mv
    return None

def create_version_with_runs_uri(name: str, run_id: str):
    # ★ 정식 스킴 (artifacts/ 없음)
    src = f"runs:/{run_id}/model"
    return client.create_model_version(name=name, source=src, run_id=run_id)

def promote(name: str, version: str, stage: str = "Production", archive: bool = True):
    client.transition_model_version_stage(
        name=name, version=version, stage=stage, archive_existing_versions=archive
    )

def main():
    if not MODEL_NAME or not RUN_ID:
        raise SystemExit("MODEL_NAME / RUN_ID env required")

    ensure_registered_model(MODEL_NAME)

    mv = find_existing_version(MODEL_NAME, RUN_ID)
    # 기존 버전이 있어도 source가 잘못된 스킴이면 새로 생성
    if mv is None or not str(mv.source).startswith("runs:/") or "/artifacts/" in str(mv.source):
        mv = create_version_with_runs_uri(MODEL_NAME, RUN_ID)

    promote(MODEL_NAME, mv.version, stage="Production", archive=True)
    print(f"✅ Promoted run {RUN_ID} -> {MODEL_NAME} v{mv.version} (Production) | source={mv.source}")

if __name__ == "__main__":
    main()

