import os, sys
from mlflow.tracking import MlflowClient

MODEL = os.getenv("MODEL_NAME", "light-logreg")
VER = os.getenv("MODEL_VERSION")
c = MlflowClient()

def latest_staging_version(name: str) -> int:
    vs = c.search_model_versions(f"name='{name}'")
    cand = [v for v in vs if (v.current_stage or "").lower() == "staging"]
    if not cand:
        print("No Staging version found", file=sys.stderr)
        sys.exit(2)
    return max(int(v.version) for v in cand)

ver = int(VER) if VER else latest_staging_version(MODEL)
print(f"[promote] {MODEL} v{ver} -> Production")
c.transition_model_version_stage(
    name=MODEL, version=ver, stage="Production", archive_existing_versions=True
)
print("[promote] done")

