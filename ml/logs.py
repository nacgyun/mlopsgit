# ml/logs.py
import os, json
from pathlib import Path

# JSON 라인 로그 ON/OFF (기본 ON)
LOG_JSON = os.getenv("LOG_JSON", "1") == "1"

def log_json_line(payload: dict):
    """
    한 줄 JSON 로그를 stdout으로 출력.
    Promtail이 수집 → Loki로 전달되는 전제.
    """
    if not LOG_JSON:
        return
    try:
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), flush=True)
    except Exception:
        # 로깅 실패는 학습을 막지 않도록 무시
        pass

def log_ghcr_metadata_to_mlflow():
    """
    GHCR 관련 ENV를 MLflow 태그와 아티팩트로 남긴다.
    - 태그: ci.run_id, git.sha, ghcr.image, ghcr.tag, ghcr.digest, ghcr.ref
    - 아티팩트: build/image.json
    """
    import mlflow as mlmod

    ghcr_image   = os.getenv("GHCR_IMAGE", "")
    ghcr_tag     = os.getenv("GHCR_TAG", "")
    ghcr_digest  = os.getenv("GHCR_DIGEST", "")
    ghcr_ref     = os.getenv("GHCR_IMAGE_REF", "")
    run_id_ci    = os.getenv("GITHUB_RUN_ID", "")
    git_sha      = os.getenv("GIT_SHA", "")

    # MLflow 태그
    try:
        mlmod.set_tag("ci.run_id", run_id_ci)
        mlmod.set_tag("git.sha",   git_sha)
        mlmod.set_tag("ghcr.image",  ghcr_image)
        mlmod.set_tag("ghcr.tag",    ghcr_tag)
        mlmod.set_tag("ghcr.digest", ghcr_digest)
        mlmod.set_tag("ghcr.ref",    ghcr_ref)
    except Exception:
        pass

    # 아티팩트 저장
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
            json.dump(meta, f, indent=2, ensure_ascii=False)
        mlmod.log_artifact("build/image.json", artifact_path="build")
    except Exception:
        pass

