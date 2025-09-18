# syntax=docker/dockerfile:1.6

FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (필요시) 빌드 도구
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

# 1) 의존성만 먼저 복사 → 이 레이어는 requirements가 바뀔 때만 다시 빌드
COPY ml/requirements.txt /tmp/requirements.txt

# 2) pip 설치 (BuildKit 캐시 활용)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# 3) 애플리케이션 코드는 마지막에
COPY ml/ /app/

# (필요시) 기본 실험/엔드포인트 환경변수
ENV MLFLOW_EXPERIMENT_NAME=iris-rf

CMD ["python","-u","/app/train.py"]

