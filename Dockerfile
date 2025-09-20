# Dockerfile (repo root)
FROM python:3.11-slim

# 필수 OS 패키지(과학연산 휠 설치 보조)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 소스 복사 (ml/ 밑에 train.py, requirements.txt 있다고 가정)
COPY ml/ /app/

# 의존성 (없으면 건너뜀)
RUN if [ -f /app/requirements.txt ]; then \
      pip install --no-cache-dir -r /app/requirements.txt ; \
    fi

ENV MLFLOW_EXPERIMENT_NAME=iris-rf
CMD ["python","-u","/app/train.py"]

