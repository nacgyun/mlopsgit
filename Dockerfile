# Dockerfile (repo root)
# Debian testing(trixie) 대신 stable(bookworm) 사용 추천
FROM python:3.11-slim-bookworm

# IPv6 회피(IPv4 강제) + HTTP→HTTPS 미러 전환 + 필수 빌드 도구
RUN set -eux; \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
    echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential gcc g++; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 소스 복사 (ml/ 밑에 train.py, requirements.txt 있다고 가정)
COPY ml/ /app/

# pip 최신화 + 의존성 설치 (있으면)
RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ -f /app/requirements.txt ]; then \
      pip install --no-cache-dir -r /app/requirements.txt; \
    fi

ENV MLFLOW_EXPERIMENT_NAME=iris-rf \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["python","-u","/app/train.py"]

