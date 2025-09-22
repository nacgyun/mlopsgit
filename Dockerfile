# Dockerfile (repo root)
FROM python:3.11-slim-bookworm

# HTTPS 미러로 전환(두 포맷 모두 대응) + IPv4 강제 + 필수 빌드 도구
RUN set -eux; \
    # legacy sources.list 사용 시
    if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
    fi; \
    # deb822 포맷(debian.sources) 사용 시
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4; \
    apt-get -o Acquire::ForceIPv4=true update; \
    apt-get install -y --no-install-recommends build-essential gcc g++; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ml/ /app/

RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ -f /app/requirements.txt ]; then \
      pip install --no-cache-dir -r /app/requirements.txt; \
    fi

ENV MLFLOW_EXPERIMENT_NAME=iris-rf \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["python","-u","/app/train.py"]

