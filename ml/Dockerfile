FROM python:3.10-slim

# scikit-learn 실행에 필요한 libgomp1만 추가
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# pip 최신화 + 필요한 패키지 설치 (버전은 예시, 고정 추천)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
      mlflow==2.14.3 \
      scikit-learn==1.4.2 \
      boto3==1.34.131 \
      pandas==2.2.2

WORKDIR /app
# 실제 /app/train.py 는 K8s에서 ConfigMap으로 마운트하므로 여기서 복사할 필요 없음
CMD ["bash","-lc","python /app/train.py"]
