#!/usr/bin/env bash
set -euo pipefail
NS=mlops
JOB=mlflow-train-job
SVC_URL="http://mlflow.mlops.svc.cluster.local:5000"

echo "== 0) 잡/파드 현황 =="
kubectl -n $NS get job $JOB -o wide || true
kubectl -n $NS get pods -l job-name=$JOB -o wide || true
echo

POD=$(kubectl -n $NS get pods -l job-name=$JOB \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].0.metadata.name}' 2>/dev/null || true)

if [ -z "${POD:-}" ]; then
  echo "!! 잡 파드가 아직 없습니다. (스케줄/어드미션 문제일 수 있음)"
  kubectl -n $NS describe job $JOB || true
  exit 1
fi

echo "== 1) 파드 환경변수(MLFLOW_TRACKING_URI 포함) =="
kubectl -n $NS exec "$POD" -- printenv | egrep 'MLFLOW|AWS_|N_ESTIMATORS|MAX_DEPTH' || true
echo

echo "== 2) 파드 로그(마지막 200줄) =="
kubectl -n $NS logs "$POD" --tail=200 || true
echo

echo "== 3) 클러스터 내부에서 MLflow API 확인 =="
kubectl -n $NS run ml-curl --restart=Never --image=curlimages/curl:8.11.1 --rm -i -- \
  -sS "$SVC_URL/version" -w "\nHTTP:%{http_code}\n" || true
echo

# Experiment 목록에서 iris-rf 존재 확인
echo "== 4) Experiment 목록 (iris-rf 유무) =="
kubectl -n $NS run ml-exp --restart=Never --image=curlimages/curl:8.11.1 --rm -i -- \
  -sS -H 'Content-Type: application/json' \
  "$SVC_URL/api/2.0/mlflow/experiments/list" | jq . || true
echo

# iris-rf experiment_id 추출 → 최근 1시간 내 run 검색
EXP_ID=$(
kubectl -n $NS run ml-exp2 --restart=Never --image=curlimages/curl:8.11.1 --rm -i -- \
  -sS "$SVC_URL/api/2.0/mlflow/experiments/list" \
  | jq -r '.experiments[] | select(.name=="iris-rf") | .experiment_id' 2>/dev/null || true
)
echo "Detected experiment_id for iris-rf: ${EXP_ID:-<none>}"
echo

if [ -n "${EXP_ID:-}" ]; then
  echo "== 5) iris-rf 최근 Run 검색(마지막 1시간) =="
  NOW=$(date -u +%s)
  ONEHOUR_AGO=$((NOW - 3600))
  kubectl -n $NS run ml-runs --restart=Never --image=curlimages/curl:8.11.1 --rm -i -- \
    -sS -H 'Content-Type: application/json' \
    -X POST "$SVC_URL/api/2.0/mlflow/runs/search" \
    -d "{\"experiment_ids\":[\"$EXP_ID\"],\"max_results\":50,\"order_by\":[\"attributes.start_time DESC\"]}" | jq . || true
  echo
else
  echo "!! iris-rf experiment가 아직 없거나 이름이 다릅니다. train.py의 set_experiment(\"iris-rf\") 확인 요망."
fi

echo "== 6) MLflow 서버 로그 최근 200줄 =="
kubectl -n $NS logs deploy/mlflow --tail=200 || true
