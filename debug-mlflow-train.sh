#!/usr/bin/env bash
set -euo pipefail
NS="mlops"
JOB="mlflow-train-job"

echo "== Job 상태 =="
kubectl -n "$NS" get job "$JOB" -o wide || true
echo

echo "== 최근 이벤트 (네임스페이스) =="
kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -n 50 || true
echo

echo "== Pod 대기 (최대 90초) =="
POD=""
for i in {1..30}; do
  POD=$(kubectl -n "$NS" get pods -l job-name="$JOB" \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].0.metadata.name}' 2>/dev/null || true)
  if [ -n "${POD}" ]; then
    echo "  -> Pod 발견: ${POD}"
    break
  fi
  sleep 3
done

if [ -z "${POD}" ]; then
  echo "!! 파드가 생성되지 않았습니다. (스케줄/어드미션/이미지풀 문제 가능)"
  echo
  echo "== Job 상세 (describe) =="
  kubectl -n "$NS" describe job "$JOB" || true
  echo
  echo "== 네임스페이스 이벤트 (상세) =="
  kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -n 100 || true
  exit 1
fi

echo
echo "== Pod 상태 =="
kubectl -n "$NS" get pod "$POD" -o wide || true
echo

echo "== Pod 디스크립션 (하단) =="
kubectl -n "$NS" describe pod "$POD" | tail -n +1 || true
echo

PHASE=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
if [ "$PHASE" = "Pending" ]; then
  echo ">> 상태: Pending (이미지 풀/볼륨/네트워크 정책/스케줄 조건 확인 필요)"
fi

echo
echo "== 컨테이너 로그 팔로우 =="
kubectl -n "$NS" logs -f "$POD" || true
