#!/usr/bin/env sh
set -eu
URL=${URL:-http://light-serve.mlops.svc.cluster.local:8000}

# /ping 헬스체크
curl -fsS "$URL/ping" >/dev/null

# 예시 입력으로 /invocations 호출
cat >/tmp/instances.json <<EOF
{"instances":[[0.12,-0.45,1.03,0.00,-0.18,0.55,-0.77,0.31,1.21,-0.09,0.40,-0.11,
0.07,0.88,-1.34,0.05,0.24,-0.66,0.13,-0.22,0.91,-0.42,0.03,0.50,-0.27,0.61,
-0.33,0.19,-0.48,0.72]]}
EOF

curl -fsS -X POST "$URL/invocations" \
  -H "Content-Type: application/json" \
  --data @/tmp/instances.json >/dev/null

echo "[smoke] OK"

