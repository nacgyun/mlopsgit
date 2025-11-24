# Github Action MLOps 프로젝트
Git push 한 번으로 학습 → 배포 → 모니터링까지 자동화되는 MLOps 파이프라인

## 프로젝트 소개

이 프로젝트는 GitHub Actions 기반 CI/CD, MLflow Model Registry,
MinIO 기반 데이터·모델 스토리지, FastAPI 서빙,
Prometheus·Grafana·Loki 모니터링까지 포함한 MLOps 파이프라인 구축 프로젝트입니다.

## 시스템 아키텍처

<img width="1317" height="803" alt="mlops시스템아키텍처" src="https://github.com/user-attachments/assets/560576a3-1074-4662-ad30-999da8262231" />

- 개발자는 `git push`만 수행
- GitHub Actions → Kaniko로 이미지 빌드 → GHCR 푸시
- Kubernetes Job에서 **train.py** 실행, MinIO에서 데이터 로드, MLflow로 로그/모델 기록
- 성능 기준 충족 시, 모델 자동 **승격(Promotion)** 및 Serving Deployment 업데이트
- Prometheus/Grafana/Loki로 리소스 · 메트릭 · 로그 모니터링


## 주요 기능 (Features)

-  컨테이너 기반 **모델 학습 Job** 자동 실행
-  MinIO(S3)를 이용한 **데이터/모델 아티팩트 관리**
-  MLflow를 활용한 **실험 관리 & 모델 레지스트리**
-  FastAPI 기반 **서빙 서비스(light-serve)** 배포
-  GitHub Actions + Kaniko를 이용한 **CI/CD 파이프라인**
-  Prometheus + Grafana + Loki로 **리소스/지표/로그 모니터링**

