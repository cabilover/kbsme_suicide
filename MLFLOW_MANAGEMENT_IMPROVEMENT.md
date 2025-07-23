# MLflow 실험 디렉토리 관리 개선사항

## [2025-07-23] 최신 코드 반영 내역

### 🔧 **MLflow meta.yaml 손상 문제 완전 해결**
- **문제 상황**: MLflow 실험 중 `meta.yaml` 파일이 반복적으로 손상되어 실험 추적 불가능
- **해결 방안**:
  - **안전한 MLflow Run 관리**: `safe_mlflow_run` 컨텍스트 매니저로 예외 발생 시 자동 `FAILED` 상태 종료
  - **안전한 로깅 시스템**: `safe_log_param`, `safe_log_metric`, `safe_log_artifact` 함수로 모든 로깅에 예외 처리
  - **실험 무결성 검증**: 실험 시작 전 `meta.yaml` 파일 무결성 검증 및 자동 복구
  - **Orphaned 실험 정리**: `meta.yaml` 없는 실험 디렉토리 자동 감지 및 백업 후 정리
  - **safe_log_metric 함수의 logger_instance 인자 사용 버그 수정 (TypeError 완전 해결)**

### 🔧 **MLflow 파라미터 중복 로깅 문제 해결**
- **문제**: `resampling_enabled` 파라미터가 `True`/`False`로 중복 로깅되어 MLflow 오류 발생
- **해결**: 중복 로깅 제거 및 안전한 로깅 방식 적용으로 모든 경고 메시지 해결

### 🔧 **primary_metric 로깅 실패 문제 해결**
- **문제**: `configs/base/evaluation.yaml`에 `primary_metric` 설정 누락으로 KeyError 발생
- **해결**: `primary_metric: "f1"` 설정 추가 및 안전한 참조 방식 적용

### 🔧 **실험 전 사전 정리 시스템 구현**
- **현재 실험 상태 확인**: `print_experiment_summary()` 함수로 MLflow 상태 출력
- **Orphaned 실험 정리**: 실험 실행 전 자동 백업 및 정리
- **실험 무결성 검증**: 실험 시작 전 무결성 검증 및 복구 시도

## [2025-07-21] 최신 코드 반영 내역

- **신규 유틸리티/스크립트**
    - `src/utils/mlflow_manager.py`: MLflow 실험 관리 클래스 및 함수 신설
    - `scripts/cleanup_mlflow_experiments.py`: orphaned 실험/오래된 run 정리, 백업, meta.yaml 문제 자동화
- **핵심 함수/클래스**
    - `MLflowExperimentManager`: 실험 요약, orphaned 실험 정리, 안전한 실험 생성, 백업 등 제공
    - `setup_mlflow_experiment_safely`, `cleanup_mlflow_experiments`: 외부에서 안전하게 호출 가능
- **기존 실험 스크립트 변경**
    - `scripts/run_hyperparameter_tuning.py`, `scripts/run_resampling_experiment.py`에서
      기존 `mlflow.get_experiment_by_name`/`mlflow.create_experiment` 직접 호출 →
      `from src.utils.mlflow_manager import setup_mlflow_experiment_safely`로 변경
    - 실험 생성/조회/ID 반환이 모두 안전하게 동작하며 orphaned 경고 방지
- **경고 해결**
    - orphaned 실험(예: meta.yaml 없는 디렉토리) 자동 탐지 및 삭제/백업
    - `mlruns/backups/` 등 관리용 디렉토리는 MLflow 실험 목록에서 제외
    - 오래된 run(30일 이상) 자동 정리 기능 추가
- **문서화**
    - 본 문서에 전체 개선 내역, 사용법, 코드 예시, 주의사항, 향후 계획 등 상세 기술

---

# 이하 기존 내용(상세 설명, 사용법, 예시 등)

## 개요

MLflow의 실험 디렉토리 관리 문제를 해결하기 위해 다음과 같은 개선사항을 구현했습니다:

1. **Orphaned 실험 정리**: `meta.yaml` 파일이 없는 실험 디렉토리 자동 감지 및 정리
2. **안전한 실험 생성**: 실험 생성 시 중복 및 삭제된 실험 상태 처리
3. **자동 백업**: 실험 삭제 전 자동 백업 기능
4. **오래된 Run 정리**: 30일 이상 된 run 자동 정리
5. **실험 요약 정보**: 현재 실험 상태를 한눈에 볼 수 있는 요약 기능

## 문제 상황

### 원인
- MLflow 실험 디렉토리에서 `meta.yaml` 파일이 누락된 실험들이 존재
- 이전 실험 ID가 재사용되거나 실험 디렉토리 정리가 안 된 경우 발생
- MLflow가 이러한 orphaned 실험을 참조하려 할 때 경고 메시지 발생

### 경고 메시지 예시
```
WARNING:root:Malformed experiment '5fbbd685931c4616bdc67005b055056d'. 
Detailed error Yaml file './mlruns/5fbbd685931c4616bdc67005b055056d/meta.yaml' does not exist.
```

## 해결 방법

### 1. MLflow 실험 관리자 클래스 (`src/utils/mlflow_manager.py`)

```python
from src.utils.mlflow_manager import MLflowExperimentManager

# 관리자 초기화
manager = MLflowExperimentManager()

# 실험 요약 정보 출력
manager.print_experiment_summary()

# Orphaned 실험 정리 (백업 포함)
deleted_experiments = manager.cleanup_orphaned_experiments(backup=True)

# 오래된 run 정리 (30일 이상)
deleted_runs = manager.cleanup_old_runs(days_old=30)
```

### 2. 안전한 실험 생성 함수

```python
from src.utils.mlflow_manager import setup_mlflow_experiment_safely

# 기존 실험이 있으면 재사용, 없으면 새로 생성
experiment_id = setup_mlflow_experiment_safely("my_experiment")
```

### 3. 정리 스크립트 (`scripts/cleanup_mlflow_experiments.py`)

```bash
# 실험 목록 확인
python scripts/cleanup_mlflow_experiments.py --action list

# Orphaned 실험 정리 (백업 포함)
python scripts/cleanup_mlflow_experiments.py --action cleanup --backup --force

# 오래된 run 정리 (7일 이상)
python scripts/cleanup_mlflow_experiments.py --action cleanup --days-old 7 --force
```

## 기존 실험 스크립트 업데이트

### 변경 전
```python
def setup_mlflow_experiment(experiment_name: str):
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id
```

### 변경 후
```python
def setup_mlflow_experiment(experiment_name: str):
    from src.utils.mlflow_manager import setup_mlflow_experiment_safely
    experiment_id = setup_mlflow_experiment_safely(experiment_name)
    return experiment_id
```

## 정리 결과

### 정리 전
- **Orphaned 실험**: 2개 (`5fbbd685931c4616bdc67005b055056d`, `70e4cbddb9bd4af6aa8bc94b7da92397`)
- **총 실험 수**: 17개
- **총 Run 수**: 1293개

### 정리 후
- **Orphaned 실험**: 0개
- **총 실험 수**: 15개
- **총 Run 수**: 1150개 (143개 오래된 run 삭제)
- **백업**: `mlruns/backups/` 디렉토리에 2개 실험 백업

## 사용법

### 1. 정기적인 정리 (권장)
```bash
# 주 1회 실행 권장
python scripts/cleanup_mlflow_experiments.py --action cleanup --backup --days-old 30
```

### 2. 실험 상태 확인
```bash
# 실험 상태 확인
python scripts/cleanup_mlflow_experiments.py --action list
```

### 3. 프로그래밍 방식
```python
from src.utils.mlflow_manager import cleanup_mlflow_experiments

# 정리 실행
result = cleanup_mlflow_experiments(backup=True, days_old=30)
print(f"삭제된 실험: {result['total_deleted_experiments']}개")
print(f"삭제된 run: {result['deleted_runs_count']}개")
```

## 주의사항

1. **백업 권장**: 실험 삭제 전 항상 백업을 활성화하세요
2. **정기 정리**: 주 1회 정도 정리 스크립트를 실행하는 것을 권장합니다
3. **백업 디렉토리**: `mlruns/backups/` 디렉토리는 MLflow가 실험으로 인식하지 않도록 제외 처리됩니다

## 향후 개선 계획

1. **자동화**: cron job을 통한 자동 정리 스크립트
2. **웹 인터페이스**: MLflow UI 확장을 통한 정리 기능
3. **정책 기반 정리**: 실험 중요도에 따른 차등 정리 정책
4. **알림 기능**: 정리 결과를 이메일이나 슬랙으로 알림

## 관련 파일

- `src/utils/mlflow_manager.py`: MLflow 실험 관리자 클래스
- `scripts/cleanup_mlflow_experiments.py`: 정리 스크립트
- `scripts/run_hyperparameter_tuning.py`: 업데이트된 실험 스크립트
- `scripts/run_resampling_experiment.py`: 업데이트된 실험 스크립트 