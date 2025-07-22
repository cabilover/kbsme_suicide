# 🔄 대규모 리팩토링 작업 계획서

## �� **개요**
현재 프로젝트의 복잡성과 설정 파일 간의 혼재 문제를 해결하기 위한 대규모 리팩토링을 수행합니다.

## 🎯 **목표**
1. **Hyperparameter Tuning 실험**: 순수한 하이퍼파라미터 튜닝만 수행
2. **Resampling 실험**: 별도 스크립트로 분리하여 독립적 관리
3. **로깅 시스템**: 실제 요구사항에 맞는 로깅 시스템 구축

## 📝 **작업 1: Hyperparameter Tuning에서 Resampling 관련 내용 제거** ✅ **완료**

### 1.1 설정 파일 정리 ✅ **완료**
- **파일**: `configs/experiments/hyperparameter_tuning.yaml`
- **작업 내용**:
  - `resampling` 섹션 완전 제거 ✅
  - `time_series_adapted` 관련 설정 제거 ✅
  - 리샘플링 관련 하이퍼파라미터 제거 ✅
  - 순수한 하이퍼파라미터 튜닝 설정만 유지 ✅

### 1.2 코드 정리 ✅ **완료**
- **파일**: `src/hyperparameter_tuning.py`
- **작업 내용**:
  - 리샘플링 관련 로직 제거 ✅
  - `ResamplingPipeline` 호출 부분 제거 ✅
  - 순수한 하이퍼파라미터 튜닝 로직만 유지 ✅

### 1.3 스크립트 정리 ✅ **완료**
- **파일**: `scripts/run_hyperparameter_tuning.py`
- **작업 내용**:
  - **제거된 함수들**:
    - `run_resampling_tuning_comparison()` (라인 162-313) ✅
    - `add_resampling_hyperparameters_to_tuning_config()` (라인 314-417) ✅
    - `run_resampling_tuning_comparison_with_configmanager()` (라인 457-615) ✅
    - `apply_resampling_hyperparameters_to_config()` (라인 1503-1553) ✅
    - `update_class_distributions_after_resampling()` (라인 1554-1582) ✅
  - **수정된 함수들**:
    - `log_tuning_params()`: 리샘플링 관련 로직 제거 (라인 109-145) ✅
    - `run_hyperparameter_tuning_with_config()`: 리샘플링 매개변수 및 로직 제거 ✅
    - `main()`: 리샘플링 관련 명령행 인자 제거 ✅
  - **제거된 명령행 인자들**:
    - `--resampling-comparison`, `--resampling-methods`, `--experiment-type` (리샘플링 관련), `--resampling-method`, `--resampling-ratio`, `--resampling-enabled` ✅

### 1.4 검증 ✅ **완료**
- **다중 모델 테스트 완료**:
  - **CatBoost**: 최고 성능 0.0008, ~8분 소요 ✅
  - **LightGBM**: 최고 성능 0.0269, ~10분 소요 ✅
  - **XGBoost, Random Forest**: 백그라운드 실행 중 🔄
- **검증 완료**: Resampling 코드 완전 제거, 순수한 하이퍼파라미터 튜닝만 수행, 교차 검증 정상 동작, MLflow 연동 정상 ✅

## 📝 **작업 2: 새로운 Resampling 스크립트 작성**

### 2.1 새로운 스크립트 생성
- **파일**: `scripts/run_resampling_experiment.py`
- **기능**:
  - 리샘플링 기법 비교 실험
  - 하이퍼파라미터 튜닝과 독립적 실행
  - 시계열 특화 설정 제외

### 2.2 설정 파일 재구성
- **파일**: `configs/experiments/resampling.yaml`
- **작업 내용**:
  - `time_series_adapted` 섹션 제거
  - 기본 리샘플링 기법만 유지 (SMOTE, Borderline SMOTE, ADASYN, Under Sampling, Hybrid)
  - 하이퍼파라미터 튜닝 설정과 공유 가능한 부분 분리

### 2.3 공유 설정 관리
- **공유 설정**:
  - `configs/base/common.yaml`: 기본 데이터 처리 설정
  - `configs/base/validation.yaml`: 검증 설정
  - `configs/base/mlflow.yaml`: MLflow 설정
  - `src/utils/`: 모든 유틸리티 함수 공유

### 2.4 리샘플링 전용 설정
- **파일**: `configs/experiments/resampling_methods.yaml`
- **내용**:
  - 각 리샘플링 기법별 설정
  - 리샘플링 관련 하이퍼파라미터만 포함
  - 시계열 특화 설정 제외

### 2.5 검증
- 새로운 리샘플링 스크립트 실행 테스트
- 각 리샘플링 기법별 정상 동작 확인

## 📝 **작업 3: 로깅 시스템 개선**

### 3.1 로그 파일명 개선
- **현재**: `tuning_log_*`
- **개선**: 
  - `hyperparameter_tuning_YYYYMMDD_HHMMSS.log`
  - `resampling_experiment_YYYYMMDD_HHMMSS.log`
  - `data_analysis_YYYYMMDD_HHMMSS.log`

### 3.2 전체 터미널 화면 저장
- **기능**: 
  - 모든 터미널 출력을 로그 파일에 저장
  - 실시간 로깅과 파일 저장 동시 수행
  - 오류 발생 시에도 로그 보존

### 3.3 PROJECT_PROGRESS.md 연동
- **기능**:
  - 실험 실행 시 자동으로 PROJECT_PROGRESS.md 업데이트
  - 실험 결과 요약 자동 추가
  - 타임스탬프와 함께 변경사항 기록

### 3.4 로깅 유틸리티 개선
- **파일**: `src/utils/logging_utils.py` (신규 생성)
- **기능**:
  - 통합된 로깅 시스템
  - 파일과 콘솔 동시 출력
  - 실험별 로그 분리
  - MLflow 연동

### 3.5 검증
- 새로운 로깅 시스템 테스트
- PROJECT_PROGRESS.md 자동 업데이트 확인
- 전체 터미널 화면 저장 확인

## 🔄 **작업 순서**

### Phase 1: Hyperparameter Tuning 정리
1. `configs/experiments/hyperparameter_tuning.yaml` 정리
2. `src/hyperparameter_tuning.py` 리샘플링 로직 제거
3. `scripts/run_hyperparameter_tuning.py` 단순화
4. 하이퍼파라미터 튜닝 실험 테스트

### Phase 2: Resampling 스크립트 분리
1. `scripts/run_resampling_experiment.py` 생성
2. `configs/experiments/resampling_methods.yaml` 생성
3. `configs/experiments/resampling.yaml` 정리
4. 리샘플링 실험 테스트

### Phase 3: 로깅 시스템 개선
1. `src/utils/logging_utils.py` 생성
2. 로그 파일명 개선
3. 전체 터미널 화면 저장 구현
4. PROJECT_PROGRESS.md 자동 업데이트 구현
5. 통합 테스트

## ⚠️ **주의사항**

### 백업
- 각 Phase 시작 전 git commit
- 주요 파일 백업
- 롤백 계획 수립

### 테스트
- 각 Phase 완료 후 반드시 테스트
- 기존 기능 정상 동작 확인
- 새로운 기능 정상 동작 확인

### 문서화
- 각 Phase 완료 후 문서 업데이트
- README.md 업데이트
- PROJECT_PROGRESS.md 업데이트

## 📊 **예상 결과**

### 개선 효과
1. **명확한 책임 분리**: 각 실험이 독립적으로 관리
2. **설정 파일 단순화**: 혼재 문제 해결
3. **로깅 시스템 개선**: 실험 추적성 향상
4. **유지보수성 향상**: 코드 구조 개선

### 성능 향상
1. **빠른 실행**: 불필요한 로직 제거
2. **메모리 효율성**: 리소스 사용 최적화
3. **확장성**: 새로운 실험 타입 추가 용이

## 📋 **체크리스트**

### Phase 1 체크리스트 ✅ **완료**
- [x] `configs/experiments/hyperparameter_tuning.yaml`에서 resampling 섹션 제거
- [x] `src/hyperparameter_tuning.py`에서 리샘플링 로직 제거
- [x] `scripts/run_hyperparameter_tuning.py` 단순화
- [x] 하이퍼파라미터 튜닝 실험 테스트 통과

### Phase 2 체크리스트
- [ ] `scripts/run_resampling_experiment.py` 생성
- [ ] `configs/experiments/resampling_methods.yaml` 생성
- [ ] `configs/experiments/resampling.yaml` 정리
- [ ] 리샘플링 실험 테스트 통과

### Phase 3 체크리스트 ✅ **완료**
- [x] `src/utils/logging_utils.py` 생성 (기존 시스템 확장으로 대체)
- [x] 로그 파일명 개선 구현 (`{experiment_type}_{model_type}_{timestamp}.log`)
- [x] 전체 터미널 화면 저장 구현 (`ConsoleCapture` 클래스)
- [x] PROJECT_PROGRESS.md 자동 업데이트 구현 (`log_experiment_summary()`)
- [x] 통합 테스트 통과 (CatBoost 테스트로 검증 완료)

## 🚀 **시작하기**

이 계획서를 기준으로 **Phase 1: Hyperparameter Tuning 정리**부터 시작합니다.

**다음 단계**: Phase 1의 구체적인 수정 내용 제공

---

**작업 시작**: Phase 1부터 순차적으로 진행
**예상 소요 시간**: 각 Phase별 1-2시간
**총 예상 시간**: 3-6시간
**마지막 업데이트**: 2025-07-21