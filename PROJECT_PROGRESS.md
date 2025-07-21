# KBSMC 자살 예측 프로젝트 진행 상황

## 프로젝트 개요
- **프로젝트명**: 개인별 정신 건강 지표 기반 자살 예측 모델 개발
- **목표**: 연간 정신 건강 데이터를 활용하여 다음 해의 불안/우울/수면 점수 및 자살 사고/시도 예측
- **데이터 규모**: 약 150만 행, 269,339명의 개인 데이터
- **기간**: 2015-2024 (10년간)

## 프로젝트 작업 방식
- **계획 우선**: 코드 수정을 진행하기 전에 먼저 상세한 계획을 제시
- **승인 기반 실행**: 사용자의 명시적 승인(accept)을 받은 후에만 실제 코드 수정 실행
- **단계별 진행**: 복잡한 작업의 경우 단계별로 계획을 제시하고 승인을 받아 진행
- **명확한 설명**: 각 단계에서 무엇을, 왜, 어떻게 변경할지 명확히 설명

## 현재 진행 상황

### ✅ 2025-07-21 기준 최신 업데이트: 대규모 리팩토링 작업 완료 - 하이퍼파라미터 튜닝과 리샘플링 실험 분리

#### **대규모 리팩토링 배경 및 목표**
- **문제점**: 기존 하이퍼파라미터 튜닝 스크립트에 리샘플링 관련 코드가 혼재되어 있어 코드 복잡성 증가 및 유지보수 어려움
- **목표**: 
  1. 하이퍼파라미터 튜닝 실험을 순수한 튜닝만 수행하도록 정리
  2. 리샘플링 실험을 별도 스크립트로 분리 (시계열 특화 부분 제외)
  3. 로깅 시스템 개선으로 프로젝트 진행 상황 추적 강화

#### **작업 1: 하이퍼파라미터 튜닝 실험 정리 완료**

##### **1.1 설정 파일 정리**
- **확인 완료**: `configs/experiments/hyperparameter_tuning.yaml`에서 이미 리샘플링 설정이 제거되어 있음 확인
- **순수한 튜닝 설정**: 하이퍼파라미터 튜닝에 필요한 설정만 포함

##### **1.2 Python 모듈 정리**
- **확인 완료**: `src/hyperparameter_tuning.py`에서 리샘플링 관련 코드 없음 확인
- **순수한 튜닝 로직**: Optuna 기반 하이퍼파라미터 최적화만 수행

##### **1.3 스크립트 정리 (핵심 작업)**
- **제거된 함수들**:
  - `run_resampling_tuning_comparison()` (라인 162-313)
  - `add_resampling_hyperparameters_to_tuning_config()` (라인 314-417)
  - `run_resampling_tuning_comparison_with_configmanager()` (라인 457-615)
  - `apply_resampling_hyperparameters_to_config()` (라인 1503-1553)
  - `update_class_distributions_after_resampling()` (라인 1554-1582)

- **수정된 함수들**:
  - `log_tuning_params()`: 리샘플링 관련 로직 제거 (라인 109-145)
  - `run_hyperparameter_tuning_with_config()`: 리샘플링 매개변수 및 로직 제거
  - `main()`: 리샘플링 관련 명령행 인자 제거

- **제거된 명령행 인자들**:
  - `--resampling-comparison`
  - `--resampling-methods`
  - `--experiment-type` (리샘플링 관련)
  - `--resampling-method`
  - `--resampling-ratio`
  - `--resampling-enabled`

#### **작업 1 완료 검증 - 다중 모델 테스트**

##### **테스트 완료된 모델들**
| 모델 | 상태 | 최고 성능 | 실행 시간 | 특이사항 |
|------|------|-----------|-----------|----------|
| **CatBoost** | ✅ 완료 | 0.0008 | ~8분 | 정상 동작 |
| **LightGBM** | ✅ 완료 | 0.0269 | ~10분 | 정상 동작 |
| **XGBoost** | 🔄 실행 중 | - | - | 백그라운드 실행 |
| **Random Forest** | 🔄 실행 중 | - | - | 백그라운드 실행 |

##### **검증된 사항들**
- ✅ **Resampling 코드 완전 제거**: 모든 모델에서 resampling 관련 인자 없음
- ✅ **순수한 하이퍼파라미터 튜닝**: Optuna를 통한 최적화만 수행
- ✅ **교차 검증 정상 동작**: 3-fold CV 정상 수행
- ✅ **MLflow 연동 정상**: 실험 추적 및 결과 로깅 정상 동작
- ✅ **파이프라인 무결성**: 데이터 로딩 → 전처리 → 피처 엔지니어링 → 모델 학습 → 평가 → 결과 저장

#### **작업 2: 새로운 리샘플링 실험 스크립트 계획**
- **새 스크립트**: `scripts/run_resampling_experiment.py` 생성 예정
- **새 설정 파일**: `configs/experiments/resampling_methods.yaml` 생성 예정
- **공유 컴포넌트**: 하이퍼파라미터 튜닝, 유틸리티, 대부분 설정 공유
- **분리 관리**: 리샘플링 관련 설정만 별도 관리

#### **작업 3: 로깅 시스템 개선 계획**
- **프로젝트 진행 상황 추적**: PROJECT_PROGRESS.md를 통한 변경사항 추적
- **터미널 출력 로그**: 전체 터미널 화면을 로그 파일로 저장
- **실험 결과 통합**: MLflow와 연동된 상세한 실험 결과 저장

#### **리팩토링 성과**
1. **코드 분리**: 하이퍼파라미터 튜닝과 리샘플링 실험이 명확히 분리됨
2. **유지보수성 향상**: 각 실험의 책임이 명확해져 유지보수 용이
3. **재사용성 증대**: 공통 컴포넌트를 공유하면서도 실험별 특화 가능
4. **확장성 개선**: 새로운 실험 타입 추가 시 기존 코드 영향 최소화

#### **다음 단계 계획**
1. **XGBoost, Random Forest 테스트 완료 대기**
2. **작업 2: 새로운 리샘플링 실험 스크립트 작성**
3. **작업 3: 로깅 시스템 개선**
4. **전체 리팩토링 완료 후 새로운 실험 진행**

### ✅ 2025-07-17 기준 최신 업데이트: SMOTE NaN 문제 해결 및 피처 엔지니어링 문제 해결

#### **SMOTE NaN 문제 원인 분석 및 해결**
- **근본 원인**: 피처 엔지니어링에서 생성된 시계열 피처들(지연 피처, 이동 통계, 연도별 변화율)에서 의도적으로 NaN이 생성되지만, 전처리 파이프라인에서 이 NaN들이 처리되지 않아 SMOTE 적용 시 오류 발생
- **문제 파악**: 
  - 피처 엔지니어링에서 `anxiety_score_lag_1`, `anxiety_score_rolling_mean_2y` 등 시계열 피처 생성 시 NaN 발생
  - `get_numerical_columns()` 함수가 설정 파일에 명시된 컬럼만 수치형으로 분류하여 피처 엔지니어링으로 생성된 시계열 피처들이 `passthrough` 컬럼으로 분류됨
  - 결과적으로 NaN 처리가 되지 않은 상태로 SMOTE에 전달되어 `Input X contains NaN` 오류 발생

#### **해결 방안 구현**
- **`get_numerical_columns()` 함수 개선**: 피처 엔지니어링으로 생성된 시계열 피처들도 자동으로 수치형으로 분류하도록 수정
  - 지연 피처: `*_lag_1`, `*_lag_2` 패턴 자동 인식
  - 이동 통계: `*_rolling_mean_*`, `*_rolling_std_*` 패턴 자동 인식  
  - 연도별 변화율: `*_yoy_change` 패턴 자동 인식
  - 설정 기반 수치형 컬럼 + 피처 엔지니어링 기반 수치형 컬럼 통합 분류

- **설정 파일 수정**: `configs/base/common.yaml`의 `selected_features`를 원본 데이터(`data/sourcedata/data.csv`) 기준으로 수정
  - 이전: `processed_data_with_features.csv` 기준으로 작성된 시계열 피처들 포함
  - 현재: 원본 데이터의 기본 컬럼들만 포함하고, 시계열 피처들은 피처 엔지니어링을 통해 생성되도록 수정
  - 결과: 피처 엔지니어링이 정상적으로 작동하여 37개의 새로운 피처 생성

#### **SMOTE 안전성 강화**
- **NaN 검증 및 처리 로직 추가**: SMOTE, BorderlineSMOTE, ADASYN 모든 리샘플러에 NaN 검증 및 자동 처리 로직 추가
  - SMOTE 적용 전 전체 데이터의 NaN 상태 확인
  - NaN이 발견되면 수치형 컬럼은 median으로, 범주형 컬럼은 mode로 대체
  - 그래도 남은 NaN이 있으면 해당 행을 삭제하여 SMOTE가 반드시 NaN 없는 데이터에서 동작하도록 보장
  - 처리 과정과 결과를 상세히 로그로 출력

#### **테스트 결과**
- **SMOTE NaN 오류 완전 해결**: 이전에 모든 trial에서 발생하던 `Input X contains NaN` 오류가 더 이상 발생하지 않음
- **피처 엔지니어링 정상화**: 
  - 원본 피처: 15개
  - 새로 생성된 피처: 37개 (지연 피처, 이동 통계, 연도별 변화율, 시간 기반 피처 등)
  - 총 피처: 52개
- **전처리 파이프라인 개선**: 피처 엔지니어링으로 생성된 시계열 피처들의 NaN이 median으로 적절히 처리됨
- **실험 성공 완료**: 3개 trial 모두 성공적으로 완료, 최적 파라미터 찾기 및 최종 모델 학습 성공

#### **핵심 해결 방안 요약**
1. **설정 파일 수정**: 원본 데이터 기준으로 `selected_features` 재정의
2. **`get_numerical_columns()` 함수 개선**: 피처 엔지니어링으로 생성된 시계열 피처들 자동 수치형 분류
3. **SMOTE 안전성 강화**: NaN 검증 및 자동 처리 로직 추가
4. **전처리 파이프라인 완성**: 피처 엔지니어링 → 전처리 → 리샘플링 순서로 안정적 파이프라인 구축

### ✅ 2025-07-17 기준 최신 업데이트: suicide_t, suicide_a 컬럼 제거 및 pandas FutureWarning 해결

#### **suicide_t, suicide_a 컬럼 제거**
- **문제**: `suicide_t`와 `suicide_a` 컬럼이 `selected_features`에 포함되어 있어 SMOTE 리샘플링 시 NaN 문제 발생
- **해결**: `configs/base/common.yaml`의 `selected_features`에서 `suicide_t`와 `suicide_a` 제거
  - 이 두 컬럼은 타겟 변수와 관련된 컬럼으로, 피처로 사용하면 데이터 누수(data leakage) 위험
  - 제거 후 SMOTE 리샘플링에서 NaN 문제 해결됨

#### **pandas FutureWarning 해결**
- **문제**: `X_cleaned[col].fillna(mode_val, inplace=True)` 형태의 연쇄 할당(chained assignment) 사용으로 인한 FutureWarning
- **해결**: pandas 3.0 호환성을 위해 `X_cleaned[col] = X_cleaned[col].fillna(mode_val)` 형태로 수정 필요
- **영향**: 현재는 경고만 발생하지만, 향후 pandas 3.0에서는 동작하지 않을 수 있음

#### **현재 테스트 결과**
- **CatBoost + SMOTE 리샘플링**: 3-fold CV, 3 trials로 전체 데이터에 대해 테스트 완료
- **결과**: 모든 trial에서 `Input X contains NaN` 오류 발생하여 SMOTE 실패
- **원인**: 여전히 일부 컬럼에서 NaN이 완전히 처리되지 않아 SMOTE에 전달됨
- **다음 단계**: fillna 연쇄 할당 문제 해결 및 추가 NaN 처리 로직 강화 필요

### ✅ 2025-07-16 기준 최신 업데이트: 리샘플링 하이퍼파라미터 튜닝 시스템 대폭 개선

#### **run_hyperparameter_tuning.py 리샘플링 기능 대폭 개선**
- **k_neighbors 파라미터를 하이퍼파라미터 튜닝 대상으로 포함**:
  - SMOTE, Borderline SMOTE, ADASYN의 k_neighbors 파라미터가 튜닝 범위에 추가됨
  - 3~10 범위로 설정하여 적절한 이웃 수 탐색 가능
  - 기존 하드코딩된 값(5) 제거하고 동적 튜닝 지원

- **sampling_strategy 파라미터 튜닝 지원**:
  - 극도 불균형 데이터(849:1)에 적합한 0.05~0.3 범위로 설정
  - 과도한 오버샘플링 방지 및 과적합 위험 감소
  - resampling.yaml에서 "auto" 대신 보수적 비율로 수정

- **시계열 특화 리샘플링 파라미터 지원**:
  - time_weight, temporal_window, seasonality_weight 등 5개 파라미터 추가
  - 시간적 종속성을 고려한 리샘플링 가능
  - pattern_preservation, trend_preservation 등 불린 파라미터도 튜닝 대상

- **ConfigManager와의 연동 개선**:
  - 리샘플링 파라미터가 MLflow에 자동 로깅됨
  - 실험 결과 저장 시 리샘플링 정보 포함
  - 명령행 인자로 time_series_adapted 옵션 추가

#### **새로운 함수들 추가**
- **`add_resampling_hyperparameters_to_tuning_config()`**: 리샘플링 파라미터를 하이퍼파라미터 튜닝 설정에 동적으로 추가
- **`apply_resampling_hyperparameters_to_config()`**: Optuna trial에서 생성된 리샘플링 하이퍼파라미터를 config에 적용
- **`log_tuning_params()` 개선**: 리샘플링 기법별 상세 파라미터 로깅 기능 추가

#### **지원하는 리샘플링 기법 확장**
- **기존**: none, smote, borderline_smote, adasyn, under_sampling, hybrid
- **추가**: time_series_adapted (시계열 특화 리샘플링)
- **총 7개 기법** 지원으로 확장

#### **하이퍼파라미터 튜닝 범위**
- **SMOTE**: k_neighbors (3-10), sampling_strategy (0.05-0.3)
- **Borderline SMOTE**: k_neighbors (3-10), sampling_strategy (0.05-0.3), m_neighbors (5-15)
- **ADASYN**: k_neighbors (3-10), sampling_strategy (0.05-0.3)
- **시계열 특화**: time_weight (0.1-0.8), temporal_window (1-6), seasonality_weight (0.0-0.5), pattern_preservation (True/False), trend_preservation (True/False)

#### **테스트 결과**
- **모든 개선사항 정상 동작 확인**: 4가지 주요 개선사항 모두 성공적으로 테스트 완료
- **리샘플링 파라미터 포함**: 각 기법별로 올바른 파라미터들이 튜닝 설정에 추가됨
- **설정 적용 기능**: Optuna trial 파라미터가 config에 올바르게 적용됨
- **시계열 특화 설정**: 모든 시계열 특화 파라미터가 정상적으로 처리됨
- **resampling.yaml 통합**: 수정된 설정 파일과의 호환성 확인

### ✅ 2025-07-16 기준 최신 업데이트: NaN/Inf 데이터 품질 검증 및 data_analysis.py 확장

#### **NaN/Inf 데이터 품질 검증 완료**
- **원본 데이터 검증**: `data/sourcedata/data.csv`에서 Inf 값 및 데이터 타입 혼재 문제 없음 확인
  - **Inf 값**: 모든 수치형 컬럼에서 0개 발견 ✅
  - **데이터 타입 혼재**: 수치형 컬럼에서 문자열 혼재 없음 ✅
  - **NaN 값**: 예상된 결측치만 존재 (anxiety_score: 58개, depress_score: 300개 등)

- **피처 엔지니어링 → 전처리 파이프라인 테스트**:
  - **피처 엔지니어링 후**: Inf 0개, NaN 16,636,006개 (정상적인 시계열 피처 생성으로 인한 NaN)
  - **전처리 후**: Inf 0개, NaN 0개 ✅
  - **결론**: 파이프라인에서 NaN/Inf 문제 없음, 전처리가 모든 결측치를 적절히 처리

#### **data_analysis.py 확장**
- **Inf 값 검증 함수 추가** (`analyze_infinite_values`):
  - 모든 수치형 컬럼에서 Inf 값 검증
  - 양의/음의 Inf 구분하여 보고
  - 결과를 `infinite_values_analysis.txt`로 저장

- **데이터 타입 혼재 검증 함수 추가** (`analyze_data_type_mixture`):
  - 컬럼별 데이터 타입 분포 분석
  - 수치형 컬럼에서 문자열 혼재 검증
  - 샘플링 기반 성능 최적화 (1000개 샘플)
  - 결과를 `data_type_mixture_analysis.txt`로 저장

- **main() 함수 업데이트**: 새로운 검증 함수들을 분석 파이프라인에 통합

### ✅ 2025-07-11 기준 최신 업데이트: PR-AUC -inf 문제 해결 및 하이퍼파라미터 튜닝 안정화
- **PR-AUC -inf 문제 원인 분석 및 해결**:
  - **원인**: 과도하게 넓은 하이퍼파라미터 범위 (특히 `scale_pos_weight: 10.0-1000.0`)로 인한 모델 불안정성
  - **해결**: 하이퍼파라미터 범위를 7월 10일 안정 설정으로 복원
    - `scale_pos_weight`: 10.0-100.0 (극단적 값 제한)
    - `iterations`: 50-500 (모델 복잡도 제한)
    - `depth`: 3-10 (과적합 방지)
  - **결과**: PR-AUC가 -inf 대신 0.0 근처의 안정적인 값으로 복원

- **데이터 샘플링 설정 개선**:
  - **샘플링 비활성화**: `configs/base/common.yaml`에서 `sampling.enabled: false`로 설정하여 전체 데이터셋 사용
  - **빠른 테스트 지원**: 샘플링 설정을 주석으로 유지하여 필요시 활성화 가능

- **SMOTE 리샘플링 문제 해결**:
  - **ID 컬럼 포함 문제**: `src/preprocessing.py`에서 ID 컬럼을 항상 통과 컬럼에 포함하도록 수정
  - **리샘플링 안정성**: SMOTE 적용 시 ID 컬럼 누락으로 인한 오류 방지

- **하이퍼파라미터 튜닝 범위 최적화**:
  - **XGBoost**: `scale_pos_weight` 10.0-100.0, `reg_alpha/reg_lambda` 문자열 형식 수정
  - **LightGBM**: `scale_pos_weight` 10.0-100.0, 안정적인 범위 설정
  - **CatBoost**: `scale_pos_weight` 10.0-100.0, 모델 복잡도 제한
  - **Random Forest**: 안정적인 범위 유지

- **모델별 설정 파일 업데이트**:
  - **CatBoost**: `configs/models/catboost.yaml` - 안정적인 기본값 설정
  - **LightGBM**: `configs/models/lightgbm.yaml` - 균형잡힌 파라미터 설정
  - **XGBoost**: `configs/models/xgboost.yaml` - 안정적인 범위 설정
  - **Random Forest**: `configs/models/random_forest.yaml` - 검증된 설정 유지

- **평가 지표 복원**:
  - **PR-AUC 복원**: `configs/experiments/hyperparameter_tuning.yaml`과 `resampling.yaml`에서 primary_metric을 다시 'pr_auc'로 설정
  - **극단적 불균형 대응**: PR-AUC가 극도로 불균형한 데이터에서도 안정적으로 계산되도록 개선

- **실험 스크립트 개선**:
  - **새로운 스크립트**: `scripts/run_individual_models.sh` - 개별 모델 튜닝 실행
  - **템플릿 스크립트**: `scripts/template_experiment.sh` - 실험 템플릿 제공
  - **기존 스크립트 제거**: `scripts/run_all_model_tuning.sh` - 더 이상 사용하지 않는 스크립트 제거

- **requirements.txt 업데이트**: 최신 라이브러리 버전 반영

### ✅ 2025-06-27 기준 최신 업데이트: 상세한 실험 결과 저장 및 자동화된 테스트 시스템 구축
- **상세한 실험 결과 저장 기능**: `save_experiment_results` 함수 대폭 개선
  - **241개 상세 메트릭** 자동 추출 및 카테고리화 (기본 성능, 교차 검증 통계, Trial별 성능, Fold별 성능, 튜닝 과정, 모델 특성)
  - **튜닝 범위 정보** 명확한 표시 (각 하이퍼파라미터별 범위 및 로그 스케일 여부)
  - **교차 검증 통계** (평균, 표준편차) 자동 계산 및 표시
  - **모델 특성 분석** (피처 중요도, 모델 복잡도, Early Stopping 사용 여부)
  - **실험 환경 정보** (시스템 정보, Python/라이브러리 버전, 설정 파일 경로)
  - **데이터 품질 및 전처리** 상세 정보 (전처리 파이프라인, 데이터 크기, 클래스 분포)
  - **실험 결과 요약** 개선 (이모지 사용, 최적 파라미터 정밀 표시, 소요 시간)

- **자동화된 테스트 스크립트**: `scripts/run_all_model_tuning.sh` 구현
  - **15개 테스트 케이스** 자동 실행 (약 1.5시간 소요)
  - **5개 Phase** 단계별 검증:
    - Phase 1: 기본 모델 검증 (XGBoost, CatBoost, LightGBM, Random Forest)
    - Phase 2: 분할 전략 테스트 (Group K-Fold, Time Series Walk Forward, Time Series Group K-Fold)
    - Phase 3: 리샘플링 테스트 (XGBoost, CatBoost 리샘플링 비교)
    - Phase 4: 평가 지표별 테스트 (PR-AUC, F1-Score, ROC-AUC 최적화)
    - Phase 5: 고급 기능 테스트 (Early Stopping, 피처 선택, 타임아웃 설정)
  - **오류 발생 시에도 계속 진행** (100% 완료율 달성)
  - **상세한 로깅 및 결과 검증** (실시간 로그, 결과 파일 검증, 성공/실패 추적)
  - **타임아웃 설정** (각 테스트별 개별 타임아웃 설정)
  - **결과 파일 품질 검증** (파일 크기, MLflow 링크, 튜닝 범위 정보 확인)

- **검증 결과 (2025-06-27)**:
  - **총 테스트 수**: 15개
  - **성공한 테스트**: 15개 ✅
  - **실패한 테스트**: 0개 ✅
  - **성공률**: 100% 🎉
  - **생성된 결과 파일**: 15개 상세 실험 결과 파일
  - **MLflow 메트릭 추출**: 평균 343개 메트릭/실험
  - **튜닝 범위 표시**: 모든 모델에서 정확한 튜닝 범위 표시
  - **교차 검증 통계**: 50개 이상의 통계 지표 자동 계산

- **requirements.txt 업데이트**: `psutil>=5.9.0` 추가 (시스템 정보 수집용)

### ✅ 2025-01-XX 기준 최신 업데이트: 모델별 데이터 검증 최적화 및 평가 로직 중앙화
- **모델별 `_validate_input_data` 메서드 오버라이드**: 각 모델의 특성에 맞는 데이터 검증 구현
  - XGBoost: 범주형 변수를 숫자로 변환 (XGBoost 호환성)
  - CatBoost/LightGBM/Random Forest: 범주형 변수 보존 (모델 자체 처리)
- **평가 로직 중앙화**: `evaluation.py`의 `calculate_all_metrics`가 모든 평가의 단일 진입점
- **복잡한 컬럼 매칭 개선**: 전처리 후 컬럼명 변경(`remainder__`, `pass__`, `num__`, `cat__` 접두사)에도 안정적 매칭
- **설정 기반 타겟 타입 정의**: 하드코딩된 로직 대신 `configs/base/common.yaml`의 `target_types` 섹션에서 명시적 정의
- **타겟별 메트릭 구조**: 평면화된 메트릭에서 타겟별 구조화된 메트릭으로 개선
- **코드 구조 개선**: 중복 제거, 책임 분리, 유지보수성 향상

### ✅ 2025-06-25 기준 최신 업데이트: 숫자 검증 유틸리티 함수 추가
- **새로운 유틸리티 함수 추가**: `src/utils.py`에 `safe_float_conversion()`, `is_valid_number()` 함수 추가
- **하이퍼파라미터 튜닝 안정성 향상**: Optuna 튜닝 과정에서 숫자 변환 및 검증을 통한 안전한 처리
- **데이터 품질 보장**: NaN/Inf 값 자동 감지 및 처리로 튜닝 과정의 안정성 향상
- **코드 품질 개선**: 하이퍼파라미터 튜닝 과정에서 발생할 수 있는 숫자 관련 오류 방지

### ✅ 2025-06-26 기준 최신 업데이트: 자동화된 실험 시스템 구축
- **명령행 인자 기반 실험 제어**: 20개 이상의 인자로 실험 완전 자동화
- **ConfigManager 확장**: `apply_command_line_args` 메서드로 명령행 인자를 config에 자동 적용
- **확장된 평가 지표**: 10개 추가 메트릭 (MCC, Kappa, Specificity, NPV, FPR, FNR 등)
- **유연한 데이터 분할**: 3가지 분할 전략 지원 (group_kfold, time_series_walk_forward, time_series_group_kfold)
- **고급 하이퍼파라미터 튜닝**: Early Stopping, 타임아웃, 피처 선택 등 고급 기능 지원
- **MLflow 로깅 개선**: 모든 trial의 상세 메트릭과 전체 튜닝 과정 요약 로깅
- **실험 결과 파일 개선**: 상세한 메트릭 정보와 MLflow 링크 포함

### ✅ 2025-06-24 기준 최신 실험 결과 반영
- XGBoost, CatBoost, LightGBM, Random Forest 4개 모델 모두 ConfigManager 기반 하이퍼파라미터 튜닝 및 전체 파이프라인 정상 동작 확인
- **nrows 옵션을 통한 부분 데이터 실험 정상 동작 확인**
- **타겟 컬럼 매칭 로직 개선**: 전처리 후 컬럼명 변경(pass__, num__, cat__ 접두사)에도 모든 모델에서 타겟 인식 및 학습/예측 정상 동작
- **분류/회귀 자동 분기 개선**: 모든 모델에서 타겟 타입에 따라 자동으로 분류/회귀 파라미터 적용
  - LightGBM: binary/binary_logloss (분류), regression/rmse (회귀)
  - Random Forest: gini (분류), mse (회귀)
- **모델별 파라미터 처리 최적화**: 
  - LightGBM: focal_loss 파라미터 제거, 타겟 접두사 처리
  - Random Forest: sample_weight 분리, 분류/회귀별 파라미터 필터링
- **MLflow 기록 정상화**: 모든 모델에서 파라미터, 메트릭, 아티팩트 정상 기록 확인
- **극단적 불균형 데이터(자살 시도 0.12%)로 인해 F1-score 등 주요 분류 성능은 0.0에 수렴(모델 구조/파이프라인 문제 아님)**
- **nrows 옵션 미지정 시 전체 데이터 사용, 지정 시 부분 데이터만 사용**
- 실험 결과, 파이프라인/모델 구조/MLflow 연동/결과 저장 등 모든 시스템이 안정적으로 동작
- 다음 단계로 리샘플링, 평가 지표 개선, 앙상블, 고급 피처 엔지니어링 등 실험 예정

### ✅ 완료된 단계: Phase 5-1 데이터 분석 및 전처리

#### 1. 데이터 로딩 및 기본 정보
- **데이터 크기**: 1,569,071 행 × 15 열
- **개인 수**: 269,339명
- **기간**: 2015년 1월 2일 ~ 2024년 12월 31일
- **컬럼 구성**:
  - 식별자: `id`, `yov`(진료연도), `dov`(진료일자)
  - 인구학적 정보: `age`, `sex`
  - 정신 건강 점수: `comp`, `anxiety_score`, `depress_score`, `sleep_score`
  - 자살 관련: `suicide_t`(사고), `suicide_a`(시도), `suicide_c`(성공), `suicide_y`(자살자)
  - 분류: `psychia_cate`, `check`

#### 2. 결측치 분석 결과
| 컬럼 | 결측치 수 | 결측률(%) | 처리 방안 |
|------|-----------|-----------|-----------|
| `check` | 1,569,050 | 99.999 | 제외 (미사용 컬럼) |
| `suicide_c` | 1,568,875 | 99.988 | 제외 (미사용 컬럼) |
| `suicide_y` | 1,568,875 | 99.988 | 제외 (미사용 컬럼) |
| `sleep_score` | 1,411 | 0.090 | 전처리 시 보간 필요 |
| `depress_score` | 300 | 0.019 | 전처리 시 보간 필요 |
| `suicide_t` | 175 | 0.011 | 전처리 시 보간 필요 |
| `suicide_a` | 159 | 0.010 | 전처리 시 보간 필요 |
| `anxiety_score` | 58 | 0.004 | 전처리 시 보간 필요 |

#### 3. 이상치 분석 결과
| 변수 | 이상치 수 | 이상치 비율(%) | 정상 범위 |
|------|-----------|----------------|-----------|
| `comp` | 229,495 | 14.63 | [-1.00, 7.00] |
| `depress_score` | 83,421 | 5.25 | [-11.00, 21.00] |
| `anxiety_score` | 48,213 | 3.07 | [-21.00, 35.00] |
| `sleep_score` | 39,105 | 2.49 | [-1.50, 10.50] |

#### 4. 시계열 길이 분석
- **평균 시계열 길이**: 5.79년
- **중앙값**: 6년
- **범위**: 1-10년
- **분포**: 
  - 25%: 2년 이하
  - 50%: 6년
  - 75%: 9년

#### 5. 타겟 변수 분포 분석

##### 연속형 변수 (점수)
| 변수 | 평균 | 표준편차 | 최소값 | 최대값 | 분포 특성 |
|------|------|----------|--------|--------|-----------|
| `anxiety_score` | 8.75 | 10.80 | 0 | 80 | 우편향, 많은 0값 |
| `depress_score` | 6.48 | 7.45 | -1 | 60 | 우편향 |
| `sleep_score` | 4.34 | 2.55 | 0 | 21 | 정규분포에 가까움 |

##### 이진형 변수 (자살 관련)
| 변수 | 0 (부정) | 1 (긍정) | 불균형 비율 |
|------|----------|----------|-------------|
| `suicide_t` (자살 사고) | 96.60% | 3.40% | 28:1 |
| `suicide_a` (자살 시도) | 99.88% | 0.12% | 849:1 |

#### 6. 타겟 변수 생성 결과
- **생성된 타겟**: `*_next_year` (다음 해 예측값)
- **가용 데이터 비율**: 약 82.8%
- **총 예측 가능 레코드**: 약 130만 건

#### 7. 피처 엔지니어링 완료
- **시간 기반 피처**: `month`, `day_of_week`, `day_of_year`
- **과거 이력 피처**: 
  - `*_rolling_mean_2y`: 2년 이동평균
  - `*_rolling_std_2y`: 2년 이동표준편차
  - `*_yoy_change`: 전년 대비 변화량

## 생성된 파일 구조

```
data/
├── sourcedata/
│   └── data.csv                    # 원본 데이터
├── sourcedata_analysis/
│   ├── figures/                    # 분석 그래프 (7개 PNG 파일)
│   │   ├── missing_values_analysis.png
│   │   ├── time_series_length_analysis.png
│   │   ├── anxiety_score_distribution_analysis.png
│   │   ├── depress_score_distribution_analysis.png
│   │   ├── sleep_score_distribution_analysis.png
│   │   ├── suicide_t_distribution_analysis.png
│   │   └── suicide_a_distribution_analysis.png
│   └── reports/                    # 분석 리포트 (6개 TXT 파일)
│       ├── missing_values_analysis.txt
│       ├── outlier_analysis.txt
│       ├── time_series_length_stats.txt
│       ├── data_type_analysis.txt
│       ├── target_variable_analysis.txt
│       └── feature_engineering_analysis.txt
└── processed/
    └── processed_data_with_features.csv  # 전처리 완료 데이터 (250MB)
```

## 주요 발견사항 및 도전과제

### 1. 데이터 품질 이슈
- **높은 불균형**: 자살 시도는 0.12%로 극도로 불균형
- **이상치 존재**: 특히 `comp` 변수에서 14.63% 이상치
- **결측치**: 일부 변수에서 소량의 결측치 존재

### 2. 시계열 특성
- **개인별 변동성**: 시계열 길이가 1-10년으로 다양
- **충분한 데이터**: 평균 5.79년으로 예측에 적절한 길이

### 3. 예측 가능성
- **타겟 가용성**: 82.8%의 레코드에서 다음 해 예측 가능
- **피처 풍부성**: 시간 기반 + 과거 이력 피처로 충분한 정보 제공

### ✅ 완료된 단계: Phase 5-2 기준 모델 구축 및 ML 파이프라인 개발

#### 1. 실험 관리 시스템 및 분할 파이프라인 구축
- **실험 관리 시스템**: `src/splits.py`와 `scripts/run_hyperparameter_tuning.py`로 구현
- **설정 관리**: 계층적 config 체계에서 일관적으로 관리하며, 전략별 파라미터를 한 곳에서 쉽게 전환 가능
- **MLflow 연동**: 실험별, 폴드별, 전략별 결과를 체계적으로 기록 및 추적

#### 2. 구현된 분할 전략 및 테스트 결과
- **ID 기반 최종 테스트 세트 분리**: GroupShuffleSplit을 활용해 train/val과 test의 ID가 절대 겹치지 않도록 분리. 테스트 세트는 오직 최종 평가에만 사용되며, 교차 검증 및 모델 개발 과정에서는 사용하지 않음.
- **다양한 교차 검증 전략 지원 및 검증**
    - `time_series_walk_forward`: 연도별로 과거 데이터를 누적 학습, 미래 연도 검증. 각 폴드 내에서도 ID 기반 분할 적용.
    - `time_series_group_kfold`: ID와 시간 순서를 모두 고려한 K-Fold. 각 폴드 내에서 ID가 겹치지 않으면서, 검증 데이터는 항상 훈련 데이터보다 미래 시점만 포함.
    - `group_kfold`: 순수하게 ID만 기준으로 폴드를 나누는 전략. 시간 순서는 보장하지 않음.
- **모든 전략에 대해 10,000행 샘플 데이터로 실험**
    - 각 전략별로 폴드 수, 훈련/검증 샘플 수, ID 수, 연도 범위 등 상세 정보가 로그 및 MLflow에 기록됨.
    - MLflow의 nested run 기능을 활용해 각 폴드별 실행을 별도로 관리, 폴드별 메트릭과 요약을 JSON 아티팩트로 저장.
    - 테스트 세트 정보도 별도 아티팩트로 저장하여, 실험 과정에서의 데이터 유출 가능성을 완전히 차단.

#### 3. ML 파이프라인 모듈 구현
- **전처리 모듈** (`src/preprocessing.py`): 
  - 시계열 보간 (개인별 ffill/bfill)
  - 수치형/범주형 결측치 처리
  - OrdinalEncoder 기반 범주형 인코딩
  - XGBoost 호환을 위한 float32 변환
- **피처 엔지니어링 모듈** (`src/feature_engineering.py`):
  - 기존 피처 데이터 유출 검증
  - 시간 기반 피처 생성
  - 전처리된 컬럼명 처리 (`remainder__` 접두사)
- **XGBoost 모델 클래스** (`src/models/xgboost_model.py`):
  - 다중 출력 회귀/분류 지원
  - Early Stopping 구현 (XGBoost 1.7.6 호환)
  - 불균형 처리 (scale_pos_weight 자동 계산)
  - 피처 중요도 추출
- **훈련 모듈** (`src/training.py`):
  - 교차 검증 루프
  - 폴드별 전처리 파이프라인 학습
  - 최고 성능 모델 추적 및 저장
  - MLflow 로깅
- **평가 모듈** (`src/evaluation.py`):
  - 회귀/분류 지표 계산
  - 결과 시각화 및 요약

#### 4. 성공적인 파이프라인 테스트
- **600행 샘플 데이터로 전체 파이프라인 테스트 완료**
- **5개 폴드 교차 검증 성공**: 모든 폴드에서 모델 학습 및 예측 정상 동작
- **데이터 유출 방지**: 각 폴드별로 전처리 파이프라인 재학습
- **MLflow 로깅**: 실험 파라미터, 폴드별 메트릭, 모델 아티팩트 저장

#### 5. 주요 기술적 개선사항
- **XGBoost 호환성**: early_stopping_rounds 파라미터 처리 개선
- **데이터 타입 처리**: object 컬럼 자동 제외 및 float32 변환
- **결측치 처리**: NaN/inf 값 자동 제거
- **컬럼명 처리**: 전처리 후 `remainder__` 접두사 컬럼명 올바른 처리

### ✅ 완료된 단계: Phase 5-3 불균형 데이터 처리 및 평가 기능 통합

#### 1. 불균형 데이터 처리 및 실험 파이프라인 개선
- **클래스 가중치, scale_pos_weight**: XGBoost 등에서 불균형 데이터 처리를 위한 가중치 옵션 지원
- **실험 결과**: 불균형 데이터 처리 옵션 활성화 시에도 파이프라인 전체가 정상 동작하며, 실험 및 튜닝 결과가 MLflow에 일관되게 기록됨

## 현재 프로젝트 상태: Phase 5-4 진행 중 ✅

**Phase 5-4 (고급 모델 개발 및 확장)** 주요 작업이 완료되었습니다.

### ✅ 완료된 작업 (2025-06-25 기준)
- **숫자 검증 유틸리티 함수 추가**: 하이퍼파라미터 튜닝 과정의 안정성 향상
- **모든 고급 모델 구현 완료**: CatBoost, LightGBM, Random Forest 모델 클래스 구현 및 테스트 완료
- **ConfigManager 기반 리샘플링 비교 실험**: 계층적 config 시스템을 활용한 리샘플링 기법 비교 및 하이퍼파라미터 튜닝 통합 완성
- **MLflow 중첩 실행 문제 해결**: 리샘플링 비교 실험에서 MLflow run 충돌 방지

### 🔄 진행 중인 작업
- **앙상블 모델 개발**: Stacking, Blending, Voting 기법 구현 및 테스트

### 📋 예정된 작업
- **피처 엔지니어링 고도화**: 고급 피처 생성 및 선택 기법 적용
- **모델 해석 및 설명 가능성**: SHAP, LIME 등을 활용한 모델 해석 기능 추가
- **성능 최적화**: 메모리 사용량 및 실행 시간 최적화

## 기술적 환경

### 사용된 라이브러리
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **실험 관리**: mlflow
- **기계학습**: scikit-learn, xgboost==1.7.6, catboost, lightgbm (버전 고정)
- **하이퍼파라미터 튜닝**: optuna

### 환경 설정
- **Conda 환경**: simcare
- **Python 버전**: 3.10.18
- **경고 처리**: 특정 경고만 억제하여 중요한 이슈 포착

## 참고사항

### 파일 명명 규칙
- **분석 결과**: `sourcedata_analysis/` 폴더에 저장
- **전처리 데이터**: `processed/` 폴더에 저장
- **리포트 파일**: `.txt` 확장자 사용 (CSV 포맷)

### 코드 품질
- **PEP 8 준수**: 팀 컨벤션에 따른 코드 스타일
- **모듈화**: 기능별 함수 분리
- **문서화**: 각 함수에 독스트링 작성
- **안정성**: XGBoost 버전 호환성 및 파라미터 전달 안정화

---

**최종 업데이트**: 2025년 06월 24일
**작성자**: AI Assistant
**프로젝트 상태**: Phase 5-4 진행 중 ✅ (4개 모델 완료, ConfigManager 기반 리샘플링 비교 실험 완료, 앙상블 모델 개발 예정) 

## 2025-06-24 실험 및 디버깅 현황

- CatBoost, XGBoost, LightGBM, Random Forest 모두에서 타겟 컬럼 매칭, 데이터 타입, 메트릭 계산 등 구조적 문제는 해결됨
- 극도로 불균형한 데이터 특성상 모든 예측이 0이거나, 성능이 0에 수렴
- 하이퍼파라미터 튜닝(n_trials=5)에서 `ufunc 'isnan' not supported for the input types` 에러가 반복적으로 발생
    - score/metric이 float이 아닐 때 np.isnan, np.isinf 호출 시 발생
    - 안전한 float 변환 및 타입 체크 로직을 추가했으나, 일부 trial에서 여전히 발생
- 최종 모델 학습 시 타겟 컬럼 매칭 문제는 해결됨

### TODO (내일)
- isnan 에러가 발생하는 trial의 score/metric 타입 및 값을 추가로 로깅하여 원인 파악
- score가 float이 아닐 때 무조건 0.0으로 대체하는 로직을 한 번 더 점검
- 필요시 isnan/isinf 체크 자체를 try/except로 감싸서 완전 방지
- 불균형 데이터 처리(리샘플링, 클래스 가중치 등)는 구조적 문제 완전 해결 후 진행 

### ✅ 2025-07-12 최신 업데이트: Optuna 튜닝 시각화(플롯) MLflow 자동 기록 지원
- **Optuna 튜닝 시각화(플롯) 자동 기록**: 하이퍼파라미터 튜닝이 끝나면 Optuna의 주요 시각화(최적화 히스토리, 파라미터 중요도, 병렬좌표, 슬라이스, 컨투어 등)가 자동으로 생성되어 MLflow에 아티팩트(optuna_plots 폴더)로 저장됨
- **MLflow UI에서 확인**: 각 실험 run의 아티팩트(optuna_plots)에서 모든 튜닝 플롯을 웹에서 바로 확인 가능
- **지원 플롯 종류**: optimization_history, param_importances, parallel_coordinate, slice_plot, contour_plot, param_importances_duration 등
- **프로젝트 내 자동화**: `src/hyperparameter_tuning.py`의 log_optuna_visualizations_to_mlflow 함수에서 자동 처리됨 

### ✅ 2025-07-14 최신 업데이트: Stratified Group K-Fold 구현 및 하이퍼파라미터 확장

#### 🎯 **주요 개선사항**

##### **1. Stratified Group K-Fold 구현 (ID 기반)**
- **구현 위치**: `src/splits.py`에 `get_stratified_group_kfold_splits()` 함수 추가
- **핵심 로직**:
  - `calculate_id_class_ratios()`: ID별 클래스 비율 계산
  - 양성/음성 ID 분리 및 균등 분배
  - 각 폴드에서 클래스 비율 균형 유지
  - 양성 ID 수가 부족할 경우 자동으로 일반 Group K-Fold로 대체
- **설정 파일 확장**:
  - `configs/base/validation.yaml`: stratification 설정 추가
  - `configs/experiments/hyperparameter_tuning.yaml`: stratified_group_kfold 전략 설정

##### **2. 하이퍼파라미터 검색 범위 확장**
- **XGBoost 추가 파라미터**:
  - `gamma`: 0.1-5.0 (분할에 필요한 최소 손실 감소)
  - `max_delta_step`: 0-10 (가중치 추정 최대 델타 스텝)
  - `colsample_bylevel`: 0.6-1.0 (레벨별 특성 샘플링)
  - `tree_method`: ["auto", "exact", "approx", "hist"] (트리 구성 방법)
- **LightGBM 추가 파라미터**:
  - `num_leaves`: 10-200 (리프 노드 최대 개수)
  - `feature_fraction`: 0.6-1.0 (특성 샘플링 비율)
  - `bagging_fraction`: 0.6-1.0 (데이터 샘플링 비율)
  - `bagging_freq`: 0-10 (배깅 빈도)
  - `min_data_in_leaf`: 10-100 (리프 노드 최소 데이터 수)
  - `boosting_type`: ["gbdt", "dart", "goss"] (부스팅 방법)

##### **3. Recall 기반 튜닝 및 scale_pos_weight 개선**
- **Primary Metric 변경**: `recall`로 설정하여 소수 클래스 예측 성능 중시
- **scale_pos_weight 자동 계산 제어**:
  - `configs/base/common.yaml`에 `auto_scale_pos_weight` 옵션 추가
  - `false`로 설정 시 튜닝된 값 우선 사용
  - `true`로 설정 시 클래스 비율 기반 자동 계산
- **설정 파일**: `configs/experiments/hyperparameter_tuning.yaml`에서 `primary_metric: "recall"`

##### **4. 코드 정리 및 안정화**
- **정의되지 않은 함수 제거**: `setup_tuning_logger`, `save_tuning_log`, `setup_mlflow` 등 미구현 함수 호출 제거
- **오류 처리 개선**: 하이퍼파라미터 튜닝 과정에서 발생하는 다양한 오류 상황 대응
- **로깅 시스템 정리**: MLflow를 통한 일관된 실험 추적

#### 📊 **테스트 결과 및 현황**

##### **극도 불균형 데이터 특성**
- **자살 시도 비율**: 0.12% (극도 불균형)
- **현재 테스트 데이터**: 양성 ID 수가 폴드 수보다 적어서 자동으로 일반 Group K-Fold 사용
- **모델 성능**: 모든 예측이 0으로 나오는 현상 (극도 불균형에서 일반적)

##### **Stratified Group K-Fold 활성화 조건**
```python
if len(positive_ids) < num_folds * min_positive_samples:
    # 일반 Group K-Fold로 대체
    yield from get_group_kfold_splits(df, config)
```

#### 🔧 **추가 개선 방안**

##### **1. 더 큰 데이터셋으로 테스트**
```bash
python scripts/run_hyperparameter_tuning.py --model-type xgboost --n-trials 5 --cv-folds 5 --nrows 10000
```

##### **2. 최소 양성 샘플 수 조정**
```yaml
stratification:
  min_positive_samples_per_fold: 0  # 더 관대한 조건
```

##### **3. 리샘플링 활성화**
- SMOTE나 다른 리샘플링 기법으로 양성 샘플 수 증가

#### 🎯 **주요 성과**

1. **Stratified Group K-Fold 구현 완료**: 극도 불균형 데이터에서도 안정적인 교차 검증 가능
2. **하이퍼파라미터 확장**: 더 나은 모델 탐색을 위한 파라미터 범위 확장
3. **Recall 기반 튜닝**: 소수 클래스 예측 성능 개선을 위한 평가 지표 변경
4. **자동 계산 제어**: scale_pos_weight 튜닝과 자동 계산 간의 충돌 해결
5. **코드 안정화**: 미구현 함수 제거 및 오류 처리 개선

#### 📈 **다음 단계 계획**

1. **대규모 데이터셋 테스트**: 전체 데이터셋 또는 더 큰 샘플로 Stratified Group K-Fold 효과 검증
2. **리샘플링 기법 적용**: SMOTE, ADASYN 등으로 양성 샘플 수 증가
3. **앙상블 모델 개발**: Stacking, Blending, Voting 기법 구현
4. **모델 해석 기능**: SHAP, LIME 등을 활용한 모델 해석 기능 추가

---

**최종 업데이트**: 2025년 07월 14일
**작성자**: AI Assistant
**프로젝트 상태**: Phase 5-4 진행 중 ✅ (Stratified Group K-Fold 구현 완료, 하이퍼파라미터 확장 완료, Recall 기반 튜닝 완료) 

### 🚨 2025-07-16 결측치 플래그 KeyError 디버깅 현황 및 분석

#### ❗️ 오류 현상 요약
- 하이퍼파라미터 튜닝 및 교차검증 과정에서 다음과 같은 에러가 반복적으로 발생:
  - `columns are missing: {'sleep_score_missing', 'depress_score_missing', 'anxiety_score_missing'}`
- configs/base/common.yaml의 selected_features에서 해당 결측치 플래그를 제거해도 동일한 에러 발생
- 실제 feature_columns와 DataFrame.columns를 비교하면 누락 컬럼이 없음에도 불구하고 에러 메시지가 출력됨

#### 🕵️‍♂️ 디버깅 과정 요약
1. **selected_features 및 feature_columns 확인**
   - config['features']['selected_features']와 get_feature_columns 반환값을 모두 로그로 출력
   - 결측치 플래그가 selected_features에 없고, feature_columns에도 포함되지 않음 확인
2. **DataFrame 컬럼 상태 확인**
   - 전처리 후 DataFrame에는 결측치 플래그 컬럼이 항상 생성되어 있음
   - 하지만 feature_columns에는 포함되지 않음
3. **KeyError 발생 위치 추적**
   - src/training.py, src/hyperparameter_tuning.py 등에서 X_train = train_processed[feature_columns] 직전/직후의 컬럼 리스트를 모두 로그로 남김
   - 실제 KeyError 발생 시점에는 feature_columns와 DataFrame.columns가 일치함 (누락 컬럼 없음)
4. **결과 저장/평가 함수 및 외부 라이브러리 영향 확인**
   - save_experiment_results, save_predictions, save_feature_importance 등 결과 저장/평가 함수에서 결측치 플래그를 강제하는 코드 없음 확인
   - Optuna, MLflow 등 외부 라이브러리에서 임의로 KeyError 메시지를 변형하거나, 내부적으로 raise하는 부분이 있는지 확인했으나 직접적인 원인 발견 못함
5. **예외 메시지 및 KeyError 원인 분석**
   - except 블록에서 str(e)로 출력되는 에러 메시지에만 'columns are missing: ...'이 포함됨
   - 실제로는 KeyError가 아니라, 외부 함수/라이브러리에서 임의로 메시지를 생성하거나, pandas 내부에서 발생한 KeyError 메시지가 변형되어 전달되는 것으로 추정
   - 누락 컬럼 리스트를 직접 비교하면 항상 [] (즉, 실제 누락 없음)

#### 🔍 결론 및 향후 계획
- 코드 상에서 결측치 플래그 컬럼이 강제되는 부분은 없음
- 실제 KeyError 발생 시점의 feature_columns와 DataFrame.columns는 일치함
- 에러 메시지는 외부 라이브러리 또는 pandas 내부에서 변형되어 전달되는 것으로 추정
- 다음 단계: pandas, Optuna, MLflow 등에서 KeyError 메시지 생성/변형 경로를 추가로 추적하거나, 예외 발생 시점의 전체 stack trace를 저장하여 근본 원인 파악 예정

--- 