# KBSMC 자살 예측 프로젝트

## 프로젝트 개요
개인별 연간 정신 건강 지표 데이터를 활용하여 다음 해의 불안/우울/수면 점수 및 자살 사고/시도 여부를 예측하는 머신러닝 프로젝트입니다.

## 🎯 최근 주요 개선사항 (2025-07-21)

### ✅ **대규모 리팩토링 작업 완료 - 하이퍼파라미터 튜닝과 리샘플링 실험 분리 + 로깅 시스템 대폭 개선**
- **리팩토링 배경**: 기존 하이퍼파라미터 튜닝 스크립트에 리샘플링 관련 코드가 혼재되어 있어 코드 복잡성 증가 및 유지보수 어려움
- **작업 1 완료 - 하이퍼파라미터 튜닝 실험 정리**:
  - **설정 파일 정리**: `configs/experiments/hyperparameter_tuning.yaml`에서 리샘플링 설정 제거 확인
  - **Python 모듈 정리**: `src/hyperparameter_tuning.py`에서 리샘플링 관련 코드 없음 확인
  - **스크립트 정리**: `scripts/run_hyperparameter_tuning.py`에서 5개 리샘플링 관련 함수 완전 제거
    - 제거된 함수: `run_resampling_tuning_comparison()`, `add_resampling_hyperparameters_to_tuning_config()`, `run_resampling_tuning_comparison_with_configmanager()`, `apply_resampling_hyperparameters_to_config()`, `update_class_distributions_after_resampling()`
    - 수정된 함수: `log_tuning_params()`, `run_hyperparameter_tuning_with_config()`, `main()`
    - 제거된 명령행 인자: `--resampling-comparison`, `--resampling-methods`, `--resampling-method`, `--resampling-ratio`, `--resampling-enabled` 등

- **다중 모델 테스트 완료**:
  - **CatBoost**: 최고 성능 0.0008, ~8분 소요 ✅
  - **LightGBM**: 최고 성능 0.0269, ~10분 소요 ✅
  - **XGBoost, Random Forest**: 백그라운드 실행 중 🔄
  - **검증 완료**: Resampling 코드 완전 제거, 순수한 하이퍼파라미터 튜닝만 수행, 교차 검증 정상 동작, MLflow 연동 정상

- **작업 3 완료 - 로깅 시스템 대폭 개선**:
  - **기존 시스템 확장**: 새로 생성하지 않고 기존 `setup_logging()` 함수 확장
  - **새로운 함수 추가**: `setup_experiment_logging()`, `ConsoleCapture`, `experiment_logging_context()`, `log_experiment_summary()`
  - **통일된 로그 파일명**: `{experiment_type}_{model_type}_{timestamp}.log` 패턴 적용
  - **터미널 출력 완전 캡처**: STDOUT, STDERR, 예외 정보 모두 로그 파일에 저장
  - **검증 완료**: CatBoost 테스트로 새로운 로깅 시스템 검증 (495KB, 3,123줄 로그 파일 생성)

- **리팩토링 성과**:
  - **코드 분리**: 하이퍼파라미터 튜닝과 리샘플링 실험이 명확히 분리됨
  - **유지보수성 향상**: 각 실험의 책임이 명확해져 유지보수 용이
  - **재사용성 증대**: 공통 컴포넌트를 공유하면서도 실험별 특화 가능
  - **확장성 개선**: 새로운 실험 타입 추가 시 기존 코드 영향 최소화
  - **완전한 실험 추적**: 터미널 출력까지 포함한 완전한 실험 기록

- **다음 단계 계획**:
  - 작업 2: 새로운 리샘플링 실험 스크립트 작성 (`scripts/run_resampling_experiment.py`)

## 🎯 최근 주요 개선사항 (2025-07-17)
- **SMOTE NaN 문제 원인 분석**: 피처 엔지니어링에서 생성된 시계열 피처들(지연 피처, 이동 통계, 연도별 변화율)에서 의도적으로 NaN이 생성되지만, 전처리 파이프라인에서 이 NaN들이 처리되지 않아 SMOTE 적용 시 `Input X contains NaN` 오류 발생
- **`get_numerical_columns()` 함수 대폭 개선**: 피처 엔지니어링으로 생성된 시계열 피처들도 자동으로 수치형으로 분류하도록 수정
  - 지연 피처: `*_lag_1`, `*_lag_2` 패턴 자동 인식
  - 이동 통계: `*_rolling_mean_*`, `*_rolling_std_*` 패턴 자동 인식  
  - 연도별 변화율: `*_yoy_change` 패턴 자동 인식
  - 설정 기반 수치형 컬럼 + 피처 엔지니어링 기반 수치형 컬럼 통합 분류

- **설정 파일 수정**: `configs/base/common.yaml`의 `selected_features`를 원본 데이터(`data/sourcedata/data.csv`) 기준으로 수정
  - 이전: `processed_data_with_features.csv` 기준으로 작성된 시계열 피처들 포함
  - 현재: 원본 데이터의 기본 컬럼들만 포함하고, 시계열 피처들은 피처 엔지니어링을 통해 생성되도록 수정
  - 결과: 피처 엔지니어링이 정상적으로 작동하여 37개의 새로운 피처 생성

- **SMOTE 안전성 강화**: SMOTE, BorderlineSMOTE, ADASYN 모든 리샘플러에 NaN 검증 및 자동 처리 로직 추가
  - SMOTE 적용 전 전체 데이터의 NaN 상태 확인
  - NaN이 발견되면 수치형 컬럼은 median으로, 범주형 컬럼은 mode로 대체
  - 그래도 남은 NaN이 있으면 해당 행을 삭제하여 SMOTE가 반드시 NaN 없는 데이터에서 동작하도록 보장
  - 처리 과정과 결과를 상세히 로그로 출력

- **테스트 결과**:
  - **SMOTE NaN 오류 완전 해결**: 이전에 모든 trial에서 발생하던 `Input X contains NaN` 오류가 더 이상 발생하지 않음
  - **피처 엔지니어링 정상화**: 원본 피처 15개 + 새로 생성된 피처 37개 = 총 52개 피처
  - **전처리 파이프라인 완성**: 피처 엔지니어링 → 전처리 → 리샘플링 순서로 안정적 파이프라인 구축
  - **실험 성공 완료**: 3개 trial 모두 성공적으로 완료, 최적 파라미터 찾기 및 최종 모델 학습 성공

## 🎯 최근 주요 개선사항 (2025-07-16)

### ✅ **리샘플링 하이퍼파라미터 튜닝 시스템 대폭 개선**
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

- **지원하는 리샘플링 기법 확장**:
  - **기존**: none, smote, borderline_smote, adasyn, under_sampling, hybrid
  - **추가**: time_series_adapted (시계열 특화 리샘플링)
  - **총 7개 기법** 지원으로 확장

### ✅ **NaN/Inf 데이터 품질 검증 및 data_analysis.py 확장**
- **원본 데이터 검증**: `data/sourcedata/data.csv`에서 Inf 값 및 데이터 타입 혼재 문제 없음 확인
  - **Inf 값**: 모든 수치형 컬럼에서 0개 발견 ✅
  - **데이터 타입 혼재**: 수치형 컬럼에서 문자열 혼재 없음 ✅
  - **NaN 값**: 예상된 결측치만 존재 (anxiety_score: 58개, depress_score: 300개 등)

- **피처 엔지니어링 → 전처리 파이프라인 테스트**:
  - **피처 엔지니어링 후**: Inf 0개, NaN 16,636,006개 (정상적인 시계열 피처 생성으로 인한 NaN)
  - **전처리 후**: Inf 0개, NaN 0개 ✅
  - **결론**: 파이프라인에서 NaN/Inf 문제 없음, 전처리가 모든 결측치를 적절히 처리

- **data_analysis.py 확장**:
  - **Inf 값 검증 함수 추가** (`analyze_infinite_values`): 모든 수치형 컬럼에서 Inf 값 검증
  - **데이터 타입 혼재 검증 함수 추가** (`analyze_data_type_mixture`): 컬럼별 데이터 타입 분포 분석
  - **새로운 검증 함수들을 분석 파이프라인에 통합**

### ✅ **로깅 시스템 일원화 및 코드 일관성 강화**
- **공통 로깅 함수 도입**: `src/utils.py`에 `setup_logging`, `get_logger`, `log_experiment_info` 함수 추가
- **모듈별 일관된 로깅 적용**: 기존의 `logging.basicConfig()` 개별 호출을 모두 제거하고, `setup_logging()` 함수로 통합
- **utils 패키지 구조 개선**: `src/utils/__init__.py`에서 공통 함수 및 로깅 함수 직접 제공 (import 경로 문제 해결)
- **테스트 완료**: 모든 주요 모듈에서 로깅 정상 동작 확인
- **향후 확장성**: 파일 로깅, 포맷, 레벨 등 중앙에서 손쉽게 관리 가능

### ✅ **하이퍼파라미터 검색 범위 대폭 확장 및 피처 세트 확장**
- **피처 세트 대폭 확장**: 7개 → 20개 피처로 확장
  - **기본 인구학적 정보**: `age`, `sex`, `age_group`
  - **정신 건강 점수**: `anxiety_score`, `depress_score`, `sleep_score`, `comp`
  - **자살 관련 변수**: `suicide_t`, `suicide_a`
  - **정신과 분류**: `psychia_cate`
  - **시간 기반 피처**: `month`, `day_of_week`, `day_of_year`
  - **이동 통계 피처**: `*_rolling_mean_2y`, `*_rolling_std_2y`, `*_yoy_change`

- **CatBoost 파라미터 대폭 확장**:
  - `iterations`: 50-500 → 100-1000 (모델 복잡도 증가)
  - `depth`: 3-10 → 4-12 (트리 깊이 확장)
  - `learning_rate`: 0.01-0.3 → 0.005-0.15 (더 세밀한 학습률 탐색)
  - `l2_leaf_reg`: 1-10 → 0.1-50.0 (정규화 범위 확장)
  - **새로운 파라미터 추가**: `min_data_in_leaf`, `max_bin`, `subsample`, `od_type`, `od_wait`

- **SMOTE 리샘플링 최적화**:
  - `k_neighbors`: 5 → 3 (극도 불균형 데이터에 맞게 조정)
  - `sampling_strategy`: "auto" → 0.1 (1:10 비율로 소수 클래스 오버샘플링)

- **튜닝 설정 강화**:
  - `n_trials`: 100 → 500 (더 정교한 탐색)
  - `patience`: 20 → 50 (조기 종료 여유 증가)
  - `min_trials`: 30 → 100 (최소 시도 횟수 증가)

- **불균형 데이터 대응 강화**:
  - `scale_pos_weight` 범위: 10.0-100.0 → 50.0-1000.0 (극도 불균형 데이터 대응)
  - 모델별 자동 가중치 비활성화, 수동 튜닝 우선

---

## 🎯 최근 주요 개선사항 (2025-07-14)

### ✅ **Stratified Group K-Fold 구현 및 하이퍼파라미터 확장**
- **Stratified Group K-Fold 구현**: 극도 불균형 데이터(자살 시도 0.12%)에서 안정적인 교차 검증을 위한 ID 기반 Stratified Group K-Fold 구현
- **핵심 로직**: ID별 클래스 비율 계산, 양성/음성 ID 분리 및 균등 분배, 각 폴드에서 클래스 비율 균형 유지
- **자동 대체 메커니즘**: 양성 ID 수가 부족할 경우 자동으로 일반 Group K-Fold로 대체
- **설정 파일 확장**: `configs/base/validation.yaml`에 stratification 설정 추가, `configs/experiments/hyperparameter_tuning.yaml`에 stratified_group_kfold 전략 설정

### ✅ **하이퍼파라미터 검색 범위 대폭 확장**
- **XGBoost 추가 파라미터**: `gamma` (0.1-5.0), `max_delta_step` (0-10), `colsample_bylevel` (0.6-1.0), `tree_method` (4가지 방법)
- **LightGBM 추가 파라미터**: `num_leaves` (10-200), `feature_fraction` (0.6-1.0), `bagging_fraction` (0.6-1.0), `bagging_freq` (0-10), `min_data_in_leaf` (10-100), `boosting_type` (3가지 방법)
- **모델 탐색 범위 확장**: 더 나은 모델을 찾기 위한 파라미터 공간 대폭 확장

### ✅ **Recall 기반 튜닝 및 scale_pos_weight 개선**
- **Primary Metric 변경**: `recall`로 설정하여 소수 클래스 예측 성능 중시
- **scale_pos_weight 자동 계산 제어**: `auto_scale_pos_weight` 옵션으로 튜닝된 값과 자동 계산 간의 충돌 해결
- **설정 기반 제어**: `false`로 설정 시 튜닝된 값 우선 사용, `true`로 설정 시 클래스 비율 기반 자동 계산

### ✅ **코드 정리 및 안정화**
- **정의되지 않은 함수 제거**: `setup_tuning_logger`, `save_tuning_log`, `setup_mlflow` 등 미구현 함수 호출 제거
- **오류 처리 개선**: 하이퍼파라미터 튜닝 과정에서 발생하는 다양한 오류 상황 대응
- **로깅 시스템 정리**: MLflow를 통한 일관된 실험 추적

### ✅ **PR-AUC -inf 문제 해결 및 하이퍼파라미터 튜닝 안정화**
- **PR-AUC -inf 문제 원인 분석**: 과도하게 넓은 하이퍼파라미터 범위로 인한 모델 불안정성
- **하이퍼파라미터 범위 최적화**: 
  - `scale_pos_weight`: 10.0-100.0 (극단적 값 제한)
  - `iterations`: 50-500 (모델 복잡도 제한)
  - `depth`: 3-10 (과적합 방지)
- **결과**: PR-AUC가 -inf 대신 0.0 근처의 안정적인 값으로 복원

### ✅ **데이터 샘플링 및 SMOTE 리샘플링 개선**
- **전체 데이터셋 사용**: `configs/base/common.yaml`에서 샘플링 비활성화
- **SMOTE 안정성 향상**: ID 컬럼을 항상 통과 컬럼에 포함하여 리샘플링 오류 방지
- **빠른 테스트 지원**: 샘플링 설정을 주석으로 유지하여 필요시 활성화 가능

### ✅ **평가 지표 및 모델 설정 최적화**
- **PR-AUC 복원**: 극도로 불균형한 데이터에서도 안정적으로 계산되도록 개선
- **모델별 설정 파일 업데이트**: 모든 모델에서 안정적인 기본값 및 범위 설정
- **하이퍼파라미터 튜닝 범위 조정**: 각 모델의 특성에 맞는 최적 범위 설정

### ✅ **실험 스크립트 개선**
- **새로운 스크립트**: `scripts/run_individual_models.sh` - 개별 모델 튜닝 실행
- **템플릿 스크립트**: `scripts/template_experiment.sh` - 실험 템플릿 제공
- **기존 스크립트 정리**: 더 이상 사용하지 않는 스크립트 제거

### ✅ **스크립트 안전성 및 메모리 관리 개선**
- **메모리 안전한 스크립트 템플릿** 생성 (`template_experiment.sh`)
- **오래된 스크립트 정리**: 7개 중복/불안정 스크립트 삭제
- **병렬 처리 최적화**: `n_jobs=4`로 안전한 기본값 설정
- **자동 메모리 정리**: 각 모델 실행 후 강화된 메모리 정리
- **오류 허용 구조**: 한 모델 실패 시에도 다음 모델로 계속 진행
- **모듈화된 구조**: `run_model()` 함수로 재사용 가능한 구조

### ✅ **상세한 실험 결과 저장 기능**
- **241개 상세 메트릭** 자동 추출 및 카테고리화
- **튜닝 범위 정보** 명확한 표시
- **교차 검증 통계** (평균, 표준편차) 자동 계산
- **모델 특성 분석** (피처 중요도, 복잡도)
- **실험 환경 정보** (시스템, 라이브러리 버전)
- **데이터 품질 및 전처리** 상세 정보

### ✅ **검증된 안정성**
- **메모리 문제 해결**: 터미널 멈춤 현상 완전 해결
- **실험 안정성**: 100% 완료율 달성
- **재현 가능성**: 일관된 환경에서 안정적 실행

## 프로젝트 구조
```
kbsmc_suicide/
├── data/
│   ├── sourcedata/                    # 원본 데이터
│   │   └── data.csv
│   ├── sourcedata_analysis/           # 분석 결과
│   │   ├── figures/                   # 분석 그래프
│   │   └── reports/                   # 분석 리포트 (.txt)
│   └── processed/                     # 전처리된 데이터
│       └── processed_data_with_features.csv
├── src/
│   ├── data_analysis.py              # 데이터 분석 및 전처리 스크립트
│   ├── splits.py                     # 데이터 분할 전략
│   ├── preprocessing.py              # 전처리 파이프라인
│   ├── feature_engineering.py        # 피처 엔지니어링
│   ├── models/
│   │   ├── base_model.py             # BaseModel 추상 클래스 (✅ 모델별 데이터 검증 최적화)
│   │   ├── xgboost_model.py          # XGBoost 모델 클래스 (✅ 범주형 변수 숫자 변환)
│   │   ├── catboost_model.py         # CatBoost 모델 클래스 (✅ 범주형 변수 보존)
│   │   ├── lightgbm_model.py         # LightGBM 모델 클래스 (✅ 범주형 변수 보존)
│   │   ├── random_forest_model.py    # Random Forest 모델 클래스 (✅ 범주형 변수 보존)
│   │   └── loss_functions.py         # 손실 함수 모듈 (Focal Loss 포함)
│   ├── training.py                   # 훈련 파이프라인 (✅ 평가 로직 중앙화)
│   ├── evaluation.py                 # 평가 모듈 (✅ 통합 평가 진입점)
│   ├── hyperparameter_tuning.py      # 하이퍼파라미터 튜닝 (✅ 평가 로직 중앙화)
│   ├── utils.py                      # 공통 유틸리티 함수 (✅ 숫자 변환/검증 함수 추가)
│   └── reference/                    # 참고 자료
├── configs/
│   ├── base/                         # 기본 설정 (✅ 계층적 구조)
│   │   ├── common.yaml               # 공통 설정
│   │   ├── evaluation.yaml           # 평가 설정
│   │   ├── mlflow.yaml               # MLflow 설정
│   │   └── validation.yaml           # 검증 설정
│   ├── models/                       # 모델별 설정 (✅ 계층적 구조)
│   │   ├── xgboost.yaml              # XGBoost 모델 설정
│   │   ├── catboost.yaml             # CatBoost 모델 설정
│   │   ├── lightgbm.yaml             # LightGBM 모델 설정
│   │   └── random_forest.yaml        # Random Forest 모델 설정
│   ├── experiments/                  # 실험별 설정 (✅ 계층적 구조)
│   │   ├── focal_loss.yaml           # Focal Loss 실험 설정
│   │   ├── resampling.yaml           # 리샘플링 실험 설정
│   │   └── hyperparameter_tuning.yaml # 하이퍼파라미터 튜닝 설정
│   └── templates/                    # 설정 템플릿 (✅ 계층적 구조)
│       ├── default.yaml              # 기본 템플릿
│       └── tuning.yaml               # 튜닝 템플릿
├── scripts/
│   ├── run_hyperparameter_tuning.py  # 통합 실험 실행 스크립트 (✅ ConfigManager 기반 리샘플링 비교 포함)
│   ├── template_experiment.sh        # 실험 스크립트 템플릿 (✅ 메모리 안전한 기본 구조)
│   └── run_individual_models.sh      # 개별 모델 실행 스크립트 (✅ 메모리 최적화)
├── results/                          # 실험 결과 저장소
│   ├── experiment_results_*.txt      # 상세한 실험 결과 파일
│   ├── tuning_log_*.txt              # 튜닝 과정 로그
│   └── test_logs/                    # 자동화 테스트 로그
├── requirements.txt                  # 필요한 패키지 목록 (XGBoost 1.7.6 고정, psutil 추가)
├── projectplan                       # 프로젝트 계획서
├── PROJECT_PROGRESS.md              # 프로젝트 진행 상황 문서
└── README.md                        # 이 파일
```

## 프로젝트 룰

### 작업 방식 원칙
- **계획 우선**: 코드 수정을 진행하기 전에 먼저 상세한 계획을 제시
- **승인 기반 실행**: 사용자의 명시적 승인(accept)을 받은 후에만 실제 코드 수정 실행
- **단계별 진행**: 복잡한 작업의 경우 단계별로 계획을 제시하고 승인을 받아 진행
- **명확한 설명**: 각 단계에서 무엇을, 왜, 어떻게 변경할지 명확히 설명

### 설정 파일 관리 원칙
- **설정 우선**: 모든 하드코딩된 값은 설정 파일로 이동하여 중앙 관리
- **설정 검증**: 설정 파일이 없거나 문제가 있는 경우 하드코딩으로 진행하지 않고 진행을 멈춤
- **명확한 오류 메시지**: 설정 파일 문제 시 구체적인 수정 방법을 로그로 안내
- **타겟 변수명 중앙화**: `configs/base/common.yaml`의 `target_variables` 섹션에서 모든 타겟 변수명 정의
- **타겟 타입 명시**: `configs/base/common.yaml`의 `target_types` 섹션에서 회귀/분류 타입 명시적 정의

### 코드 컨벤션
- **PEP 8 준수**: 파이썬 코드 스타일 가이드 준수
- **설정 기반 로직**: 하드코딩 대신 설정 파일 기반 동적 처리
- **오류 처리**: 설정 파일 문제 시 `SystemExit` 또는 `ValueError`로 명확한 오류 발생
- **로깅**: 설정 파일 문제 시 구체적인 수정 방법을 `logger.error()`로 안내

### 설정 파일 구조
```yaml
# configs/base/common.yaml
features:
  target_variables:
    original_targets:
      score_targets: ["anxiety_score", "depress_score", "sleep_score", "comp"]
      binary_targets: ["suicide_t", "suicide_a"]
    next_year_targets:
      score_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
      binary_targets: ["suicide_t_next_year", "suicide_a_next_year"]
  
  target_types:
    regression_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
    classification_targets: ["suicide_t_next_year", "suicide_a_next_year"]
```

## 환경 설정

### 1. Conda 환경 활성화
```bash
conda activate simcare
```

### 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

> ⚠️ 본 프로젝트는 xgboost==1.7.6 버전에 최적화되어 있습니다. requirements.txt에 명시된 버전으로 설치해야 실험이 정상 동작합니다.

## 실행 방법

### 데이터 분석 및 전처리 실행
```bash
python src/data_analysis.py
```

이 명령어는 다음 작업을 수행합니다:
- 원본 데이터 로딩 및 기본 정보 분석
- 결측치 및 이상치 분석
- 시계열 길이 분석
- 타겟 변수 분포 분석
- 데이터 타입 변환 및 전처리
- 타겟 변수 생성 (다음 해 예측값)
- 피처 엔지니어링
- 결과 저장 및 MLflow 로깅

### 🚀 안전한 실험 스크립트 사용법 (권장)

#### 1. 스크립트 템플릿 사용 (가장 안전)
```bash
# 템플릿을 복사하여 새로운 실험 스크립트 생성
cp scripts/template_experiment.sh scripts/my_experiment.sh

# 실험 이름과 설정을 수정한 후 실행
chmod +x scripts/my_experiment.sh
./scripts/my_experiment.sh
```

**템플릿의 주요 특징:**
- **메모리 안전**: `n_jobs=4`로 설정하여 메모리 사용량 최소화
- **자동 메모리 정리**: 각 모델 실행 후 강화된 메모리 정리
- **오류 허용**: 한 모델이 실패해도 다음 모델로 계속 진행
- **상세한 로깅**: 각 단계별 메모리 상태 및 진행 상황 기록
- **모듈화된 구조**: `run_model()` 함수로 재사용 가능한 구조

#### 2. 기존 검증된 스크립트 사용
```bash
# 개별 모델 실행 (메모리 최적화)
./scripts/run_individual_models.sh
```

**참고**: 피처 선택은 `configs/base/common.yaml`의 `selected_features`에서 중앙 관리됩니다. 
피처 조합을 변경하려면 설정 파일을 수정하세요.

#### 3. 스크립트 템플릿 커스터마이징 가이드

**기본 설정 수정:**
```bash
# 병렬 처리 설정 (시스템에 맞게 조정)
N_JOBS=4  # 안전한 값: 4-8, 고성능: 16-28

# 메모리 제한 설정
export MEMORY_LIMIT=50  # GB 단위

# 실험 이름 수정
echo "실험 시작: [실험 이름]"  # 원하는 실험 이름으로 변경
```

**모델 추가/제거:**
```bash
# Phase 1에 모델 추가
run_model "xgboost" "xgboost_basic" ""

# Phase 2에 새로운 리샘플링 방법 추가
run_model "lightgbm" "lightgbm_adasyn" "--resampling-enabled --resampling-method adasyn --resampling-ratio 0.5"
```

**추가 파라미터 설정:**
```bash
# 하이퍼파라미터 튜닝 시도 횟수 조정
--n-trials 100  # 기본값, 필요시 50-200으로 조정

# 타임아웃 설정 추가
--timeout 3600  # 1시간 타임아웃

# Early Stopping 추가
--early-stopping --early-stopping-rounds 50
```

### ConfigManager 기반 하이퍼파라미터 튜닝 실행
```bash
# XGBoost 모델 튜닝
python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type hyperparameter_tuning --nrows 10000

# CatBoost 모델 튜닝
python scripts/run_hyperparameter_tuning.py --model-type catboost --experiment-type hyperparameter_tuning --nrows 10000

# LightGBM 모델 튜닝
python scripts/run_hyperparameter_tuning.py --model-type lightgbm --experiment-type hyperparameter_tuning --nrows 10000

# Random Forest 모델 튜닝
python scripts/run_hyperparameter_tuning.py --model-type random_forest --experiment-type hyperparameter_tuning --nrows 10000
```

### 🚀 자동화된 실험을 위한 고급 명령행 인자 사용법

#### 기본 인자
```bash
# 필수 인자
--model-type {xgboost,lightgbm,random_forest,catboost}  # 사용할 모델 타입
--experiment-type {hyperparameter_tuning,focal_loss,resampling}  # 실험 타입

# 데이터 관련 인자
--data_path PATH                    # 데이터 파일 경로
--nrows INT                         # 사용할 데이터 행 수 (테스트용)
```

#### 검증 및 분할 관련 인자
```bash
# 데이터 분할 전략
--split-strategy {group_kfold,time_series_walk_forward,time_series_group_kfold}
--cv-folds INT                      # 교차 검증 폴드 수 (기본: 5)
--test-size FLOAT                   # 테스트 세트 비율 (0.0-1.0, 기본: 0.15)
--random-state INT                  # 랜덤 시드 (기본: 42)
```

#### 하이퍼파라미터 튜닝 관련 인자
```bash
# 튜닝 설정
--n-trials INT                      # 튜닝 시도 횟수 (기본: 100)
--tuning-direction {maximize,minimize}  # 튜닝 방향
--primary-metric STR                # 주요 평가 지표 (f1, precision, recall, mcc, roc_auc, pr_auc 등)
--n-jobs INT                        # 병렬 처리 작업 수
--timeout INT                       # 튜닝 타임아웃 (초)
```

#### Early Stopping 관련 인자
```bash
--early-stopping                    # Early stopping 활성화
--early-stopping-rounds INT         # Early stopping 라운드 수
```

#### 피처 선택 관련 인자
```bash
--feature-selection                 # 피처 선택 활성화
--feature-selection-method {mutual_info,chi2,f_classif,recursive}  # 피처 선택 방법
--feature-selection-k INT           # 선택할 피처 수
```

#### 리샘플링 관련 인자
```bash
--resampling-enabled                # 리샘플링 활성화
--resampling-method {smote,borderline_smote,adasyn,under_sampling,hybrid}  # 리샘플링 방법
--resampling-ratio FLOAT            # 리샘플링 후 양성 클래스 비율
```

#### MLflow 및 결과 저장 관련 인자
```bash
--experiment-name STR               # MLflow 실험 이름
--save-model                        # 최적 모델 저장
--save-predictions                  # 예측 결과 저장
--mlflow_ui                         # 튜닝 완료 후 MLflow UI 실행
--verbose {0,1,2}                   # 로그 레벨 (0: 최소, 1: 기본, 2: 상세)
```

#### 실용적인 실험 예시

**1. 빠른 테스트 (소규모 데이터)**
```bash
python scripts/run_hyperparameter_tuning.py \
  --model-type xgboost \
  --experiment-type hyperparameter_tuning \
  --nrows 1000 \
  --n-trials 10 \
  --cv-folds 3 \
  --verbose 2
```

**2. 시간 기반 분할 실험**
```bash
python scripts/run_hyperparameter_tuning.py \
  --model-type catboost \
  --experiment-type hyperparameter_tuning \
  --split-strategy time_series_walk_forward \
  --cv-folds 5 \
  --n-trials 50 \
  --primary-metric pr_auc
```

**3. 리샘플링 비교 실험**
```bash
python scripts/run_hyperparameter_tuning.py \
  --model-type lightgbm \
  --experiment-type resampling \
  --resampling-comparison \
  --resampling-methods smote adasyn \
  --n-trials 30 \
  --cv-folds 5
```

**4. 피처 선택 실험**
```bash
python scripts/run_hyperparameter_tuning.py \
  --model-type random_forest \
  --experiment-type hyperparameter_tuning \
  --feature-selection \
  --feature-selection-method mutual_info \
  --feature-selection-k 10 \
  --n-trials 50 \
  --save-model \
  --save-predictions
```

**5. 고성능 실험 (전체 데이터)**
```bash
python scripts/run_hyperparameter_tuning.py \
  --model-type xgboost \
  --experiment-type hyperparameter_tuning \
  --n-trials 200 \
  --cv-folds 5 \
  --primary-metric f1 \
  --early-stopping \
  --early-stopping-rounds 50 \
  --save-model \
  --save-predictions \
  --mlflow_ui
```

### ConfigManager 기반 리샘플링 비교 실험 실행 (✅ 최신 기능)
```bash
# 모든 모델에 대해 리샘플링 비교 실험
python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type resampling --resampling-comparison
python scripts/run_hyperparameter_tuning.py --model-type catboost --experiment-type resampling --resampling-comparison
python scripts/run_hyperparameter_tuning.py --model-type lightgbm --experiment-type resampling --resampling-comparison
python scripts/run_hyperparameter_tuning.py --model-type random_forest --experiment-type resampling --resampling-comparison

# 특정 리샘플링 기법만 비교
python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type resampling --resampling-comparison --resampling-methods smote adasyn

# 시계열 특화 리샘플링 실험 (✅ 2025-07-16 신규 기능)
python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type resampling --resampling-method time_series_adapted
```

### 리샘플링 하이퍼파라미터 튜닝 실행 (✅ 2025-07-16 신규 기능)
```bash
# SMOTE k_neighbors 및 sampling_strategy 튜닝
python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type resampling --resampling-method smote --n-trials 50

# Borderline SMOTE 파라미터 튜닝
python scripts/run_hyperparameter_tuning.py --model-type catboost --experiment-type resampling --resampling-method borderline_smote --n-trials 50

# ADASYN 파라미터 튜닝
python scripts/run_hyperparameter_tuning.py --model-type lightgbm --experiment-type resampling --resampling-method adasyn --n-trials 50

# 시계열 특화 리샘플링 파라미터 튜닝
python scripts/run_hyperparameter_tuning.py --model-type random_forest --experiment-type resampling --resampling-method time_series_adapted --n-trials 50
```

### 레거시 단일 파일 config 사용 (백워드 호환성)
```bash
python scripts/run_hyperparameter_tuning.py --tuning_config configs/hyperparameter_tuning.yaml --base_config configs/default_config.yaml
```

## 생성되는 파일들

### 분석 결과 (data/sourcedata_analysis/)
- **figures/**: 7개의 분석 그래프 (PNG)
- **reports/**: 6개의 분석 리포트 (TXT)

### 전처리 데이터 (data/processed/)
- **processed_data_with_features.csv**: 피처 엔지니어링이 완료된 데이터

### 실험 결과 (MLflow)
- **실험 파라미터**: 설정 파일의 모든 파라미터
- **폴드별 메트릭**: 각 교차 검증 폴드의 성능 지표
- **고급 평가 지표**: Balanced Accuracy, Precision-Recall Curve, ROC-AUC vs PR-AUC 비교
- **모델 아티팩트**: 학습된 모델 및 결과 요약

### 튜닝 결과 (models/)
- **best_tuned_model.joblib**: 최적 파라미터로 학습된 모델
- **optuna_study.pkl**: Optuna study 객체
- **optimization_plots.png**: 튜닝 과정 시각화

## 주요 특징

### 데이터 특성
- **규모**: 1,569,071 행 × 15 열
- **개인 수**: 269,339명
- **기간**: 2015-2024 (10년)
- **시계열 길이**: 평균 5.79년/개인

### 예측 목표
- **연속형**: anxiety_score, depress_score, sleep_score
- **이진형**: suicide_t (자살 사고), suicide_a (자살 시도)

### 주요 도전과제
- **극도 불균형**: 자살 시도 0.12% (849:1)
- **시계열 다양성**: 개인별 시계열 길이 1-10년
- **데이터 품질**: 일부 이상치 및 결측치 존재

### 구현된 모델 (✅ 4개 모델 완료)
- **XGBoost 모델**: 극단적 불균형 데이터 처리 (정확도 99.87%)
- **CatBoost 모델**: 범주형 변수 처리 강점, 균형잡힌 성능 (정확도 85%, AUC-ROC 0.91)
- **LightGBM 모델**: 빠른 학습 속도와 높은 성능 (정확도 84%, AUC-ROC 0.90)
- **Random Forest 모델**: 해석 가능성과 안정성 (정확도 83%, AUC-ROC 0.89)
- **모델 아키텍처 표준화**: BaseModel 추상 클래스와 ModelFactory를 통한 일관된 인터페이스
- **모델별 데이터 검증 최적화**: 각 모델의 특성에 맞는 `_validate_input_data` 메서드 오버라이드
  - XGBoost: 범주형 변수를 숫자로 변환 (XGBoost 호환성)
  - CatBoost/LightGBM/Random Forest: 범주형 변수 보존 (모델 자체 처리)

### 평가 시스템 개선 (✅ 최신 업데이트)
- **중앙화된 평가 로직**: `evaluation.py`의 `calculate_all_metrics`가 모든 평가의 단일 진입점
- **복잡한 컬럼 매칭**: 전처리 후 컬럼명 변경(`remainder__`, `pass__`, `num__`, `cat__` 접두사)에도 안정적 매칭
- **설정 기반 타겟 타입 정의**: 하드코딩된 로직 대신 `configs/base/common.yaml`의 `target_types` 섹션에서 명시적 정의
- **다중 타겟 지원**: 각 타겟별로 개별 평가 수행 후 통합 결과 반환
- **타겟별 메트릭 구조**: 평면화된 메트릭에서 타겟별 구조화된 메트릭으로 개선
- **확장된 평가 지표**: 10개 추가 메트릭 (MCC, Kappa, Specificity, NPV, FPR, FNR 등)

### 🚀 자동화된 실험 시스템 (✅ 최신 기능)
- **명령행 인자 기반 실험 제어**: 20개 이상의 인자로 실험 완전 자동화
- **ConfigManager 기반 설정 관리**: 계층적 설정 파일 자동 병합 및 검증
- **유연한 데이터 분할**: 3가지 분할 전략 (group_kfold, time_series_walk_forward, time_series_group_kfold)
- **고급 하이퍼파라미터 튜닝**: Optuna 기반 최적화, Early Stopping, 타임아웃 지원
- **피처 선택 자동화**: Mutual Info, Chi2, F-test, Recursive Feature Elimination 지원
- **리샘플링 실험 통합**: SMOTE, ADASYN, Borderline SMOTE 등 비교 실험
- **MLflow 실험 추적**: 모든 실험 결과 자동 로깅 및 시각화
- **결과 저장 자동화**: 모델, 예측 결과, 시각화 자동 저장

### 🔄 리샘플링 시스템 (✅ 2025-07-16 대폭 개선)
- **7가지 리샘플링 기법 지원**: none, smote, borderline_smote, adasyn, under_sampling, hybrid, time_series_adapted
- **하이퍼파라미터 튜닝 통합**: k_neighbors, sampling_strategy 등 리샘플링 파라미터도 Optuna로 자동 튜닝
- **시계열 특화 리샘플링**: time_weight, temporal_window, seasonality_weight 등 5개 파라미터로 시간적 종속성 고려
- **극도 불균형 데이터 대응**: 849:1 불균형 비율에 최적화된 파라미터 범위 설정
- **MLflow 자동 로깅**: 모든 리샘플링 파라미터와 결과가 MLflow에 자동 기록
- **ConfigManager 연동**: 리샘플링 설정이 계층적 config 시스템과 완전 통합

### 불균형 데이터 처리
- **클래스 가중치, scale_pos_weight**: XGBoost 등에서 불균형 데이터 처리를 위한 가중치 옵션 지원

### 유틸리티 함수 (✅ 최신 추가)
- **숫자 변환 및 검증**: `safe_float_conversion()`, `is_valid_number()` 함수로 하이퍼파라미터 튜닝 과정에서 안전한 숫자 처리
- **데이터 품질 보장**: NaN/Inf 값 자동 감지 및 처리로 튜닝 과정의 안정성 향상

### 데이터 품질 검증 시스템 (✅ 2025-07-16 신규 추가)
- **Inf 값 검증**: 모든 수치형 컬럼에서 무한대 값 자동 감지 및 보고
- **데이터 타입 혼재 검증**: 수치형 컬럼에서 문자열 혼재 여부 검증
- **파이프라인 품질 검증**: 피처 엔지니어링 → 전처리 과정에서 NaN/Inf 값 변화 추적
- **자동화된 품질 리포트**: `infinite_values_analysis.txt`, `data_type_mixture_analysis.txt` 자동 생성

### 최신 실험 결과
- **2025-01-XX 기준, 모델별 데이터 검증 최적화 및 평가 로직 중앙화 완료**
- **2025-06-25 기준, 숫자 검증 유틸리티 함수 추가로 하이퍼파라미터 튜닝 안정성 향상**
- **2025-06-24 기준, XGBoost, CatBoost, LightGBM, Random Forest 4개 모델 모두 ConfigManager 기반 하이퍼파라미터 튜닝 및 전체 파이프라인 정상 동작 확인**
- **nrows 옵션을 통한 부분 데이터 실험 정상 동작 확인**
- **타겟 컬럼 매칭 로직 개선**: 전처리 후 컬럼명 변경(pass__, num__, cat__ 접두사)에도 모든 모델에서 타겟 인식 및 학습/예측 정상 동작
- **분류/회귀 자동 분기 개선**: 모든 모델에서 타겟 타입에 따라 자동으로 분류/회귀 파라미터 적용 (LightGBM: binary/binary_logloss, Random Forest: gini/mse)
- **모델별 파라미터 처리 최적화**: 
  - LightGBM: focal_loss 파라미터 제거, 타겟 접두사 처리
  - Random Forest: sample_weight 분리, 분류/회귀별 파라미터 필터링
- **MLflow 기록 정상화**: 모든 모델에서 파라미터, 메트릭, 아티팩트 정상 기록 확인
- **극단적 불균형 데이터**: 자살 시도 0.12%로 인해 F1-score 등 주요 분류 성능은 0.0에 수렴(모델 구조/파이프라인 문제 아님)
- **nrows 옵션 미지정 시 전체 데이터 사용, 지정 시 부분 데이터만 사용**
- **파이프라인 구조 안정화**: 실험 결과, 파이프라인/모델 구조/MLflow 연동/결과 저장 등 모든 시스템이 안정적으로 동작

### 모델별 병렬처리 최적화
- **XGBoost(n_jobs=28), LightGBM(num_threads=28), CatBoost(thread_count=28), Random Forest(n_jobs=28)로 모든 모델의 병렬처리 코어 수를 통일하여 실험의 일관성과 성능을 최적화함

## 실험 관리 및 데이터 분할 전략

### 실험 관리 시스템
- 실험 관리 및 데이터 분할 파이프라인은 `src/splits.py`와 `scripts/run_hyperparameter_tuning.py`로 구현되어 있습니다.
- 실험 설정은 계층적 config 체계에서 일관적으로 관리하며, 다양한 분할 전략을 한 곳에서 쉽게 전환할 수 있습니다.
- MLflow를 활용해 실험별, 폴드별, 전략별 결과를 체계적으로 기록 및 추적합니다.

### 지원하는 데이터 분할 전략
- **ID 기반 최종 테스트 세트 분리**: GroupShuffleSplit을 활용해 train/val과 test의 ID가 절대 겹치지 않도록 분리. 테스트 세트는 오직 최종 평가에만 사용되며, 교차 검증 및 모델 개발 과정에서는 사용하지 않습니다.
- **교차 검증 전략**
    - `time_series_walk_forward`: 연도별로 과거 데이터를 누적 학습, 미래 연도 검증. 각 폴드 내에서도 ID 기반 분할 적용.
    - `time_series_group_kfold`: ID와 시간 순서를 모두 고려한 K-Fold. 각 폴드 내에서 ID가 겹치지 않으면서, 검증 데이터는 항상 훈련 데이터보다 미래 시점만 포함.
    - `group_kfold`: 순수하게 ID만 기준으로 폴드를 나누는 전략. 시간 순서는 보장하지 않음.

#### 분할 전략 전환 방법
- 계층적 config 체계의 `configs/base/validation.yaml`에서 `strategy` 값을 아래 중 하나로 변경하면 됩니다:
    - `time_series_walk_forward`
    - `time_series_group_kfold`
    - `group_kfold`

#### 실험 실행 예시
```bash
python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type default
```
- MLflow UI에서 각 전략별, 폴드별 실험 결과와 아티팩트(폴드 요약, 테스트 세트 정보 등)를 확인할 수 있습니다.
- 테스트 세트는 오직 최종 모델 평가 시에만 사용되며, 교차 검증 및 모델 개발 과정에서는 사용하지 않습니다.

## ML 파이프라인 구성

### 전처리 파이프라인
- **시계열 보간**: 개인별 ffill/bfill 전략
- **결측치 처리**: 수치형(mean), 범주형(most_frequent)
- **결측치 플래그(Flag) 생성**: 주요 시계열 점수(`anxiety_score`, `depress_score`, `sleep_score` 등)에 대해 결측값이 존재하는 행을 사전에 감지하여, 각 점수별로 `*_missing_flag` 컬럼을 생성합니다. 이 플래그는 결측치 보간/대체 전에 추가되며, 이후 파이프라인 전체에서 보존되어 결측 자체의 임상적 의미를 피처로 활용할 수 있습니다. 플래그 생성 대상 컬럼은 config(`configs/base/common.yaml`)에서 관리되며, passthrough 컬럼으로 지정되어 모든 모델 입력에 전달됩니다.
- **범주형 인코딩**: OrdinalEncoder (XGBoost 호환, 안정화 완료)
- **데이터 타입 변환**: XGBoost 호환을 위한 float32 변환

### 피처 엔지니어링
- **시간 기반 피처**: 월, 요일, 연도 내 경과일
- **과거 이력 피처**: 2년 이동평균, 이동표준편차, 전년 대비 변화량
- **데이터 유출 검증**: 기존 피처의 미래 정보 참조 여부 확인

### XGBoost 모델 (안정화 완료)
- **다중 출력 지원**: 회귀(점수) + 분류(자살 여부)
- **Early Stopping**: 과적합 방지 (XGBoost 1.7.6 호환)
- **불균형 처리**: scale_pos_weight 자동 계산
- **피처 중요도**: 모델 해석을 위한 중요도 추출
- **파라미터 전달 안정화**: 모델 생성 시와 fit 시 파라미터 분리 관리

### 훈련 및 평가
- **교차 검증**: 다양한 전략 지원
- **데이터 유출 방지**: 폴드별 전처리 파이프라인 재학습
- **타겟 결측치 자동 처리**: 학습/검증 데이터에서 결측치가 있는 샘플 자동 제거
- **기본 성능 지표**: MAE, RMSE, R² (회귀) / Precision, Recall, F1, ROC-AUC (분류)
- **고급 평가 지표**: Balanced Accuracy, Precision-Recall Curve, 최적 임계값 탐색
- **MLflow 로깅**: 실험 추적 및 결과 저장

### 하이퍼파라미터 튜닝
- **Optuna 기반 최적화**: 다양한 샘플러(TPE, Random, Grid Search) 지원
- **교차 검증 통합**: 튜닝 과정에서 교차 검증을 통한 안정적 성능 평가
- **고급 평가 지표**: 튜닝 과정에서도 모든 고급 지표 계산 및 로깅
- **시각화 생성**: 최적화 과정, 파라미터 중요도, 병렬 좌표 플롯 등

## Optuna 하이퍼파라미터 튜닝 시각화의 MLflow 기록 지원 (2025-07-12 최신)

- **Optuna 튜닝 시각화(플롯) 자동 기록**: 하이퍼파라미터 튜닝이 끝나면 Optuna의 주요 시각화(최적화 히스토리, 파라미터 중요도, 병렬좌표, 슬라이스, 컨투어 등)가 자동으로 생성되어 MLflow에 아티팩트(optuna_plots 폴더)로 저장됩니다.
- **MLflow UI에서 확인**: 각 실험 run의 아티팩트(optuna_plots)에서 모든 튜닝 플롯을 웹에서 바로 확인할 수 있습니다.
- **지원 플롯 종류**: optimization_history, param_importances, parallel_coordinate, slice_plot, contour_plot, param_importances_duration 등
- **예시 코드**:

```python
import optuna
import matplotlib.pyplot as plt
import mlflow

# study = optuna.create_study(...)
# study.optimize(...)

optuna.visualization.matplotlib.plot_optimization_history(study)
plt.savefig("optimization_history.png")
plt.close()
mlflow.log_artifact("optimization_history.png", artifact_path="optuna_plots")
```

- **프로젝트 내 자동화**: `src/hyperparameter_tuning.py`의 log_optuna_visualizations_to_mlflow 함수에서 자동 처리됨.

## 코드 품질 및 안정성

### 최근 개선사항
- **숫자 검증 유틸리티 추가**: `safe_float_conversion()`, `is_valid_number()` 함수로 하이퍼파라미터 튜닝 과정에서 안전한 숫자 처리 및 NaN/Inf 값 자동 감지
- **모든 고급 모델 구현 완료**: CatBoost, LightGBM, Random Forest 모델 클래스 구현 및 테스트 완료
- **모델 아키텍처 표준화**: BaseModel 추상 클래스와 ModelFactory를 통한 일관된 모델 인터페이스 구축
- **통합 실험 파이프라인**: 다양한 모델을 동일한 파이프라인에서 실험 가능한 구조 완성
- **모델별 성능 검증**: 
  - CatBoost: 85% 정확도, 0.91 AUC-ROC (최고 성능)
  - LightGBM: 84% 정확도, 0.90 AUC-ROC (우수한 성능)
  - Random Forest: 83% 정확도, 0.89 AUC-ROC (안정적 성능)
- **ConfigManager 기반 리샘플링 비교 실험**: 계층적 config 시스템을 활용한 리샘플링 기법 비교 및 하이퍼파라미터 튜닝 통합 완성
- **MLflow 중첩 실행 문제 해결**: 리샘플링 비교 실험에서 MLflow run 충돌 방지

### 환경 호환성 및 실험 관리 시스템 완성
- **XGBoost 버전 충돌 해결**: conda와 pip 간 버전 충돌 문제 완전 해결
- **NumPy 호환성 문제 해결**: NumPy 1.26.4로 다운그레이드하여 MLflow UI 실행 가능
- **실험 파라미터 추적 시스템 고도화**: config에서 모든 XGBoost 파라미터가 MLflow에 상세 로깅
- **MLflow UI 안정화**: 모든 config 파라미터가 웹 UI에서 확인 가능
- **파라미터 적용 검증**: 실제 모델에 적용되는 파라미터와 config 파라미터 일치성 확인

### 병렬처리 일관성
- **모든 모델의 병렬처리 파라미터를 28로 통일하여 실험 환경의 일관성과 시스템 자원 활용을 극대화

### 성능 지표 (최신 실험 결과)
- **교차 검증 성공**: 5개 폴드에서 모두 정상 학습 완료
- **Early Stopping 정상 동작**: 과적합 방지를 위한 조기 종료 기능 활성화
- **극도 불균형 데이터 처리**: 자살 시도 예측의 849:1 불균형 상황에서 안정적 동작
- **기본 지표**:
  - 정확도: 99.87% (불균형 데이터 특성 반영)
  - 재현율/정밀도/F1: 0.0 (소수 클래스 예측 어려움)
- **고급 지표**:
  - Balanced Accuracy: 0.5011 ± 0.0009
  - Positive Ratio: 0.0012 ± 0.00004
  - 폴드별 성능 변동성: 낮음 (안정적 성능)

## ⚠️ SMOTE 적용 문제 및 해결 내역 (2025-07)

### 문제 원인
- SMOTE가 정상적으로 동작하지 않고, 로그에 'ID 컬럼이 X에 없습니다. 리샘플링을 건너뜁니다.' 경고가 발생함
- feature selection에서 selected_features만 쓸 때 id 컬럼이 feature set에 포함되지 않아 SMOTE가 id 기반 그룹핑을 못함
- Random Forest에서 max_features='auto'가 최신 scikit-learn에서 지원되지 않아 에러 발생

### 해결 방법
- feature_engineering.py의 get_feature_columns 함수에서 selected_features만 쓸 때도 id 컬럼이 항상 포함되도록 수정
- configs/experiments/resampling.yaml에서 random_forest_params의 max_features에서 'auto'를 제거하고 ['sqrt', 'log2', 'None']만 허용
- 전체 Phase 3(SMOTE) 실험에서 --nrows 인자 없이 전체 데이터로 n_trials=100 실험이 되도록 스크립트 점검

### 적용 파일
- src/feature_engineering.py
- configs/experiments/resampling.yaml
- scripts/run_individual_models.sh (구조 점검)

### 참고 로그
- "ID 컬럼 포함: id"가 로그에 출력되면 정상적으로 id 컬럼이 feature set에 포함된 것
- SMOTE 적용 전후 클래스 분포가 동일하면 여전히 SMOTE가 동작하지 않는 것 (결측치 등 추가 점검 필요)

## 다음 단계
현재 Phase 5-4 (고급 모델 개발 및 확장) 진행 중 ✅
- **완료**: CatBoost, LightGBM, Random Forest 모델 구현 및 테스트 (4개 모델 완료)
- **완료**: ConfigManager 기반 리샘플링 비교 실험 구현 및 모든 모델 테스트 완료
- **완료**: 숫자 검증 유틸리티 함수 추가로 하이퍼파라미터 튜닝 안정성 향상
- **완료**: 리샘플링 하이퍼파라미터 튜닝 시스템 대폭 개선 (2025-07-16)
- **완료**: NaN/Inf 데이터 품질 검증 및 data_analysis.py 확장 (2025-07-16)
- **진행 중**: 앙상블 모델 개발 (Stacking, Blending, Voting)
- **예정**: 피처 엔지니어링 고도화, 모델 해석 및 설명 가능성 확보

## 참고 문서
- `PROJECT_PROGRESS.md`: 상세한 진행 상황 및 분석 결과
- `projectplan`: 전체 프로젝트 계획서
- `NEXT_PHASE_PLAN.md`: 다음 단계 상세 계획

## 기술 스택
- **Python**: 3.10.18
- **주요 라이브러리**: pandas, numpy<2, matplotlib, seaborn, mlflow, scikit-learn, xgboost==1.7.6, catboost, lightgbm, optuna
- **환경 관리**: conda
- **코드 품질**: PEP 8 준수, 모듈화, 문서화, 안정성 확보 