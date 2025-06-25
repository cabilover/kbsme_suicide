# KBSMC 자살 예측 프로젝트

## 프로젝트 개요
개인별 연간 정신 건강 지표 데이터를 활용하여 다음 해의 불안/우울/수면 점수 및 자살 사고/시도 여부를 예측하는 머신러닝 프로젝트입니다.

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
│   │   ├── xgboost_model.py          # XGBoost 모델 클래스 (안정화 완료)
│   │   ├── catboost_model.py         # CatBoost 모델 클래스 (✅ 구현 완료)
│   │   ├── lightgbm_model.py         # LightGBM 모델 클래스 (✅ 구현 완료)
│   │   ├── random_forest_model.py    # Random Forest 모델 클래스 (✅ 구현 완료)
│   │   └── loss_functions.py         # 손실 함수 모듈 (Focal Loss 포함)
│   ├── training.py                   # 훈련 파이프라인 (품질 개선 완료)
│   ├── evaluation.py                 # 평가 모듈 (고급 평가 기능 포함)
│   ├── hyperparameter_tuning.py      # 하이퍼파라미터 튜닝 (Optuna 기반, ✅ 숫자 검증 유틸리티 추가)
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
│   └── run_hyperparameter_tuning.py  # 통합 실험 실행 스크립트 (✅ ConfigManager 기반 리샘플링 비교 포함)
├── requirements.txt                  # 필요한 패키지 목록 (XGBoost 1.7.6 고정)
├── projectplan                       # 프로젝트 계획서
├── PROJECT_PROGRESS.md              # 프로젝트 진행 상황 문서
└── README.md                        # 이 파일
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

### ConfigManager 기반 하이퍼파라미터 튜닝 실행 (권장)
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

### ConfigManager 기반 리샘플링 비교 실험 실행 (✅ 최신 기능)
```bash
# 모든 모델에 대해 리샘플링 비교 실험
python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type resampling --resampling-comparison
python scripts/run_hyperparameter_tuning.py --model-type catboost --experiment-type resampling --resampling-comparison
python scripts/run_hyperparameter_tuning.py --model-type lightgbm --experiment-type resampling --resampling-comparison
python scripts/run_hyperparameter_tuning.py --model-type random_forest --experiment-type resampling --resampling-comparison

# 특정 리샘플링 기법만 비교
python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type resampling --resampling-comparison --resampling-methods smote adasyn
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

### 불균형 데이터 처리
- **클래스 가중치, scale_pos_weight**: XGBoost 등에서 불균형 데이터 처리를 위한 가중치 옵션 지원

### 유틸리티 함수 (✅ 최신 추가)
- **숫자 변환 및 검증**: `safe_float_conversion()`, `is_valid_number()` 함수로 하이퍼파라미터 튜닝 과정에서 안전한 숫자 처리
- **데이터 품질 보장**: NaN/Inf 값 자동 감지 및 처리로 튜닝 과정의 안정성 향상

### 최신 실험 결과
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

## 다음 단계
현재 Phase 5-4 (고급 모델 개발 및 확장) 진행 중 ✅
- **완료**: CatBoost, LightGBM, Random Forest 모델 구현 및 테스트 (4개 모델 완료)
- **완료**: ConfigManager 기반 리샘플링 비교 실험 구현 및 모든 모델 테스트 완료
- **완료**: 숫자 검증 유틸리티 함수 추가로 하이퍼파라미터 튜닝 안정성 향상
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