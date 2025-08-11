# KBSMC 자살 예측 프로젝트

## 프로젝트 개요
개인별 연간 정신 건강 지표 데이터를 활용하여 다음 해의 불안/우울/수면 점수 및 자살 사고/시도 여부를 예측하는 머신러닝 프로젝트입니다.

## 주요 특징
- **데이터 규모**: 1,569,071 행 × 15 열 (269,339명, 2015-2024)
- **예측 목표**: 불안/우울/수면 점수 (회귀), 자살 사고/시도 여부 (분류)
- **주요 도전과제**: 극도 불균형 데이터 (자살 시도 0.12%, 849:1)
- **구현된 모델**: XGBoost, CatBoost, LightGBM, Random Forest

## 프로젝트 구조
```
kbsmc_suicide/
├── src/                    # 소스 코드
│   ├── models/            # 모델 구현 (BaseModel, ModelFactory)
│   ├── utils/             # 유틸리티 (ConfigManager, MLflow 관리)
│   ├── data_analysis.py   # 데이터 분석 및 전처리
│   ├── feature_engineering.py # 피처 엔지니어링
│   ├── training.py        # 훈련 파이프라인
│   ├── evaluation.py      # 평가 모듈
│   └── hyperparameter_tuning.py # Optuna 기반 튜닝
├── configs/                # 설정 파일 (계층적 구조)
│   ├── base/              # 기본 설정
│   ├── models/            # 모델별 설정
│   └── experiments/       # 실험별 설정
├── scripts/                # 실행 스크립트
│   ├── run_hyperparameter_tuning.py # 통합 실험 실행
│   ├── run_resampling_experiment.py # 리샘플링 실험
│   └── test_input_resampling/ # 대규모 실험 시스템
├── data/                   # 데이터 파일
├── models/                 # 학습된 모델
├── results/                # 실험 결과
└── logs/                   # 로그 파일
```

## 핵심 기능

### 🚀 자동화된 실험 시스템
- **ConfigManager 기반 설정 관리**: 계층적 설정 파일 자동 병합 및 검증
- **명령행 인자 기반 실험 제어**: 20개 이상의 인자로 실험 완전 자동화
- **유연한 데이터 분할**: 3가지 분할 전략 지원
- **고급 하이퍼파라미터 튜닝**: Optuna 기반 최적화, Early Stopping 지원

### 🔄 리샘플링 시스템
- **7가지 리샘플링 기법**: SMOTE, ADASYN, Borderline SMOTE 등
- **하이퍼파라미터 튜닝 통합**: 리샘플링 파라미터도 Optuna로 자동 튜닝
- **시계열 특화 리샘플링**: 시간적 종속성 고려한 고급 기법

### 📊 MLflow 통합
- **실험 추적**: 모든 실험 결과 자동 로깅
- **시각화 관리**: Optuna 플롯, 학습 곡선 등 자동 저장
- **아티팩트 관리**: 모델, 예측 결과, 시각화 파일 체계적 저장

### 🛠️ 모델 아키텍처
- **BaseModel 추상 클래스**: 일관된 모델 인터페이스
- **ModelFactory 패턴**: 동적 모델 생성 및 등록 시스템
- **모델별 최적화**: 각 모델의 특성에 맞는 데이터 검증 및 처리

## 환경 설정

### 1. Conda 환경 활성화
```bash
conda activate simcare
```

### 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

> ⚠️ 본 프로젝트는 xgboost==1.7.6 버전에 최적화되어 있습니다.

## 빠른 시작

### 데이터 분석 및 전처리
```bash
python src/data_analysis.py
```

### 하이퍼파라미터 튜닝
```bash
# XGBoost 모델 튜닝
python scripts/run_hyperparameter_tuning.py \
  --model-type xgboost \
  --experiment-type hyperparameter_tuning \
  --n-trials 50

# CatBoost 모델 튜닝
python scripts/run_hyperparameter_tuning.py \
  --model-type catboost \
  --experiment-type hyperparameter_tuning \
  --n-trials 50
```

### 리샘플링 실험
```bash
# 리샘플링 비교 실험
python scripts/run_hyperparameter_tuning.py \
  --model-type xgboost \
  --experiment-type resampling \
  --resampling-comparison

# 특정 리샘플링 기법 튜닝
python scripts/run_hyperparameter_tuning.py \
  --model-type lightgbm \
  --experiment-type resampling \
  --resampling-method smote \
  --n-trials 30
```

## 고급 사용법

### 🚀 대규모 실험 시스템
```bash
cd scripts/test_input_resampling/
chmod +x *.sh
./master_experiment_runner.sh
```

**5단계 실험 구성:**
- **Phase 1**: 기준선 설정 (4개 모델, 50 trials, 4-6시간)
- **Phase 2**: Input 범위 조정 (데이터 크기, 피처 선택, 6-8시간)
- **Phase 3**: 리샘플링 비교 (35개 실험, 8-12시간)
- **Phase 4**: 모델 심층 분석 (16개 실험, 12-16시간)
- **Phase 5**: 통합 최적화 (14개 실험, 4-6시간)

### 🔧 설정 파일 커스터마이징
```yaml
# configs/base/common.yaml
features:
  target_variables:
    next_year_targets:
      score_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
      binary_targets: ["suicide_t_next_year", "suicide_a_next_year"]
  
  target_types:
    regression_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
    classification_targets: ["suicide_t_next_year", "suicide_a_next_year"]
```

### 📊 데이터 분할 전략
```bash
# 시간 기반 분할
python scripts/run_hyperparameter_tuning.py \
  --split-strategy time_series_walk_forward \
  --cv-folds 5

# ID 기반 분할
python scripts/run_hyperparameter_tuning.py \
  --split-strategy group_kfold \
  --cv-folds 5
```

## 실험 결과

### 생성되는 파일들
- **MLflow 실험**: 모든 실험 파라미터, 메트릭, 아티팩트
- **시각화**: 최적화 과정, 파라미터 중요도, 학습 곡선
- **모델**: 최적 파라미터로 학습된 모델 (joblib)
- **결과 요약**: CSV, JSON 형태의 상세 결과

### 성능 지표
- **회귀**: MAE, RMSE, R²
- **분류**: Precision, Recall, F1, ROC-AUC, PR-AUC
- **고급 지표**: Balanced Accuracy, MCC, Kappa 등

## 시스템 요구사항

### 최소 요구사항
- **CPU**: 8코어
- **메모리**: 16GB
- **저장공간**: 50GB 여유공간

### 권장 사항
- **CPU**: 16코어 이상
- **메모리**: 32GB 이상
- **저장공간**: 100GB 이상

## 기술 스택
- **Python**: 3.10.18
- **주요 라이브러리**: pandas, numpy, scikit-learn, xgboost==1.7.6, catboost, lightgbm, optuna, mlflow
- **환경 관리**: conda
- **코드 품질**: PEP 8 준수, 모듈화, 문서화

## 참고 문서
- `PROJECT_PROGRESS.md`: 상세한 진행 상황 및 분석 결과
- `projectplan`: 전체 프로젝트 계획서

## 라이선스
이 프로젝트는 연구 목적으로 개발되었습니다.

## 기여 방법
1. 이슈 생성 또는 기존 이슈 확인
2. 기능 브랜치 생성
3. 코드 작성 및 테스트
4. Pull Request 생성

## 연락처
프로젝트 관련 문의사항은 이슈를 통해 연락해 주세요.
