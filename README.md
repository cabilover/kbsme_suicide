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
│   │   └── loss_functions.py         # 손실 함수 모듈 (Focal Loss 포함)
│   ├── training.py                   # 훈련 파이프라인 (품질 개선 완료)
│   ├── evaluation.py                 # 평가 모듈 (고급 평가 기능 포함)
│   ├── hyperparameter_tuning.py      # 하이퍼파라미터 튜닝 (Optuna 기반)
│   └── reference/                    # 참고 자료
├── configs/
│   ├── default_config.yaml           # 실험 설정 (일관성 확보)
│   ├── focal_loss_config.yaml        # Focal Loss 실험 설정
│   ├── resampling_config.yaml        # 리샘플링 실험 및 하이퍼파라미터 튜닝 통합 설정
│   └── hyperparameter_tuning.yaml    # 하이퍼파라미터 튜닝 설정
├── scripts/
│   ├── run_experiment.py             # 실험 실행 스크립트 (고급 평가 및 리샘플링 비교 포함)
│   └── run_hyperparameter_tuning.py  # 하이퍼파라미터 튜닝 실행 스크립트 (리샘플링 비교 포함)
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

### ML 파이프라인 실험 실행
```bash
python scripts/run_experiment.py --config configs/default_config.yaml --data data/processed/processed_data_with_features.csv --nrows 1000
```

이 명령어는 다음 작업을 수행합니다:
- 데이터 분할 (ID 기반 테스트 세트 분리)
- 교차 검증 (다양한 전략 지원)
- 전처리 및 피처 엔지니어링
- XGBoost 모델 학습 및 평가 (Early Stopping 지원)
- 고급 평가 지표 계산 (Balanced Accuracy, Precision-Recall Curve 등)
- MLflow를 통한 실험 결과 로깅

### 하이퍼파라미터 튜닝 실행
```bash
python scripts/run_hyperparameter_tuning.py --tuning_config configs/hyperparameter_tuning.yaml --nrows 1000
```

이 명령어는 다음 작업을 수행합니다:
- Optuna 기반 하이퍼파라미터 최적화
- Focal Loss 파라미터 튜닝 (alpha, gamma)
- 교차 검증을 통한 성능 평가
- 고급 평가 지표 계산 및 로깅
- 최적 모델 저장 및 시각화 생성

### 리샘플링 실험 실행
```bash
# 리샘플링 기법 비교 실험
python scripts/run_experiment.py --resampling-comparison --nrows 1000

# 특정 리샘플링 기법들만 비교
python scripts/run_experiment.py --resampling-comparison --resampling-methods smote borderline_smote adasyn --nrows 1000

# 리샘플링 하이퍼파라미터 튜닝 비교
python scripts/run_hyperparameter_tuning.py --resampling-comparison --nrows 1000

# 통합 설정 파일 사용 (기본값)
python scripts/run_hyperparameter_tuning.py --nrows 1000
```

이 명령어들은 다음 작업을 수행합니다:
- 다양한 리샘플링 기법 비교 (SMOTE, Borderline SMOTE, ADASYN, 언더샘플링, 하이브리드)
- 각 리샘플링 기법별 하이퍼파라미터 최적화
- MLflow를 통한 실험 추적 및 비교
- 최고 성능 기법 자동 선별

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

### 불균형 데이터 처리 및 Focal Loss 지원
- **Focal Loss 통합**: XGBoost 모델에 Focal Loss를 옵션으로 통합, 극단적 불균형 데이터(예: 자살 시도 849:1)에서 소수 클래스 예측 성능 개선 시도
- **Focal Loss 파라미터 튜닝**: Optuna 기반 하이퍼파라미터 튜닝에서 `use_focal_loss`, `focal_loss_alpha`, `focal_loss_gamma` 등 Focal Loss 관련 파라미터 탐색 가능
- **튜닝/실험 파이프라인 완전 호환**: `run_experiment.py`와 `run_hyperparameter_tuning.py` 모두에서 Focal Loss 및 관련 파라미터가 정상적으로 반영 및 실험됨
- **설정 파일 예시**: `configs/default_config.yaml`, `configs/focal_loss_config.yaml`, `configs/hyperparameter_tuning.yaml`에서 Focal Loss 옵션 및 탐색 범위 지정 가능

### 고급 평가 지표 및 분석 기능
- **Balanced Accuracy**: 클래스 불균형을 고려한 정확도 측정
- **Precision-Recall Curve**: 불균형 데이터에 적합한 성능 평가
- **ROC-AUC vs PR-AUC 비교**: 균형/불균형 데이터에서의 성능 차이 분석
- **F1-Score 임계값 최적화**: 최적 분류 임계값 자동 탐색
- **폴드별 성능 변동성 분석**: 교차 검증 안정성 평가
- **클래스별 샘플 수 통계**: 양성/음성 샘플 분포 및 비율 분석
- **MLflow 통합 로깅**: 모든 고급 지표가 MLflow에 자동 로깅되어 실험 추적 강화

### 리샘플링 실험 및 하이퍼파라미터 튜닝 통합
- **리샘플링 기법 비교**: SMOTE, Borderline SMOTE, ADASYN, 언더샘플링, 하이브리드 기법 비교
- **기법별 하이퍼파라미터 튜닝**: 각 리샘플링 기법에 대해 별도로 최적 하이퍼파라미터 탐색
- **통합 설정 파일**: `configs/resampling_config.yaml`에서 리샘플링 실험과 하이퍼파라미터 튜닝 설정 통합 관리
- **자동 성능 비교**: F1-Score 기준으로 최고 성능 기법 자동 선별
- **MLflow 중첩 실행**: 각 리샘플링 기법별로 별도 MLflow run으로 실험 추적
- **사용자 인터페이스**: `--resampling-comparison`, `--resampling-methods` 인자로 쉬운 실험 제어

## 실험 관리 및 데이터 분할 전략

### 실험 관리 시스템
- 실험 관리 및 데이터 분할 파이프라인은 `src/splits.py`와 `scripts/run_experiment.py`로 구현되어 있습니다.
- 실험 설정은 `configs/default_config.yaml`에서 일관적으로 관리하며, 다양한 분할 전략을 한 곳에서 쉽게 전환할 수 있습니다.
- MLflow를 활용해 실험별, 폴드별, 전략별 결과를 체계적으로 기록 및 추적합니다.

### 지원하는 데이터 분할 전략
- **ID 기반 최종 테스트 세트 분리**: GroupShuffleSplit을 활용해 train/val과 test의 ID가 절대 겹치지 않도록 분리. 테스트 세트는 오직 최종 평가에만 사용되며, 교차 검증 및 모델 개발 과정에서는 사용하지 않습니다.
- **교차 검증 전략**
    - `time_series_walk_forward`: 연도별로 과거 데이터를 누적 학습, 미래 연도 검증. 각 폴드 내에서도 ID 기반 분할 적용.
    - `time_series_group_kfold`: ID와 시간 순서를 모두 고려한 K-Fold. 각 폴드 내에서 ID가 겹치지 않으면서, 검증 데이터는 항상 훈련 데이터보다 미래 시점만 포함.
    - `group_kfold`: 순수하게 ID만 기준으로 폴드를 나누는 전략. 시간 순서는 보장하지 않음.

#### 분할 전략 전환 방법
- `configs/default_config.yaml`의 `validation.strategy` 값을 아래 중 하나로 변경하면 됩니다:
    - `time_series_walk_forward`
    - `time_series_group_kfold`
    - `group_kfold`

#### 실험 실행 예시
```bash
python scripts/run_experiment.py --config configs/default_config.yaml --nrows 10000
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
- **Focal Loss 지원**: 극단적 불균형 데이터 처리를 위한 Focal Loss 옵션
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
- **Focal Loss 파라미터 튜닝**: alpha, gamma 파라미터 탐색
- **교차 검증 통합**: 튜닝 과정에서 교차 검증을 통한 안정적 성능 평가
- **고급 평가 지표**: 튜닝 과정에서도 모든 고급 지표 계산 및 로깅
- **시각화 생성**: 최적화 과정, 파라미터 중요도, 병렬 좌표 플롯 등

## 코드 품질 및 안정성

### 최근 개선사항
- **리샘플링 실험 통합**: 다양한 리샘플링 기법 비교 및 하이퍼파라미터 튜닝 통합 완성
- **통합 설정 파일**: `configs/resampling_config.yaml`에서 리샘플링 실험과 하이퍼파라미터 튜닝 설정 통합 관리
- **사용자 인터페이스 개선**: 리샘플링 실험을 위한 명령행 인자 및 도움말 추가
- **고급 평가 기능 완전 통합**: Balanced Accuracy, Precision-Recall Curve, 최적 임계값 탐색 등 불균형 데이터에 특화된 평가 지표 완전 구현
- **MLflow 통합 강화**: 모든 고급 평가 지표가 MLflow에 자동 로깅되어 실험 추적 및 비교 분석 가능
- **Focal Loss 실험/튜닝 완전 지원**: XGBoost 모델 및 하이퍼파라미터 튜닝 파이프라인에서 Focal Loss 및 관련 파라미터가 완전하게 지원됨 (설정 파일, 실험, 튜닝, 모델 저장까지 일관성 보장)
- **Optuna 튜닝 파이프라인 개선**: Focal Loss 파라미터도 탐색 가능하도록 파라미터 구조 개선
- **설정 파일 구조 개선**: Focal Loss 옵션 및 파라미터가 config와 tuning config에 명확히 반영
- **XGBoost 버전 고정**: 1.7.6 버전으로 안정화하여 early_stopping_rounds 호환성 확보
- **모델 파라미터 전달 개선**: fit 메서드에서 파라미터 누락 문제 해결
- **임포트 구조 정리**: 중복 임포트 제거 및 일관된 모듈 구조 적용
- **범주형 인코딩 표준화**: OrdinalEncoder 사용으로 안정성 향상
- **타겟 결측치 자동 처리**: 학습 과정에서 결측치가 있는 샘플 자동 제거

### 환경 호환성 및 실험 관리 시스템 완성
- **XGBoost 버전 충돌 해결**: conda와 pip 간 버전 충돌 문제 완전 해결
- **NumPy 호환성 문제 해결**: NumPy 1.26.4로 다운그레이드하여 MLflow UI 실행 가능
- **실험 파라미터 추적 시스템 고도화**: config에서 모든 XGBoost 파라미터가 MLflow에 상세 로깅
- **MLflow UI 안정화**: 모든 config 파라미터가 웹 UI에서 확인 가능
- **파라미터 적용 검증**: 실제 모델에 적용되는 파라미터와 config 파라미터 일치성 확인

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
현재 Phase 5-3 (불균형 데이터 처리 및 Focal Loss 통합 + 고급 평가 기능 통합) 완료 ✅
→ Phase 5-4 (고급 모델 개발 및 성능 최적화) 진행 예정

## 참고 문서
- `PROJECT_PROGRESS.md`: 상세한 진행 상황 및 분석 결과
- `projectplan`: 전체 프로젝트 계획서

## 기술 스택
- **Python**: 3.10.18
- **주요 라이브러리**: pandas, numpy<2, matplotlib, seaborn, mlflow, scikit-learn, xgboost==1.7.6, optuna
- **환경 관리**: conda
- **코드 품질**: PEP 8 준수, 모듈화, 문서화, 안정성 확보 