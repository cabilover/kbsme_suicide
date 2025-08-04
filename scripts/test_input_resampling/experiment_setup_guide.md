# KBSME 자살 예측 모델 - 실험 설정 및 실행 가이드

## 🚀 빠른 시작

### 1. 스크립트 파일 생성 및 권한 설정

위에서 제공된 5개의 스크립트 파일을 프로젝트 루트 디렉토리에 저장하세요:

```bash
# 스크립트 파일들을 저장한 후 실행 권한 부여
chmod +x phase1_baseline.sh
chmod +x phase2_input_range.sh  
chmod +x phase3_resampling.sh
chmod +x phase4_deep_analysis.sh
chmod +x phase5_final_optimization.sh
chmod +x master_experiment_runner.sh
```

### 2. 환경 확인

```bash
# 현재 환경이 올바른지 확인
conda activate simcare
python --version  # Python 3.10.18 확인
pip list | grep mlflow  # MLflow 설치 확인
```

### 3. 실험 실행

```bash
# 마스터 스크립트로 전체 실험 관리
./master_experiment_runner.sh
```

## 📋 실험 단계별 상세 설명

### Phase 1: 기준선 설정 (4-6시간)
- **목적**: 4개 모델의 기본 성능 측정
- **실험 수**: 4개 (XGBoost, CatBoost, LightGBM, Random Forest)
- **설정**: 각 모델 50 trials 하이퍼파라미터 튜닝

```bash
./phase1_baseline.sh
```

### Phase 2: Input 범위 조정 (6-8시간)
- **목적**: 데이터 크기와 피처 선택 효과 분석
- **실험 수**: 8개
- **구성**: 
  - 데이터 크기별 (10K, 100K, 500K)
  - 피처 선택 방법별 (mutual_info, chi2, recursive)

```bash
./phase2_input_range.sh
```

### Phase 3: Resampling 비교 (8-12시간)
- **목적**: 극도 불균형 데이터(849:1)에 최적 리샘플링 기법 발견
- **실험 수**: 35개
- **구성**:
  - 전체 리샘플링 비교: 4개 모델 × 7개 기법
  - 시계열 특화 리샘플링
  - 특정 기법 집중 분석
  - 하이브리드 접근

```bash
./phase3_resampling.sh
```

### Phase 4: 모델 심층 분석 (12-16시간)
- **목적**: 각 모델의 특성과 최적 성능 도출
- **실험 수**: 16개
- **구성**:
  - 고성능 튜닝 (200 trials)
  - 시계열 분할 전략 비교
  - 모델별 특성 분석
  - 앙상블 준비

```bash
./phase4_deep_analysis.sh
```

### Phase 5: 통합 최적화 (4-6시간)
- **목적**: 최종 최적화된 모델 구축
- **실험 수**: 14개
- **구성**:
  - 최적 조합 실험
  - 시계열 최적화
  - 앙상블 구축
  - 배포 준비

```bash
./phase5_final_optimization.sh
```

## 🎯 예상 결과

### 모델별 성능 예상치
- **CatBoost**: 85%+ 정확도, 0.91+ AUC-ROC (균형잡힌 최고 성능)
- **XGBoost**: 99.87% 정확도 (극도 불균형 특화)
- **LightGBM**: 84% 정확도, 0.90 AUC-ROC (속도-성능 균형)
- **Random Forest**: 83% 정확도, 0.89 AUC-ROC (해석가능성)

### 핵심 분석 포인트
1. **극도 불균형 처리**: 자살 시도 0.12% (849:1) 비율에서 각 모델의 대응
2. **리샘플링 효과**: SMOTE, ADASYN, Borderline SMOTE 등의 성능 차이
3. **시계열 특성**: 시간 순서를 고려한 분할 전략의 중요성
4. **피처 중요도**: 각 모델에서 중요하게 간주되는 피처들

## 📊 결과 분석 방법

### MLflow UI 사용법
```bash
# MLflow UI 실행
mlflow ui --host 0.0.0.0 --port 5000

# 웹 브라우저에서 http://localhost:5000 접속
```

### 주요 확인 사항
1. **실험 비교**: Experiments 탭에서 phase1~phase5 비교
2. **메트릭 분석**: F1-score, PR-AUC, MCC, Balanced Accuracy 중심
3. **하이퍼파라미터**: 최고 성능 모델의 파라미터 설정
4. **시각화**: 각 실험의 아티팩트에서 차트 확인

## ⚠️ 주의사항

### 시스템 요구사항
- **CPU**: 최소 8코어 (권장 16코어 이상)
- **메모리**: 최소 16GB (권장 32GB 이상)
- **저장공간**: 최소 50GB 여유공간
- **시간**: 전체 실험 2-4일 소요

### 리소스 관리
```bash
# CPU 사용량 모니터링
python scripts/check_cpu_usage.py

# 메모리 부족 시 실험 중단
pkill -f "run_hyperparameter_tuning"
pkill -f "run_resampling_experiment"

# MLflow 실험 정리 (디스크 공간 확보)
python scripts/cleanup_mlflow_experiments.py --action list
```

### 실험 중단 및 재시작
- 각 Phase는 독립적으로 실행 가능
- 중단된 Phase부터 재시작 가능
- MLflow에서 진행 상황 확인 가능
- **Ctrl+C로 언제든지 안전하게 중단 가능**

## 🛠️ 문제 해결

### 일반적인 오류들

1. **메모리 부족**
   ```bash
   # nrows 옵션으로 데이터 크기 축소
   python scripts/run_hyperparameter_tuning.py --model-type xgboost --nrows 10000
   ```

2. **XGBoost 버전 문제**
   ```bash
   pip install xgboost==1.7.6
   ```

3. **NumPy 호환성**
   ```bash
   pip install "numpy<2"
   ```

4. **MLflow UI 접속 불가**
   ```bash
   # 다른 포트로 실행
   mlflow ui --host 0.0.0.0 --port 5001
   ```

5. **리샘플링 인자 오류**
   ```bash
   # 올바른 사용법
   python scripts/run_resampling_experiment.py --model-type xgboost --resampling-method smote
   # 잘못된 사용법 (지원되지 않음)
   python scripts/run_hyperparameter_tuning.py --resampling-method smote
   ```

## 📈 실험 진행 추적

### 진행 상황 체크리스트
- [ ] Phase 1: 기준선 설정 (4개 실험)
- [ ] Phase 2: Input 범위 조정 (8개 실험)  
- [ ] Phase 3: Resampling 비교 (35개 실험)
- [ ] Phase 4: 모델 심층 분석 (16개 실험)
- [ ] Phase 5: 통합 최적화 (14개 실험)

### 실험 로그 확인
```bash
# 실험 로그 파일들
ls -la logs/
ls -la results/

# 요약 리포트들
cat phase*_summary.txt
cat comprehensive_experiment_summary.txt
```

## 🔧 개선된 기능 (2025-08-04)

### ✅ 실험 중단/재시작 기능
- **안전한 중단**: Ctrl+C로 언제든지 안전하게 중단 가능
- **프로세스 정리**: 중단 시 자동으로 실행 중인 프로세스 정리
- **재시작 지원**: 중단된 Phase부터 재시작 가능
- **오류 복구**: 실험 실패 시 계속 진행 여부 선택 가능

### ✅ 명령어 인자 검증 강화
- **파일 존재 확인**: 필수 스크립트 및 데이터 파일 자동 검증
- **인자 호환성 검사**: 리샘플링 관련 명령어 올바른 사용법 적용
- **환경 검증**: Python 환경 및 MLflow 설치 상태 확인

### ✅ 리샘플링 인자 수정
- **올바른 스크립트 사용**: 리샘플링 실험은 `run_resampling_experiment.py` 사용
- **인자 분리**: `--resampling-comparison`과 `--resampling-method` 적절히 분리
- **호환성 보장**: 각 스크립트의 지원 인자에 맞게 수정

### ✅ 실험 관리 개선
- **Phase 간 결과 확인**: 각 Phase 완료 후 MLflow UI 실행 옵션
- **시스템 리소스 모니터링**: 실험 전후 CPU/메모리 상태 확인
- **자동 요약 리포트**: 각 Phase별 상세한 요약 리포트 자동 생성

## 🎉 완료 후 다음 단계

1. **최종 모델 선정**: MLflow UI에서 최고 성능 모델 확인
2. **앙상블 구축**: 여러 모델을 조합한 앙상블 모델 개발
3. **해석가능성 분석**: SHAP 등을 활용한 모델 해석
4. **배포 준비**: 실제 운영 환경을 위한 모델 최적화
5. **임상 검증**: 실제 임상 데이터를 통한 검증

## 📋 마스터 스크립트 사용법

### 실행 모드 선택
```bash
./master_experiment_runner.sh
```

1. **전체 실험 순차 실행**: Phase 1-5를 순차적으로 실행
2. **개별 Phase 선택**: 특정 Phase만 실행
3. **빠른 테스트 모드**: 소규모 데이터로 빠른 검증
4. **환경 설정 및 초기 분석**: 시스템 상태 및 데이터 분석
5. **실험 상태 확인 및 MLflow UI**: 현재 실험 상태 확인
6. **실험 중단 및 정리**: 실행 중인 프로세스 정리

### 안전 기능
- **자동 검증**: 실행 전 필수 파일 및 환경 검증
- **안전한 중단**: Ctrl+C로 언제든지 안전하게 중단
- **오류 복구**: 실험 실패 시 계속 진행 여부 선택
- **리소스 모니터링**: 시스템 리소스 상태 확인

---

**문의사항이나 문제가 발생하면 MLflow UI의 실험 로그와 콘솔 출력을 확인하여 디버깅하세요.**