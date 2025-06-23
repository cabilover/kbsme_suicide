# 📋 Phase 5-4 작업 계획 상세

## 🎯 Phase 5-4의 핵심 목표
현재 Phase 5-3에서 완성된 불균형 데이터 처리 및 리샘플링 하이퍼파라미터 튜닝 통합 시스템을 기반으로, 고급 모델 개발과 성능 최적화를 통해 예측 성능을 한 단계 더 향상시키는 것입니다.

## 🆕 [업데이트] Baseline 실험 계획
- **최소 feature(기본 feature)와 기본 하이퍼파라미터 튜닝만으로 각 모델의 성능을 측정하는 실험을 우선적으로 진행합니다.**
    - 피처 엔지니어링, 리샘플링, 앙상블 등 고급 전략을 적용하기 전에, 가장 단순한 feature set과 모델별 기본 튜닝만으로 얻을 수 있는 baseline 성능을 명확히 기록합니다.
    - 이 baseline 결과는 이후 전략(리샘플링, 고급 피처, 앙상블 등) 도입 시 성능 향상 폭을 객관적으로 비교하는 기준이 됩니다.
    - 실험 방법: config에서 enable_feature_engineering=False, selected_features에 최소 feature만 지정, tuning space도 기본값 위주로 제한
    - 모든 모델(XGBoost, CatBoost, LightGBM, Random Forest)에 대해 동일하게 적용
    - 결과는 MLflow 및 결과 파일로 기록

## 📊 현재 상황 분석
- **완성된 시스템**: 
  - XGBoost 모델 (불균형 데이터 처리)
  - 고급 평가 기능 (Balanced Accuracy, Precision-Recall Curve 등)
  - 리샘플링 실험 및 하이퍼파라미터 튜닝 통합 시스템
  - MLflow 기반 실험 관리 시스템
  - ✅ 계층적 ConfigManager 시스템 구축 완료
  - ✅ 모델 아키텍처 표준화 완료 (BaseModel, ModelFactory)
  - ✅ CatBoost 모델 구현 및 테스트 완료 (정확도 85%, AUC-ROC 0.91)
  - ✅ LightGBM 모델 구현 및 테스트 완료 (정확도 84%, AUC-ROC 0.90)
  - ✅ Random Forest 모델 구현 및 테스트 완료 (정확도 83%, AUC-ROC 0.89)
  - ✅ 실험 스크립트 통합 완료 (run_experiment.py 삭제)
  - ✅ ConfigManager 기반 리샘플링 비교 실험 구현 완료
  - ✅ 모든 모델 테스트 완료 (XGBoost, CatBoost, LightGBM, Random Forest)
- **현재 성능**: 
  - XGBoost: 정확도 99.87% (불균형 데이터 특성)
  - CatBoost: 정확도 85%, AUC-ROC 0.91 (최고 성능)
  - LightGBM: 정확도 84%, AUC-ROC 0.90 (우수한 성능)
  - Random Forest: 정확도 83%, AUC-ROC 0.89 (안정적 성능)
  - 재현율/정밀도/F1: 0.0 (소수 클래스 예측 어려움)
  - Balanced Accuracy: 0.5011 ± 0.0009
- **도전과제**: 
  - 극도 불균형 데이터 (자살 시도 849:1)
  - 시계열 특성을 고려한 모델 개발 필요
  - 모델 해석 가능성 확보
- **불균형 데이터 처리의 잔여 이슈**: 
    - XGBoost 실험에서 objective 파라미터 처리 문제(명시적 제거 필요)와 custom objective 사용 시 파라미터 전달 주의 필요
    - 예측/후처리 단계에서 `inverse_transform` 관련 에러 발생(타겟 인코더 None 처리 필요)
    - 전체 파이프라인 안정화 및 후처리 로직 개선이 추후 필요

## 🔧 6개 주요 작업 영역

### 1. 고급 모델 개발 및 확장 ✅ (4개 모델 완료)
- **✅ 완료된 모델**
  - CatBoost: 범주형 변수 처리 강점, 85% 정확도, 0.91 AUC-ROC 달성
  - LightGBM: 빠른 학습 속도와 높은 성능, 84% 정확도, 0.90 AUC-ROC 달성
  - Random Forest: 해석 가능성과 안정성, 83% 정확도, 0.89 AUC-ROC 달성
  - XGBoost: Focal Loss 통합, 극단적 불균형 데이터 처리
- **✅ 완료된 시스템**
  - 모델 아키텍처 표준화: BaseModel 추상 클래스와 ModelFactory를 통한 일관된 인터페이스
  - 계층적 ConfigManager 시스템: 모듈화된 설정 관리 및 자동 병합
  - 실험 스크립트 통합: run_experiment.py 삭제 및 run_hyperparameter_tuning.py로 통합
  - ConfigManager 기반 리샘플링 비교 실험: 계층적 config 시스템을 활용한 리샘플링 기법 비교 및 하이퍼파라미터 튜닝 통합 완성
  - 모든 모델 테스트 완료: XGBoost, CatBoost, LightGBM, Random Forest 모두 ConfigManager 기반 리샘플링 비교 실험 정상 동작 확인
- **앙상블 모델** (다음 단계)
  - Stacking: 다중 모델의 예측을 메타 모델로 조합
  - Blending: 검증 세트 기반 모델 가중치 최적화
  - Voting: 하드/소프트 보팅을 통한 앙상블
- **Focal Loss 파이프라인 안정화**: objective 파라미터 처리, 후처리(inverse_transform) 에러 등 잔여 이슈 정리 및 재현성 확보 필요

### 2. 피처 엔지니어링 고도화 (진행 예정)
- **시계열 피처 확장**
  - 계절성 패턴: 월별, 계절별 변동 패턴
  - 추세 분석: 장기적 변화 추세
  - 변동성 피처: 표준편차, 변동계수 등
- **도메인 특화 피처**
  - 의료 지식 기반 피처: 진단 코드 조합, 약물 상호작용
  - 상호작용 피처: 주요 변수 간 상호작용
- **피처 선택 및 차원 축소**
  - SHAP 기반 중요도 분석
  - Recursive Feature Elimination (RFE)
  - Principal Component Analysis (PCA)

### 3. 하이퍼파라미터 튜닝 최적화 (진행 예정)
- **고급 최적화 기법**
  - Bayesian Optimization: 효율적인 탐색
  - Population-based Training: 동적 하이퍼파라미터 조정
- **모델별 튜닝 전략**
  - 시계열 모델 전용 튜닝
  - 앙상블 모델 전용 튜닝
- **자동화된 튜닝 파이프라인**
  - AutoML 통합
  - 실험 자동화

### 4. 모델 해석 및 설명 가능성 (진행 예정)
- **SHAP 분석 고도화**
  - 개별 예측 해석
  - 피처 상호작용 분석
  - 글로벌 중요도 분석
- **모델 해석 도구**
  - LIME: 지역적 해석
  - Partial Dependence Plot: 피처 영향도 시각화
- **의료 현장 적용**
  - 위험도 분류 시스템
  - 개입 시점 제안

### 5. 실험 관리 및 배포 준비 (진행 예정)
- **실험 관리 시스템 고도화**
  - 실험 버전 관리
  - 모델 아카이브
- **배포 준비**
  - 모델 서빙
  - 배치/실시간 예측
- **품질 관리**
  - 모델 모니터링
  - 성능 추적

### 6. 앙상블 모델 개발 (다음 우선순위)
- **Stacking 앙상블**
  - 베이스 모델: XGBoost, CatBoost, LightGBM, Random Forest
  - 메타 모델: Logistic Regression, Random Forest
  - 교차 검증 기반 스태킹
- **Blending 앙상블**
  - 검증 세트 기반 가중치 최적화
  - 모델별 성능에 따른 가중치 조정
- **Voting 앙상블**
  - 하드 보팅: 다수결 기반
  - 소프트 보팅: 확률 기반

## 📈 Phase 5-4 세부 목표 및 진행 상황

### ✅ 완료된 목표
1. **모델 다양성 확보**: 4개 모델 구현 완료
   - XGBoost (기존)
   - CatBoost (85% 정확도, 0.91 AUC-ROC)
   - LightGBM (84% 정확도, 0.90 AUC-ROC)
   - Random Forest (83% 정확도, 0.89 AUC-ROC)

2. **모델 아키텍처 표준화**: BaseModel과 ModelFactory 구현
   - 일관된 인터페이스로 모든 모델 통합
   - 설정 파일 기반 모델 관리
   - 확장 가능한 구조 완성

3. **계층적 ConfigManager 시스템**: 모듈화된 설정 관리
   - base, models, experiments, templates로 설정 분리
   - 자동 설정 병합 및 검증
   - 백워드 호환성 지원

4. **통합 실험 파이프라인**: 하이퍼파라미터 튜닝 스크립트 통합
   - 모델 타입별 자동 설정 로딩
   - Optuna 기반 최적화
   - MLflow 실험 관리

5. **실험 스크립트 통합**: run_experiment.py 삭제 및 통합
   - 모든 실험을 run_hyperparameter_tuning.py로 통합
   - 계층적 ConfigManager 기반 자동 설정 병합
   - 모델별 테스트 완료

6. **ConfigManager 기반 리샘플링 비교 실험**: 계층적 config 시스템을 활용한 리샘플링 기법 비교 및 하이퍼파라미터 튜닝 통합 완성
   - 각 리샘플링 기법별로 ConfigManager를 통해 config를 생성/수정하여 실험을 반복 실행
   - MLflow 중첩 실행 문제 해결
   - 모든 모델에서 정상 동작 확인

7. **모든 모델 테스트 완료**: XGBoost, CatBoost, LightGBM, Random Forest 모두 ConfigManager 기반 리샘플링 비교 실험 정상 동작 확인
   - 각 모델별로 리샘플링 기법 비교 실험 성공
   - MLflow를 통한 실험 추적 완성
   - 최고 성능 리샘플링 기법 자동 선별

### 🔄 진행 중인 목표
1. **앙상블 모델 개발**: Stacking, Blending, Voting 구현
2. **모델 성능 비교 분석**: 통합 성능 평가 및 분석

### 📋 남은 목표
1. **피처 엔지니어링 고도화**: 시계열 피처, 도메인 특화 피처
2. **하이퍼파라미터 튜닝 최적화**: 고급 최적화 기법 적용
3. **모델 해석 및 설명 가능성**: SHAP 분석 고도화
4. **실험 관리 및 배포 준비**: 시스템 고도화

## 🎯 성능 목표
- **F1-Score**: 0.0 → 0.3 이상
- **재현율**: 소수 클래스 예측 성능 크게 개선
- **모델 다양성**: 5개 이상 고급 모델 (현재 4개 완료)
- **해석 가능성**: SHAP 분석 완성
- **실험 자동화**: 자동화된 실험 파이프라인 구축

## 📊 예상 성과
- **앙상블 모델**: 개별 모델보다 2-5% 성능 향상 예상
- **피처 엔지니어링**: 3-8% 성능 향상 예상
- **통합 최적화**: 전체적으로 5-10% 성능 향상 목표

## 🔄 다음 단계 계획

### 앙상블 모델 개발 (우선순위 1)
1. **Stacking 앙상블 구현**
   - 베이스 모델: XGBoost, CatBoost, LightGBM, Random Forest
   - 메타 모델: Logistic Regression, Random Forest
   - 교차 검증 기반 스태킹 파이프라인

2. **Blending 앙상블 구현**
   - 검증 세트 기반 가중치 최적화
   - 모델별 성능에 따른 가중치 조정
   - 성능 기반 자동 가중치 계산

3. **Voting 앙상블 구현**
   - 하드 보팅: 다수결 기반 분류
   - 소프트 보팅: 확률 기반 가중 평균
   - 앙상블 성능 비교 및 최적 조합 탐색

### 피처 엔지니어링 고도화 (우선순위 2)
1. **시계열 피처 확장**
   - 계절성 패턴 분석 및 피처 생성
   - 추세 분석 및 변화율 피처
   - 변동성 및 안정성 지표

2. **도메인 특화 피처**
   - 의료 지식 기반 피처 엔지니어링
   - 변수 간 상호작용 피처
   - 위험도 지표 및 복합 점수

### 모델 해석 및 설명 가능성 (우선순위 3)
1. **SHAP 분석 고도화**
   - 개별 예측 해석 시스템
   - 피처 상호작용 분석
   - 글로벌 중요도 및 지역적 중요도

2. **의료 현장 적용 준비**
   - 위험도 분류 시스템 구축
   - 개입 시점 제안 시스템
   - 의료진을 위한 해석 가능한 결과 제공

---
**최종 업데이트**: 2025년 06월 23일
**현재 상태**: Phase 5-4 진행 중 ✅ (4개 모델 완료, ConfigManager 기반 리샘플링 비교 실험 완료, 앙상블 모델 개발 예정)


