#!/bin/bash

# Phase 4: 4개 모델 심층 분석
# 목적: 각 모델의 특성과 상호 비교를 통한 최적 모델 선정

# 실험 중단 처리 함수
cleanup_on_exit() {
    echo ""
    echo "⚠️  실험이 중단되었습니다."
    echo "현재 진행 중인 프로세스를 정리합니다..."
    pkill -f "run_hyperparameter_tuning"
    pkill -f "run_resampling_experiment"
    echo "정리 완료."
    exit 1
}

# 시그널 핸들러 설정
trap cleanup_on_exit INT TERM

echo "==================================================================="
echo "Phase 4: 4개 모델 심층 분석 실험 시작"
echo "- 고성능 하이퍼파라미터 튜닝 (200 trials)"
echo "- 시계열 분할 전략별 성능 비교"
echo "- 모델별 특성 및 강점 분석"
echo "- 앙상블 가능성 탐색"
echo "==================================================================="

# 실험 시작 시간 기록
start_time=$(date)
echo "실험 시작 시간: $start_time"

# 메모리 사용량 체크
echo "현재 시스템 상태 체크 중..."
python scripts/check_cpu_usage.py

echo ""
echo "========================================="
echo "Phase 4-1: 고성능 하이퍼파라미터 튜닝"
echo "각 모델별 200 trials로 심층 튜닝"
echo "========================================="

echo ""
echo ">>> 4-1-1: XGBoost 고성능 튜닝 (200 trials)"
echo "------------------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type xgboost \
    --experiment-type hyperparameter_tuning \
    --n-trials 200 \
    --primary-metric f1 \
    --early-stopping \
    --early-stopping-rounds 50 \
    --save-model \
    --save-predictions \
    --experiment-name "phase4_deep_tuning_xgboost" \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ XGBoost 고성능 튜닝 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-1-2: CatBoost 고성능 튜닝 (200 trials)"
echo "--------------------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type catboost \
    --experiment-type hyperparameter_tuning \
    --n-trials 200 \
    --primary-metric pr_auc \
    --early-stopping \
    --early-stopping-rounds 50 \
    --save-model \
    --save-predictions \
    --experiment-name "phase4_deep_tuning_catboost" \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ CatBoost 고성능 튜닝 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-1-3: LightGBM 고성능 튜닝 (200 trials)"
echo "--------------------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type lightgbm \
    --experiment-type hyperparameter_tuning \
    --n-trials 200 \
    --primary-metric mcc \
    --early-stopping \
    --early-stopping-rounds 50 \
    --save-model \
    --save-predictions \
    --experiment-name "phase4_deep_tuning_lightgbm" \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ LightGBM 고성능 튜닝 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-1-4: Random Forest 고성능 튜닝 (200 trials)"
echo "------------------------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type random_forest \
    --experiment-type hyperparameter_tuning \
    --n-trials 200 \
    --primary-metric balanced_accuracy \
    --save-model \
    --save-predictions \
    --experiment-name "phase4_deep_tuning_random_forest" \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ Random Forest 고성능 튜닝 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo "========================================="
echo "Phase 4-2: 시계열 분할 전략별 성능 비교"
echo "========================================="

echo ""
echo ">>> 4-2-1: Time Series Walk Forward - CatBoost"
echo "----------------------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type catboost \
    --experiment-type hyperparameter_tuning \
    --split-strategy time_series_walk_forward \
    --cv-folds 5 \
    --n-trials 100 \
    --experiment-name "phase4_ts_walk_forward_catboost" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ Time Series Walk Forward 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-2-2: Time Series Group K-Fold - CatBoost"
echo "----------------------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type catboost \
    --experiment-type hyperparameter_tuning \
    --split-strategy time_series_group_kfold \
    --cv-folds 5 \
    --n-trials 100 \
    --experiment-name "phase4_ts_group_kfold_catboost" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ Time Series Group K-Fold 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-2-3: Group K-Fold - CatBoost"
echo "----------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type catboost \
    --experiment-type hyperparameter_tuning \
    --split-strategy group_kfold \
    --cv-folds 5 \
    --n-trials 100 \
    --experiment-name "phase4_group_kfold_catboost" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ Group K-Fold 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-2-4: XGBoost로도 시계열 분할 비교"
echo "--------------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type xgboost \
    --experiment-type hyperparameter_tuning \
    --split-strategy time_series_walk_forward \
    --cv-folds 5 \
    --n-trials 100 \
    --experiment-name "phase4_ts_walk_forward_xgboost" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ XGBoost 시계열 분할 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo "========================================="
echo "Phase 4-3: 모델별 특성 분석 실험"
echo "========================================="

echo ""
echo ">>> 4-3-1: XGBoost 특화 실험 (극도 불균형 최적화)"
echo "------------------------------------------------"
python scripts/run_resampling_experiment.py \
    --model-type xgboost \
    --resampling-method smote \
    --n-trials 150 \
    --experiment-name "phase4_xgboost_imbalance_specialized" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ XGBoost 불균형 특화 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-3-2: CatBoost 특화 실험 (범주형 처리 강점)"
echo "-----------------------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type catboost \
    --experiment-type hyperparameter_tuning \
    --primary-metric roc_auc \
    --n-trials 150 \
    --experiment-name "phase4_catboost_categorical_specialized" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ CatBoost 범주형 특화 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-3-3: LightGBM 특화 실험 (속도 vs 성능 균형)"
echo "------------------------------------------------"
python scripts/run_resampling_experiment.py \
    --model-type lightgbm \
    --resampling-method borderline_smote \
    --n-trials 150 \
    --experiment-name "phase4_lightgbm_speed_performance_balance" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ LightGBM 속도-성능 균형 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-3-4: Random Forest 특화 실험 (해석가능성)"
echo "----------------------------------------------"
python scripts/run_hyperparameter_tuning.py \
    --model-type random_forest \
    --experiment-type hyperparameter_tuning \
    --feature-selection \
    --feature-selection-method mutual_info \
    --feature-selection-k 15 \
    --primary-metric balanced_accuracy \
    --n-trials 150 \
    --experiment-name "phase4_random_forest_interpretability" \
    --save-model \
    --save-predictions \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ Random Forest 해석가능성 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo "========================================="
echo "Phase 4-4: 앙상블 준비 실험"
echo "========================================="

echo ""
echo ">>> 4-4-1: 다양한 메트릭 최적화 (앙상블용)"
echo "-----------------------------------------"
# F1 최적화 모델
python scripts/run_hyperparameter_tuning.py \
    --model-type xgboost \
    --experiment-type hyperparameter_tuning \
    --primary-metric f1 \
    --n-trials 100 \
    --experiment-name "phase4_ensemble_prep_f1_xgboost" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ F1 최적화 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

# Precision 최적화 모델
python scripts/run_hyperparameter_tuning.py \
    --model-type catboost \
    --experiment-type hyperparameter_tuning \
    --primary-metric precision \
    --n-trials 100 \
    --experiment-name "phase4_ensemble_prep_precision_catboost" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ Precision 최적화 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

# Recall 최적화 모델
python scripts/run_hyperparameter_tuning.py \
    --model-type lightgbm \
    --experiment-type hyperparameter_tuning \
    --primary-metric recall \
    --n-trials 100 \
    --experiment-name "phase4_ensemble_prep_recall_lightgbm" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ Recall 최적화 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

echo ""
echo ">>> 4-4-2: 다양한 리샘플링으로 다양성 확보"
echo "------------------------------------------"
python scripts/run_resampling_experiment.py \
    --model-type random_forest \
    --resampling-method adasyn \
    --n-trials 100 \
    --experiment-name "phase4_ensemble_prep_adasyn_random_forest" \
    --save-model \
    --verbose 1

# 실험 중단 확인
if [ $? -ne 0 ]; then
    echo "❌ ADASYN 다양성 확보 실험 실패"
    read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cleanup_on_exit
    fi
fi

# 실험 종료 시간 기록
end_time=$(date)
echo ""
echo "==================================================================="
echo "Phase 4 심층 분석 실험 완료!"
echo "실험 시작: $start_time"
echo "실험 종료: $end_time"
echo ""
echo "실험 결과 요약:"
echo "- 고성능 튜닝: 4개 모델 × 200 trials = 800 trials"
echo "- 시계열 분할 비교: 4개 실험"
echo "- 모델별 특성 분석: 4개 실험"
echo "- 앙상블 준비: 4개 실험"
echo "- 총 16개 심층 분석 실험 완료"
echo ""
echo "모델별 기대 특성:"
echo "- XGBoost: 극도 불균형 처리 우수 (99.87% 정확도)"
echo "- CatBoost: 범주형 처리 및 균형잡힌 성능 (85% 정확도, 0.91 AUC)"
echo "- LightGBM: 속도와 성능 균형 (84% 정확도, 0.90 AUC)"
echo "- Random Forest: 해석가능성 및 안정성 (83% 정확도, 0.89 AUC)"
echo ""
echo "다음 단계:"
echo "1. MLflow UI에서 모델별 심층 성능 분석"
echo "2. 최적 하이퍼파라미터 및 설정 도출"
echo "3. Phase 5 실행: ./phase5_final_optimization.sh"
echo "==================================================================="

# Phase 4 실험 요약 리포트 생성
echo ""
echo "Phase 4 심층 분석 실험 요약 리포트 생성 중..."
cat > phase4_deep_analysis_summary.txt << EOF
Phase 4: 4개 모델 심층 분석 완료 리포트
=====================================

실험 기간: $start_time ~ $end_time

실행된 실험 카테고리:
1. 고성능 하이퍼파라미터 튜닝 (4개 실험, 200 trials 각)
   - XGBoost (F1 최적화): phase4_deep_tuning_xgboost
   - CatBoost (PR-AUC 최적화): phase4_deep_tuning_catboost
   - LightGBM (MCC 최적화): phase4_deep_tuning_lightgbm
   - Random Forest (Balanced Accuracy): phase4_deep_tuning_random_forest

2. 시계열 분할 전략 비교 (4개 실험)
   - Time Series Walk Forward: phase4_ts_walk_forward_catboost/xgboost
   - Time Series Group K-Fold: phase4_ts_group_kfold_catboost
   - Group K-Fold: phase4_group_kfold_catboost

3. 모델별 특성 분석 (4개 실험)
   - XGBoost 불균형 특화: phase4_xgboost_imbalance_specialized
   - CatBoost 범주형 특화: phase4_catboost_categorical_specialized
   - LightGBM 균형 특화: phase4_lightgbm_speed_performance_balance
   - Random Forest 해석가능성: phase4_random_forest_interpretability

4. 앙상블 준비 (4개 실험)
   - 다양한 메트릭 최적화: F1, Precision, Recall
   - 다양한 리샘플링: ADASYN

총 16개 심층 분석 실험 완료

각 모델의 예상 강점:
- XGBoost: 극도 불균형 데이터에서 높은 정확도 (scale_pos_weight 활용)
- CatBoost: 범주형 변수 자동 처리, 과적합 방지, 균형잡힌 성능
- LightGBM: 빠른 학습 속도, 메모리 효율성, 좋은 일반화 성능
- Random Forest: 해석 가능성, 안정성, 피처 중요도 신뢰성

분석 포인트:
- 각 모델의 최적 하이퍼파라미터 패턴
- 시계열 분할 전략에 따른 성능 변화
- 불균형 데이터에서의 모델별 특성
- 앙상블 구성을 위한 다양성 확보

MLflow UI 접속: http://localhost:5000
EOF

echo "Phase 4 요약 리포트가 'phase4_deep_analysis_summary.txt'에 저장되었습니다."

# 성능 비교 차트 생성 제안
echo ""
echo "추가 분석 도구:"
echo "- MLflow UI에서 실험별 비교 차트 확인"
echo "- 모델별 성능 매트릭스 비교"
echo "- 하이퍼파라미터 분포 분석"
echo "- 교차 검증 안정성 분석"

# 시그널 핸들러 제거
trap - INT TERM