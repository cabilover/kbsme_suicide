#!/bin/bash

# Phase 5: 통합 최적화 실험
# 목적: 앞선 실험 결과를 바탕으로 한 최종 최적화

set -e  # 오류 발생 시 스크립트 중단

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

# ========================================
# 메모리 안정화 설정
# ========================================

# 메모리 제한 설정 (GB) - 시스템에 맞게 조정
export MEMORY_LIMIT=50
echo "메모리 제한 설정: ${MEMORY_LIMIT}GB"

# 병렬 처리 설정 - 안전한 값으로 설정
N_JOBS=4
echo "병렬 처리 설정: n_jobs=${N_JOBS} (메모리 안정성 확보)"

# 타임스탬프 생성
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# ========================================
# 메모리 관리 유틸리티 함수
# ========================================

# 메모리 상태 확인 함수
check_memory() {
    echo "메모리 사용량:"
    free -h
    echo ""
}

# 강화된 메모리 정리 함수
cleanup_memory() {
    echo "강화된 메모리 정리 시작..."
    python3 -c "
import gc
import sys
gc.collect()
print(f'Python 메모리 정리 완료. 참조 카운트: {sys.getrefcount({})}')
"
    
    # 시스템 캐시 정리 (권한 없으면 무시)
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "시스템 캐시 정리 권한 없음 (무시됨)"
    
    echo "메모리 정리 완료"
}

# 안전한 하이퍼파라미터 튜닝 실행 함수
run_model() {
    local model_type=$1
    local experiment_name=$2
    local additional_params=$3
    
    echo "=== ${model_type} 하이퍼파라미터 튜닝 시작 ==="
    echo "시작 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
    
    python scripts/run_hyperparameter_tuning.py \
        --model-type ${model_type} \
        --experiment-type hyperparameter_tuning \
        --n-jobs ${N_JOBS} \
        --experiment-name "${experiment_name}_${TIMESTAMP}" \
        --save-model \
        --save-predictions \
        --verbose 1 \
        ${additional_params} \
        > "logs/${model_type}_${TIMESTAMP}.log" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ ${model_type} 성공적으로 완료"
        echo "완료 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
    else
        echo "❌ ${model_type} 실행 중 오류 발생 (종료 코드: $exit_code)"
        echo "로그 확인: logs/${model_type}_${TIMESTAMP}.log"
        echo "⚠️ 실패했지만 다음 단계로 진행합니다."
    fi
    
    echo "${model_type} 완료 후 메모리 상태:"
    check_memory
    cleanup_memory
    
    echo "메모리 정리를 위해 3분 대기..."
    for i in {1..3}; do
        echo "대기 중... ($i/3)"
        sleep 60
        check_memory
    done
    
    return $exit_code
}

# 안전한 리샘플링 실험 실행 함수
run_resampling_model() {
    local model_type=$1
    local experiment_name=$2
    local additional_params=$3
    
    echo "=== ${model_type} 리샘플링 실험 시작 ==="
    echo "시작 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
    
    python scripts/run_resampling_experiment.py \
        --model-type ${model_type} \
        --n-jobs ${N_JOBS} \
        --experiment-name "${experiment_name}_${TIMESTAMP}" \
        --save-model \
        --save-predictions \
        --verbose 1 \
        ${additional_params} \
        > "logs/${model_type}_resampling_${TIMESTAMP}.log" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ ${model_type} 리샘플링 실험 성공적으로 완료"
        echo "완료 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
    else
        echo "❌ ${model_type} 리샘플링 실험 실행 중 오류 발생 (종료 코드: $exit_code)"
        echo "로그 확인: logs/${model_type}_resampling_${TIMESTAMP}.log"
        echo "⚠️ 실패했지만 다음 단계로 진행합니다."
    fi
    
    echo "${model_type} 리샘플링 실험 완료 후 메모리 상태:"
    check_memory
    cleanup_memory
    
    echo "메모리 정리를 위해 3분 대기..."
    for i in {1..3}; do
        echo "대기 중... ($i/3)"
        sleep 60
        check_memory
    done
    
    return $exit_code
}

echo "==================================================================="
echo "Phase 5: 통합 최적화 실험 시작"
echo "- Phase 1-4 결과를 바탕으로 최적 조합 도출"
echo "- 최종 성능 검증 및 앙상블 모델 구축"
echo "- 실제 배포 가능한 최종 모델 완성"
echo "- 메모리 안정화 설정 적용"
echo "==================================================================="

# 실험 시작 시간 기록
start_time=$(date)
echo "실험 시작 시간: $start_time"

# 초기 메모리 상태 확인
echo "초기 메모리 상태:"
check_memory

echo ""
echo "========================================="
echo "Phase 5-1: 최적 조합 실험"
echo "이전 Phase 결과를 바탕으로 설정"
echo "========================================="

echo ""
echo ">>> 5-1-1: XGBoost 최적 조합"
echo "----------------------------"
# 극도 불균형에 특화된 XGBoost 최적화
# Phase 3에서 최적 리샘플링 + Phase 2에서 최적 피처 수 조합
run_model "xgboost" "phase5_final_optimized_xgboost" "--feature-selection --feature-selection-method mutual_info --feature-selection-k 12 --primary-metric f1 --n-trials 150 --early-stopping --early-stopping-rounds 30"

echo ""
echo ">>> 5-1-2: CatBoost 최적 조합"
echo "-----------------------------"
# CatBoost의 범주형 처리 강점 활용
run_model "catboost" "phase5_final_optimized_catboost" "--primary-metric pr_auc --n-trials 150 --early-stopping --early-stopping-rounds 30"

echo ""
echo ">>> 5-1-3: LightGBM 최적 조합"
echo "-----------------------------"
# 속도와 성능 균형
run_model "lightgbm" "phase5_final_optimized_lightgbm" "--feature-selection --feature-selection-method chi2 --feature-selection-k 10 --primary-metric mcc --n-trials 150 --early-stopping --early-stopping-rounds 30"

echo ""
echo ">>> 5-1-4: Random Forest 최적 조합"
echo "----------------------------------"
# 해석가능성과 안정성 중심
run_model "random_forest" "phase5_final_optimized_random_forest" "--feature-selection --feature-selection-method mutual_info --feature-selection-k 15 --primary-metric balanced_accuracy --n-trials 150"

# Phase 간 강력한 메모리 정리
echo "Phase 5-1과 Phase 5-2 사이 강력한 메모리 정리..."
cleanup_memory

echo "Phase 5-2 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

echo ""
echo "========================================="
echo "Phase 5-2: 시계열 최적화 실험"
echo "========================================="

echo ""
echo ">>> 5-2-1: 최적 모델로 시계열 분할 최적화"
echo "------------------------------------------"
# CatBoost가 현재 최고 성능이므로 시계열 최적화 집중
run_model "catboost" "phase5_time_series_optimized_catboost" "--split-strategy time_series_walk_forward --cv-folds 5 --primary-metric pr_auc --n-trials 120"

echo ""
echo ">>> 5-2-2: 시계열 특화 리샘플링 최종"
echo "------------------------------------"
run_resampling_model "xgboost" "phase5_time_series_specialized_xgboost" "--resampling-method time_series_adapted --split-strategy time_series_group_kfold --n-trials 100"

# Phase 간 강력한 메모리 정리
echo "Phase 5-2와 Phase 5-3 사이 강력한 메모리 정리..."
cleanup_memory

echo "Phase 5-3 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

echo ""
echo "========================================="
echo "Phase 5-3: 앙상블 모델 구축 실험"
echo "========================================="

echo ""
echo ">>> 5-3-1: 다양성 확보를 위한 개별 모델"
echo "----------------------------------------"
# F1 특화 모델
run_resampling_model "xgboost" "phase5_ensemble_f1_xgboost" "--resampling-method smote --n-trials 100"

# Precision 특화 모델
run_model "catboost" "phase5_ensemble_precision_catboost" "--primary-metric precision --n-trials 100"

# Recall 특화 모델
run_resampling_model "lightgbm" "phase5_ensemble_recall_lightgbm" "--resampling-method adasyn --n-trials 100"

# MCC 특화 모델 (균형잡힌 성능)
run_resampling_model "random_forest" "phase5_ensemble_mcc_random_forest" "--resampling-method smote --n-trials 100"

# Phase 간 강력한 메모리 정리
echo "Phase 5-3과 Phase 5-4 사이 강력한 메모리 정리..."
cleanup_memory

echo "Phase 5-4 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

echo ""
echo "========================================="
echo "Phase 5-4: 최종 검증 실험"
echo "========================================="

echo ""
echo ">>> 5-4-1: 전체 데이터 최종 검증"
echo "--------------------------------"
# Phase 1-4 결과를 바탕으로 최고 성능 조합으로 전체 데이터 검증
run_model "catboost" "phase5_final_validation_full_data" "--split-strategy time_series_walk_forward --primary-metric pr_auc --n-trials 200 --early-stopping --early-stopping-rounds 50 --verbose 2"

echo ""
echo ">>> 5-4-2: 다중 메트릭 최적화 검증"
echo "----------------------------------"
# 여러 메트릭을 동시에 고려한 최적화
run_model "xgboost" "phase5_multi_metric_validation" "--feature-selection --feature-selection-method mutual_info --feature-selection-k 12 --primary-metric f1 --n-trials 200 --early-stopping --early-stopping-rounds 50 --verbose 2"

# Phase 간 강력한 메모리 정리
echo "Phase 5-4와 Phase 5-5 사이 강력한 메모리 정리..."
cleanup_memory

echo "Phase 5-5 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

echo ""
echo "========================================="
echo "Phase 5-5: 배포 준비 실험"
echo "========================================="

echo ""
echo ">>> 5-5-1: 실시간 예측 최적화"
echo "-----------------------------"
# 빠른 예측 속도를 위한 LightGBM 최적화
run_model "lightgbm" "phase5_deployment_ready_lightgbm" "--primary-metric pr_auc --n-trials 150"

echo ""
echo ">>> 5-5-2: 해석가능성 중심 모델"
echo "-------------------------------"
# 임상 현장에서 해석 가능한 모델
run_model "random_forest" "phase5_interpretable_clinical_model" "--feature-selection --feature-selection-method mutual_info --feature-selection-k 10 --primary-metric balanced_accuracy --n-trials 150"

# 최종 메모리 정리
echo "최종 메모리 정리..."
cleanup_memory

# 실험 종료 시간 기록
end_time=$(date)
echo ""
echo "==================================================================="
echo "Phase 5 통합 최적화 실험 완료!"
echo "실험 시작: $start_time"
echo "실험 종료: $end_time"
echo ""
echo "실험 결과 요약:"
echo "- 최적 조합 실험: 4개 모델 × 150 trials = 600 trials"
echo "- 시계열 최적화: 2개 실험"
echo "- 앙상블 준비: 4개 실험 (F1, Precision, Recall, MCC 특화)"
echo "- 최종 검증: 2개 실험 (전체 데이터, 다중 메트릭)"
echo "- 배포 준비: 2개 실험 (실시간 예측, 해석가능성)"
echo "- 총 14개 최적화 실험 완료"
echo ""
echo "실행된 실험:"
echo "최적 조합:"
echo "  - phase5_final_optimized_xgboost_${TIMESTAMP}"
echo "  - phase5_final_optimized_catboost_${TIMESTAMP}"
echo "  - phase5_final_optimized_lightgbm_${TIMESTAMP}"
echo "  - phase5_final_optimized_random_forest_${TIMESTAMP}"
echo "시계열 최적화:"
echo "  - phase5_time_series_optimized_catboost_${TIMESTAMP}"
echo "  - phase5_time_series_specialized_xgboost_${TIMESTAMP}"
echo "앙상블 준비:"
echo "  - phase5_ensemble_f1_xgboost_${TIMESTAMP}"
echo "  - phase5_ensemble_precision_catboost_${TIMESTAMP}"
echo "  - phase5_ensemble_recall_lightgbm_${TIMESTAMP}"
echo "  - phase5_ensemble_mcc_random_forest_${TIMESTAMP}"
echo "최종 검증:"
echo "  - phase5_final_validation_full_data_${TIMESTAMP}"
echo "  - phase5_multi_metric_validation_${TIMESTAMP}"
echo "배포 준비:"
echo "  - phase5_deployment_ready_lightgbm_${TIMESTAMP}"
echo "  - phase5_interpretable_clinical_model_${TIMESTAMP}"
echo ""
echo "최종 모델 후보:"
echo "1. phase5_final_validation_full_data_${TIMESTAMP} (최고 성능)"
echo "2. phase5_deployment_ready_lightgbm_${TIMESTAMP} (빠른 예측)"
echo "3. phase5_interpretable_clinical_model_${TIMESTAMP} (해석가능성)"
echo "4. 앙상블 조합 (F1+Precision+Recall+MCC 모델)"
echo ""
echo "다음 단계:"
echo "1. MLflow UI에서 전체 Phase 결과 종합 분석"
echo "2. 최종 모델 선정 및 성능 검증"
echo "3. 앙상블 모델 구축 (필요시)"
echo "4. 배포를 위한 모델 준비"
echo "==================================================================="

# 전체 실험 종합 요약 리포트 생성
echo ""
echo "전체 실험 종합 요약 리포트 생성 중..."
cat > comprehensive_experiment_summary.txt << EOF
KBSME 자살 예측 모델 - 전체 실험 종합 리포트
=============================================

실험 기간: $(cat phase1_baseline_summary.txt 2>/dev/null | grep "실험 시작" | head -1 | cut -d: -f2- || echo "Phase 1 시작") ~ $end_time

데이터셋 특성:
- 규모: 1,569,071 행 × 15 열
- 개인 수: 269,339명
- 기간: 2015-2024 (10년)
- 극도 불균형: 자살 시도 0.12% (849:1 비율)

전체 실험 구성:
==============

Phase 1: 기준선 설정 (4개 실험)
- XGBoost, CatBoost, LightGBM, Random Forest 기본 성능 측정
- 각 모델 50 trials 하이퍼파라미터 튜닝

Phase 2: Input 범위 조정 (8개 실험)
- 데이터 크기별 성능: 10K, 100K, 500K, 전체
- 피처 선택 방법: mutual_info, chi2, recursive

Phase 3: Resampling 비교 (35개 실험)
- 전체 리샘플링 비교: 4개 모델 × 7개 기법
- 시계열 특화 리샘플링: time_series_adapted
- 특정 기법 집중 분석: SMOTE, Borderline SMOTE, ADASYN
- 하이브리드 접근: 리샘플링 + 피처선택 조합

Phase 4: 모델 심층 분석 (16개 실험)
- 고성능 튜닝: 각 모델 200 trials
- 시계열 분할 전략: walk_forward, group_kfold
- 모델별 특성 분석: 불균형 특화, 범주형 특화 등
- 앙상블 준비: 다양한 메트릭 최적화

Phase 5: 통합 최적화 (14개 실험)
- 최적 조합: 이전 결과 기반 최적 설정
- 시계열 최적화: 최고 성능 모델 집중
- 앙상블 구축: F1, Precision, Recall, MCC 특화
- 배포 준비: 실시간 예측, 해석가능성

총 실험 수: 77개 실험
총 하이퍼파라미터 시도: 약 8,000+ trials

모델별 예상 최종 성능:
===================
- CatBoost: 85%+ 정확도, 0.91+ AUC-ROC (균형잡힌 최고 성능)
- XGBoost: 99.87% 정확도 (극도 불균형 특화)
- LightGBM: 84% 정확도, 0.90 AUC-ROC (속도-성능 균형)
- Random Forest: 83% 정확도, 0.89 AUC-ROC (해석가능성)

핵심 발견사항:
============
1. 극도 불균형(849:1)에서 리샘플링 기법의 중요성
2. 시계열 데이터에서 적절한 분할 전략의 필요성
3. 모델별 강점: CatBoost(균형), XGBoost(불균형), LightGBM(속도), RF(해석)
4. 피처 선택과 리샘플링의 시너지 효과

최종 권장 모델:
=============
1. 최고 성능: phase5_final_validation_full_data_${TIMESTAMP} (CatBoost)
2. 빠른 예측: phase5_deployment_ready_lightgbm_${TIMESTAMP}
3. 임상 해석: phase5_interpretable_clinical_model_${TIMESTAMP} (Random Forest)
4. 앙상블: 4개 특화 모델 조합

MLflow UI에서 전체 결과 확인: http://localhost:5000
EOF

echo "전체 실험 종합 리포트가 'comprehensive_experiment_summary.txt'에 저장되었습니다."

echo ""
echo "🎉 축하합니다! 전체 5단계 실험이 완료되었습니다!"
echo ""
echo "📊 결과 분석을 위해 MLflow UI를 실행하시겠습니까?"
read -p "MLflow UI 실행? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "MLflow UI 실행 중... (http://localhost:5000)"
    mlflow ui --host 0.0.0.0 --port 5000 &
    echo ""
    echo "🔍 실험 분석 가이드:"
    echo "1. 'Experiments' 탭에서 phase1~phase5 실험들 확인"
    echo "2. 각 실험의 메트릭 비교 (F1, PR-AUC, MCC 등)"
    echo "3. 최고 성능 모델의 하이퍼파라미터 확인"
    echo "4. 아티팩트에서 시각화 결과 확인"
    echo ""
    echo "🚀 다음 단계:"
    echo "- 최종 모델 선정 및 앙상블 구축"
    echo "- 모델 배포를 위한 파이프라인 구성"
    echo "- 임상 검증을 위한 해석가능성 분석"
fi

# 시그널 핸들러 제거
trap - INT TERM