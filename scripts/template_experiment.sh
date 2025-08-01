#!/bin/bash

# 실험 스크립트 템플릿
# 메모리 안전하고 안정적인 머신러닝 실험을 위한 기본 구조

set -e  # 오류 발생 시 스크립트 중단

# ========================================
# 설정 섹션
# ========================================
echo "=========================================="
echo "실험 시작: [실험 이름을 여기에 입력하세요]"
echo "=========================================="
echo "시작 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
echo "타임스탬프: $(date '+%Y%m%d_%H%M%S')"
echo ""

# 타임스탬프 생성
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# 메모리 제한 설정 (GB) - 시스템에 맞게 조정
export MEMORY_LIMIT=50
echo "메모리 제한 설정: ${MEMORY_LIMIT}GB"

# 병렬 처리 설정 - 안전한 값으로 설정
N_JOBS=4
echo "병렬 처리 설정: n_jobs=${N_JOBS} (메모리 안정성 확보)"

# ========================================
# 유틸리티 함수
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

# 모델 실행 함수 (안전한 설정)
run_model() {
    local model_type=$1
    local experiment_name=$2
    local additional_params=$3
    
    echo "=== ${model_type} 하이퍼파라미터 튜닝 시작 ==="
    echo "시작 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
    
    python scripts/run_hyperparameter_tuning.py \
        --model-type ${model_type} \
        --experiment-type hyperparameter_tuning \
        --experiment-name "${experiment_name}_${TIMESTAMP}" \
        --n-trials 100 \
        --n-jobs ${N_JOBS} \
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

# ========================================
# 초기 설정
# ========================================
echo "초기 메모리 상태:"
check_memory

# ========================================
# Phase 1: 기본 모델 실행
# ========================================
echo "=========================================="
echo "Phase 1: 기본 모델 실행 시작"
echo "=========================================="

# CatBoost (기본)
run_model "catboost" "catboost_basic" ""

# Random Forest (기본)
run_model "random_forest" "random_forest_basic" ""

echo "=== Phase 1 완료 ==="
echo "완료 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"

# ========================================
# Phase 간 강력한 메모리 정리
# ========================================
echo "Phase 1과 Phase 2 사이 강력한 메모리 정리..."
cleanup_memory

echo "Phase 2 시작 전 15분 대기..."
for i in {1..15}; do
    echo "대기 중... ($i/15)"
    sleep 60
    check_memory
done

# ========================================
# Phase 2: SMOTE 리샘플링 모델 실행
# ========================================
echo "=========================================="
echo "Phase 2: SMOTE 리샘플링 모델 실행 시작"
echo "=========================================="

# XGBoost (SMOTE)
run_model "xgboost" "xgboost_smote" "--resampling-enabled --resampling-method smote --resampling-ratio 0.5"

# LightGBM (SMOTE)
run_model "lightgbm" "lightgbm_smote" "--resampling-enabled --resampling-method smote --resampling-ratio 0.5"

# CatBoost (SMOTE)
run_model "catboost" "catboost_smote" "--resampling-enabled --resampling-method smote --resampling-ratio 0.5"

# Random Forest (SMOTE)
run_model "random_forest" "random_forest_smote" "--resampling-enabled --resampling-method smote --resampling-ratio 0.5"

# ========================================
# Phase 3: 추가 실험 (필요시 주석 해제)
# ========================================
# echo "=========================================="
# echo "Phase 3: 추가 실험 시작"
# echo "=========================================="
# 
# # ADASYN 리샘플링 실험 예시
# run_model "xgboost" "xgboost_adasyn" "--resampling-enabled --resampling-method adasyn --resampling-ratio 0.5"
# run_model "catboost" "catboost_adasyn" "--resampling-enabled --resampling-method adasyn --resampling-ratio 0.5"
# 
# # 피처 선택 실험 예시
# run_model "lightgbm" "lightgbm_feature_selection" "--feature-selection --feature-selection-method mutual_info --feature-selection-k 10"
# run_model "random_forest" "random_forest_feature_selection" "--feature-selection --feature-selection-method chi2 --feature-selection-k 15"

# ========================================
# 최종 정리
# ========================================
echo "최종 메모리 정리..."
cleanup_memory

echo "=========================================="
echo "전체 실행 완료"
echo "=========================================="
echo "완료 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
echo ""
echo "실행 완료! 로그 파일을 확인하세요:"
echo "로그 디렉토리: logs/"
echo "타임스탬프: ${TIMESTAMP}"
echo ""
echo "실행된 실험:"
echo "Phase 1 (기본):"
echo "  - catboost_basic_${TIMESTAMP}"
echo "  - random_forest_basic_${TIMESTAMP}"
echo "Phase 2 (SMOTE):"
echo "  - xgboost_smote_${TIMESTAMP}"
echo "  - lightgbm_smote_${TIMESTAMP}"
echo "  - catboost_smote_${TIMESTAMP}"
echo "  - random_forest_smote_${TIMESTAMP}"
echo ""
echo "사용법:"
echo "1. 이 템플릿을 복사하여 새로운 실험 스크립트 생성"
echo "2. 실험 이름과 파라미터 수정"
echo "3. Phase 3 주석 해제하여 추가 실험 실행"
echo "4. chmod +x 스크립트명 && ./스크립트명 으로 실행" 