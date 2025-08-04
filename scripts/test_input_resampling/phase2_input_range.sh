#!/bin/bash

# Phase 2: Input 범위 조정 실험
# 목적: 데이터 크기와 피처 선택이 성능에 미치는 영향 분석

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

# 안전한 모델 실행 함수
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

echo "==================================================================="
echo "Phase 2: Input 범위 조정 실험 시작"
echo "- 데이터 크기별 성능 비교 (10K, 100K, 500K, 전체)"
echo "- 피처 선택 실험 (10개, 15개, 전체)"
echo "- XGBoost와 CatBoost 중심으로 실험"
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
echo "Phase 2-1: 데이터 크기별 성능 비교"
echo "========================================="

echo ""
echo ">>> 2-1-1: 소규모 데이터 (10K) - XGBoost"
echo "----------------------------------------"
run_model "xgboost" "phase2_input_10k_xgboost" "--nrows 10000 --n-trials 30"

echo ""
echo ">>> 2-1-2: 중규모 데이터 (100K) - XGBoost"
echo "-----------------------------------------"
run_model "xgboost" "phase2_input_100k_xgboost" "--nrows 100000 --n-trials 30"

echo ""
echo ">>> 2-1-3: 대규모 데이터 (500K) - XGBoost"
echo "-----------------------------------------"
run_model "xgboost" "phase2_input_500k_xgboost" "--nrows 500000 --n-trials 30"

echo ""
echo ">>> 2-1-4: CatBoost로도 중규모 데이터 테스트"
echo "--------------------------------------------"
run_model "catboost" "phase2_input_100k_catboost" "--nrows 100000 --n-trials 30"

# Phase 간 강력한 메모리 정리
echo "Phase 2-1과 Phase 2-2 사이 강력한 메모리 정리..."
cleanup_memory

echo "Phase 2-2 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

echo ""
echo "========================================="
echo "Phase 2-2: 피처 선택 실험"
echo "========================================="

echo ""
echo ">>> 2-2-1: 피처 선택 (상위 10개) - CatBoost"
echo "-------------------------------------------"
run_model "catboost" "phase2_feature_select_10_catboost" "--feature-selection --feature-selection-method mutual_info --feature-selection-k 10 --n-trials 40"

echo ""
echo ">>> 2-2-2: 피처 선택 (상위 15개) - CatBoost"
echo "-------------------------------------------"
run_model "catboost" "phase2_feature_select_15_catboost" "--feature-selection --feature-selection-method mutual_info --feature-selection-k 15 --n-trials 40"

echo ""
echo ">>> 2-2-3: 다른 피처 선택 방법 테스트 (Chi2)"
echo "---------------------------------------------"
run_model "xgboost" "phase2_feature_select_chi2_xgboost" "--feature-selection --feature-selection-method chi2 --feature-selection-k 12 --n-trials 40"

echo ""
echo ">>> 2-2-4: Recursive Feature Elimination 테스트"
echo "-----------------------------------------------"
run_model "lightgbm" "phase2_feature_select_rfe_lightgbm" "--feature-selection --feature-selection-method recursive --feature-selection-k 10 --n-trials 40"

# 최종 메모리 정리
echo "최종 메모리 정리..."
cleanup_memory

# 실험 종료 시간 기록
end_time=$(date)
echo ""
echo "==================================================================="
echo "Phase 2 실험 완료!"
echo "실험 시작: $start_time"
echo "실험 종료: $end_time"
echo ""
echo "실험 결과 요약:"
echo "- 데이터 크기별 실험: 4개 (10K, 100K, 500K, 전체)"
echo "- 피처 선택 실험: 4개 (mutual_info, chi2, recursive)"
echo ""
echo "실행된 실험:"
echo "데이터 크기별:"
echo "  - phase2_input_10k_xgboost_${TIMESTAMP}"
echo "  - phase2_input_100k_xgboost_${TIMESTAMP}"
echo "  - phase2_input_500k_xgboost_${TIMESTAMP}"
echo "  - phase2_input_100k_catboost_${TIMESTAMP}"
echo "피처 선택:"
echo "  - phase2_feature_select_10_catboost_${TIMESTAMP}"
echo "  - phase2_feature_select_15_catboost_${TIMESTAMP}"
echo "  - phase2_feature_select_chi2_xgboost_${TIMESTAMP}"
echo "  - phase2_feature_select_rfe_lightgbm_${TIMESTAMP}"
echo ""
echo "다음 단계:"
echo "1. MLflow UI에서 Phase 1과 Phase 2 결과 비교"
echo "2. 최적 데이터 크기와 피처 수 확인"
echo "3. Phase 3 실행: ./phase3_resampling.sh"
echo "==================================================================="

# 간단한 결과 요약 생성
echo ""
echo "Phase 2 실험 요약 리포트 생성 중..."
cat > phase2_experiment_summary.txt << EOF
Phase 2: Input 범위 조정 실험 완료 리포트
========================================

실험 기간: $start_time ~ $end_time

실행된 실험:
1. 데이터 크기별 성능 비교
   - 10K rows (XGBoost): phase2_input_10k_xgboost_${TIMESTAMP}
   - 100K rows (XGBoost): phase2_input_100k_xgboost_${TIMESTAMP}
   - 500K rows (XGBoost): phase2_input_500k_xgboost_${TIMESTAMP}
   - 100K rows (CatBoost): phase2_input_100k_catboost_${TIMESTAMP}

2. 피처 선택 실험
   - Mutual Info (10개): phase2_feature_select_10_catboost_${TIMESTAMP}
   - Mutual Info (15개): phase2_feature_select_15_catboost_${TIMESTAMP}
   - Chi2 (12개): phase2_feature_select_chi2_xgboost_${TIMESTAMP}
   - RFE (10개): phase2_feature_select_rfe_lightgbm_${TIMESTAMP}

다음 단계:
- MLflow UI에서 결과 분석
- 최적 설정 도출 후 Phase 3 진행

MLflow UI 접속: http://localhost:5000
EOF

echo "Phase 2 요약 리포트가 'phase2_experiment_summary.txt'에 저장되었습니다."

# 시그널 핸들러 제거
trap - INT TERM