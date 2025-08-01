#!/bin/bash

# 빠른 테스트용 스크립트
# 소규모 데이터와 적은 n_trials로 빠른 실험 실행

set -e

# 타임스탬프 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "=== 빠른 테스트 시작 ==="
echo "시작 시간: $(date)"
echo "타임스탬프: ${TIMESTAMP}"
echo "n_trials: 10 (빠른 테스트용)"
echo "nrows: 1000 (소규모 데이터)"

# 메모리 제한 설정 (GB)
MEMORY_LIMIT=20
echo "메모리 제한 설정: ${MEMORY_LIMIT}GB"
echo "n_jobs: 2 (빠른 테스트용)"

# 메모리 상태 확인 함수
check_memory() {
    echo "메모리 사용량:"
    free -h
}

# 간단한 메모리 정리 함수
cleanup_memory() {
    echo "메모리 정리 시작..."
    python3 -c "
import gc
gc.collect()
print('Python 메모리 정리 완료')
"
    echo "메모리 정리 완료"
}

# 초기 메모리 상태 확인
echo "초기 메모리 상태:"
check_memory

# 빠른 테스트 실행 함수
run_quick_test() {
    local model_type=$1
    local experiment_name=$2
    
    echo ""
    echo "=== ${model_type} 빠른 테스트 시작 ==="
    echo "시작 시간: $(date)"
    
    python scripts/run_hyperparameter_tuning.py \
        --model-type ${model_type} \
        --experiment-type hyperparameter_tuning \
        --experiment-name "quick_test_${experiment_name}_${TIMESTAMP}" \
        --n-trials 10 \
        --n-jobs 2 \
        --nrows 1000 \
        --verbose 2 \
        > logs/quick_test_${model_type}_${TIMESTAMP}.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ ${model_type} 빠른 테스트 성공"
    else
        echo "❌ ${model_type} 빠른 테스트 실패 (종료 코드: $exit_code)"
        echo "로그 확인: logs/quick_test_${model_type}_${TIMESTAMP}.log"
    fi
    
    echo "완료 시간: $(date)"
    cleanup_memory
    
    return $exit_code
}

# 빠른 테스트 실행
echo "========================================"
echo "빠른 테스트 실행 시작"
echo "========================================"

# CatBoost 빠른 테스트
run_quick_test "catboost" "basic"

# XGBoost 빠른 테스트
run_quick_test "xgboost" "basic"

# LightGBM 빠른 테스트
run_quick_test "lightgbm" "basic"

# Random Forest 빠른 테스트
run_quick_test "random_forest" "basic"

echo ""
echo "========================================"
echo "빠른 테스트 완료"
echo "========================================"
echo "완료 시간: $(date)"

echo ""
echo "빠른 테스트 완료! 로그 파일을 확인하세요:"
echo "로그 디렉토리: logs/"
echo "타임스탬프: ${TIMESTAMP}"
echo ""
echo "실행된 테스트:"
echo "- quick_test_catboost_basic_${TIMESTAMP}"
echo "- quick_test_xgboost_basic_${TIMESTAMP}"
echo "- quick_test_lightgbm_basic_${TIMESTAMP}"
echo "- quick_test_random_forest_basic_${TIMESTAMP}"
echo ""
echo "사용법:"
echo "chmod +x scripts/quick_test.sh && ./scripts/quick_test.sh" 