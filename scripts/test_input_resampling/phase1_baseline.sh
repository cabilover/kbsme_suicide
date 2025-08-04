#!/bin/bash

# Phase 1: 기준선 설정 (Baseline Establishment)
# 목적: 현재 시스템의 성능 기준선 확립

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

# 메모리 임계값 설정 (GB) - 이 값을 넘으면 자동 정리
MEMORY_THRESHOLD=40
echo "메모리 임계값 설정: ${MEMORY_THRESHOLD}GB"

# 타임스탬프 생성
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# ========================================
# 강화된 메모리 관리 유틸리티 함수
# ========================================

# 메모리 사용량 확인 함수 (GB 단위)
get_memory_usage() {
    local memory_usage=$(free -g | awk 'NR==2{printf "%.1f", $3*100/$2}')
    echo "$memory_usage"
}

# 메모리 상태 확인 함수
check_memory() {
    echo "메모리 사용량:"
    free -h
    local usage=$(get_memory_usage)
    echo "메모리 사용률: ${usage}%"
    echo ""
}

# 실시간 메모리 모니터링 함수
monitor_memory() {
    local usage=$(get_memory_usage)
    if (( $(echo "$usage > $MEMORY_THRESHOLD" | bc -l) )); then
        echo "⚠️  메모리 사용률이 임계값(${MEMORY_THRESHOLD}%)을 초과했습니다: ${usage}%"
        echo "자동 메모리 정리를 수행합니다..."
        cleanup_memory
    fi
}

# 강화된 메모리 정리 함수
cleanup_memory() {
    echo "강화된 메모리 정리 시작..."
    
    # Python 메모리 정리
    python3 -c "
import gc
import sys
import psutil
import os

# 가비지 컬렉션 강제 실행
gc.collect()

# 메모리 사용량 확인
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f'Python 프로세스 메모리 사용량: {memory_info.rss / 1024 / 1024:.1f} MB')
print(f'Python 메모리 정리 완료. 참조 카운트: {sys.getrefcount({})}')
"
    
    # 시스템 캐시 정리 (권한 없으면 무시)
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "시스템 캐시 정리 권한 없음 (무시됨)"
    
    # 추가 대기 시간으로 메모리 안정화
    sleep 5
    
    echo "메모리 정리 완료"
    check_memory
}

# 안전한 모델 실행 함수 (메모리 모니터링 포함)
run_model() {
    local model_type=$1
    local experiment_name=$2
    local additional_params=$3
    
    echo "=== ${model_type} 하이퍼파라미터 튜닝 시작 ==="
    echo "시작 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
    
    # 실행 전 메모리 상태 확인
    echo "실행 전 메모리 상태:"
    check_memory
    
    # 백그라운드에서 메모리 모니터링 시작
    (
        while true; do
            monitor_memory
            sleep 30  # 30초마다 체크
        done
    ) &
    local monitor_pid=$!
    
    # 모델 실행
    python scripts/run_hyperparameter_tuning.py \
        --model-type ${model_type} \
        --experiment-type hyperparameter_tuning \
        --n-trials 50 \
        --n-jobs ${N_JOBS} \
        --experiment-name "${experiment_name}_${TIMESTAMP}" \
        --save-model \
        --save-predictions \
        --verbose 1 \
        ${additional_params} \
        > "logs/${model_type}_${TIMESTAMP}.log" 2>&1
    
    local exit_code=$?
    
    # 메모리 모니터링 프로세스 종료
    kill $monitor_pid 2>/dev/null || true
    
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
echo "Phase 1: 기준선 설정 실험 시작"
echo "- 4개 모델 (XGBoost, CatBoost, LightGBM, Random Forest)"
echo "- 기본 설정으로 성능 측정"
echo "- n_trials: 50 (기준선 확립용)"
echo "- 강화된 메모리 안정화 설정 적용"
echo "- 실시간 메모리 모니터링 활성화"
echo "==================================================================="

# 실험 시작 시간 기록
start_time=$(date)
echo "실험 시작 시간: $start_time"

# 초기 메모리 상태 확인
echo "초기 메모리 상태:"
check_memory

    # 메모리 사용량 체크
    echo "현재 시스템 상태 체크 중..."
    python scripts/check_cpu_usage.py

echo ""
echo ">>> Phase 1-1: XGBoost 기준선 실험"
echo "-----------------------------------"
run_model "xgboost" "phase1_baseline_xgboost" ""

echo ""
echo ">>> Phase 1-2: CatBoost 기준선 실험"  
echo "-----------------------------------"
run_model "catboost" "phase1_baseline_catboost" ""

echo ""
echo ">>> Phase 1-3: LightGBM 기준선 실험"
echo "-----------------------------------"
run_model "lightgbm" "phase1_baseline_lightgbm" ""

echo ""
echo ">>> Phase 1-4: Random Forest 기준선 실험"
echo "------------------------------------------"
run_model "random_forest" "phase1_baseline_random_forest" ""

# 최종 메모리 정리
echo "최종 메모리 정리..."
cleanup_memory

# 실험 종료 시간 기록
end_time=$(date)
echo ""
echo "==================================================================="
echo "Phase 1 실험 완료!"
echo "실험 시작: $start_time"
echo "실험 종료: $end_time"
echo ""
echo "실행된 실험:"
echo "  - phase1_baseline_xgboost_${TIMESTAMP}"
echo "  - phase1_baseline_catboost_${TIMESTAMP}"
echo "  - phase1_baseline_lightgbm_${TIMESTAMP}"
echo "  - phase1_baseline_random_forest_${TIMESTAMP}"
echo ""
echo "다음 단계:"
echo "1. MLflow UI에서 결과 확인: mlflow ui --host 0.0.0.0 --port 5000"
echo "2. Phase 2 실행: ./phase2_input_range.sh"
echo "==================================================================="

# MLflow UI 자동 실행 (선택사항)
read -p "MLflow UI를 지금 실행하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "MLflow UI 실행 중... (http://localhost:5000)"
    mlflow ui --host 0.0.0.0 --port 5000 &
    echo "백그라운드에서 MLflow UI가 실행되었습니다."
fi

# 시그널 핸들러 제거
trap - INT TERM