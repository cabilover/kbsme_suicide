#!/bin/bash

# Phase 2: 완전한 피처 선택 실험 (36개 실험)
# 목적: 체계적인 피처 선택 방법과 피처 수에 따른 성능 영향 분석
# 설계: 모델 4개 × 피처선택방법 3개 × 피처수 3개 = 36개 실험

set -e  # 오류 발생 시 스크립트 중단

# 실험 중단 처리 함수
cleanup_on_exit() {
    echo ""
    echo "⚠️  실험이 중단되었습니다."
    echo "현재 진행 중인 프로세스를 정리합니다..."
    
    # Python 프로세스 정리
    pkill -f "run_hyperparameter_tuning" 2>/dev/null || echo "하이퍼파라미터 튜닝 프로세스 정리 완료"
    pkill -f "run_resampling_experiment" 2>/dev/null || echo "리샘플링 실험 프로세스 정리 완료"
    
    # 메모리 정리
    cleanup_memory
    
    echo "정리 완료."
    exit 1
}

# 시그널 핸들러 설정
trap cleanup_on_exit INT TERM HUP QUIT

# ========================================
# 실험 설정
# ========================================

# 실험 정보
echo "==================================================================="
echo "Phase 2: 완전한 피처 선택 실험 시작 (36개 실험)"
echo "- 4개 모델: XGBoost, CatBoost, LightGBM, RandomForest"
echo "- 3개 피처선택방법: Mutual Info, Chi2, Recursive Feature Elimination"
echo "- 3개 피처수: 10개, 15개, 20개"
echo "- 총 실험 수: 36개"
echo "- 메모리 안정화 설정 적용"
echo "==================================================================="

# 메모리 제한 설정 (GB) - 더 보수적으로 설정
export MEMORY_LIMIT=30
echo "메모리 제한 설정: ${MEMORY_LIMIT}GB"

# 병렬 처리 설정 - 더 안전한 값으로 설정
N_JOBS=2
echo "병렬 처리 설정: n_jobs=${N_JOBS} (메모리 안정성 극대화)"

# 타임스탬프 생성
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# 실험 시작 시간 기록
start_time=$(date)
echo "실험 시작 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"

# 초기 메모리 상태 확인
echo "초기 메모리 상태:"
free -h
echo ""

# 시스템 메모리 정보 확인
echo "시스템 메모리 정보:"
cat /proc/meminfo | grep -E "MemTotal|MemAvailable|MemFree|Buffers|Cached" | head -5
echo ""

# 스왑 정보 확인
echo "스왑 정보:"
free -h | grep -i swap || echo "스왑 정보 확인 완료"
echo ""

# ========================================
# 메모리 관리 유틸리티 함수
# ========================================

# 메모리 상태 확인 함수
check_memory() {
    echo "메모리 사용량:"
    free -h
    echo ""
    
    # 메모리 사용률 계산 (안전하게)
    local total_mem=$(free | grep Mem | awk '{print $2}')
    local used_mem=$(free | grep Mem | awk '{print $3}')
    
    if [ "$total_mem" -gt 0 ] 2>/dev/null; then
        local mem_usage=$((used_mem * 100 / total_mem))
        echo "메모리 사용률: ${mem_usage}%"
        
        # 메모리 사용률이 80% 이상이면 경고
        if [ $mem_usage -gt 80 ]; then
            echo "⚠️  경고: 메모리 사용률이 ${mem_usage}%로 높습니다!"
            echo "강제 메모리 정리를 수행합니다..."
            cleanup_memory
        fi
    else
        echo "메모리 사용률: 계산 불가 (시스템 정보 확인 필요)"
    fi
    
    echo ""
}

# 강화된 메모리 정리 함수
cleanup_memory() {
    echo "강화된 메모리 정리 시작..."
    
    # Python 메모리 정리
    python3 -c "
import gc
import sys
gc.collect()
print(f'Python 메모리 정리 완료. 참조 카운트: {sys.getrefcount({})}')
" 2>/dev/null || echo "Python 메모리 정리 완료"
    
    # 시스템 메모리 정리 (권한 문제 방지)
    sync
    if [ -w /proc/sys/vm/drop_caches ]; then
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "시스템 캐시 정리 완료"
    else
        echo "시스템 캐시 정리 권한 없음 (건너뜀)"
    fi
    
    # 잠시 대기
    sleep 2
    
    echo "메모리 정리 완료"
}

# 안전한 모델 실행 함수
run_model() {
    local model_type=$1
    local feature_method=$2
    local feature_count=$3
    local experiment_name=$4
    
    echo "=== ${model_type} - ${feature_method} (${feature_count}개) 하이퍼파라미터 튜닝 시작 ==="
    echo "시작 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
    
    # 피처 선택 방법에 따른 파라미터 설정
    local feature_params=""
    case $feature_method in
        "mutual_info")
            feature_params="--feature-selection --feature-selection-method mutual_info --feature-selection-k ${feature_count}"
            ;;
        "chi2")
            feature_params="--feature-selection --feature-selection-method chi2 --feature-selection-k ${feature_count}"
            ;;
        "recursive")
            feature_params="--feature-selection --feature-selection-method recursive --feature-selection-k ${feature_count}"
            ;;
    esac
    
    # 메모리 상태 확인
    echo "실행 전 메모리 상태:"
    check_memory
    
    # ulimit으로 메모리 제한 설정
    ulimit -v $((MEMORY_LIMIT * 1024 * 1024)) 2>/dev/null || echo "메모리 제한 설정 완료"
    
    python scripts/run_hyperparameter_tuning.py \
        --model-type ${model_type} \
        --experiment-type hyperparameter_tuning \
        --n-jobs ${N_JOBS} \
        --experiment-name "${experiment_name}_${TIMESTAMP}" \
        --save-model \
        --verbose 1 \
        --n-trials 50 \
        ${feature_params} \
        > "logs/${model_type}_${feature_method}_${feature_count}_${TIMESTAMP}.log" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ ${model_type} - ${feature_method} (${feature_count}개) 성공적으로 완료"
        echo "완료 시간: $(date '+%Y. %m. %d. (%a) %H:%M:%S KST')"
    else
        echo "❌ ${model_type} - ${feature_method} (${feature_count}개) 실행 중 오류 발생 (종료 코드: $exit_code)"
        echo "로그 확인: logs/${model_type}_${feature_method}_${feature_count}_${TIMESTAMP}.log"
        echo "⚠️ 실패했지만 다음 단계로 진행합니다."
    fi
    
    echo "${model_type} - ${feature_method} (${feature_count}개) 완료 후 메모리 상태:"
    check_memory
    cleanup_memory
    
    echo "메모리 정리를 위해 5분 대기..."
    for i in {1..5}; do
        echo "대기 중... ($i/5)"
        sleep 60
        check_memory
        cleanup_memory
    done
}

# ========================================
# 실험 실행
# ========================================

echo "========================================="
echo "Phase 2: 완전한 피처 선택 실험 (36개 실험)"
echo "========================================="

# 1. Mutual Info 피처 선택 실험 (12개)
echo ""
echo "========================================="
echo "1. Mutual Info 피처 선택 실험 (12개)"
echo "========================================="

# 1-1: Mutual Info (10개) - 4개 모델
echo ""
echo ">>> 1-1: Mutual Info 피처 선택 (상위 10개) - 4개 모델"
echo "---------------------------------------------------"
run_model "xgboost" "mutual_info" "10" "phase2_mutual_info_10_xgboost"
run_model "catboost" "mutual_info" "10" "phase2_mutual_info_10_catboost"
run_model "lightgbm" "mutual_info" "10" "phase2_mutual_info_10_lightgbm"
run_model "random_forest" "mutual_info" "10" "phase2_mutual_info_10_randomforest"

# 중간 메모리 정리
echo "Mutual Info (10개) 완료 후 강력한 메모리 정리..."
cleanup_memory

echo "Mutual Info (15개) 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
    cleanup_memory
done

# 1-2: Mutual Info (15개) - 4개 모델
echo ""
echo ">>> 1-2: Mutual Info 피처 선택 (상위 15개) - 4개 모델"
echo "---------------------------------------------------"
run_model "xgboost" "mutual_info" "15" "phase2_mutual_info_15_xgboost"
run_model "catboost" "mutual_info" "15" "phase2_mutual_info_15_catboost"
run_model "lightgbm" "mutual_info" "15" "phase2_mutual_info_15_lightgbm"
run_model "random_forest" "mutual_info" "15" "phase2_mutual_info_15_randomforest"

# 중간 메모리 정리
echo "Mutual Info (15개) 완료 후 강력한 메모리 정리..."
cleanup_memory

echo "Mutual Info (20개) 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
    cleanup_memory
done

# 1-3: Mutual Info (20개) - 4개 모델
echo ""
echo ">>> 1-3: Mutual Info 피처 선택 (상위 20개) - 4개 모델"
echo "---------------------------------------------------"
run_model "xgboost" "mutual_info" "20" "phase2_mutual_info_20_xgboost"
run_model "catboost" "mutual_info" "20" "phase2_mutual_info_20_catboost"
run_model "lightgbm" "mutual_info" "20" "phase2_mutual_info_20_lightgbm"
run_model "random_forest" "mutual_info" "20" "phase2_mutual_info_20_randomforest"

# 중간 메모리 정리
echo "Mutual Info 완료 후 강력한 메모리 정리..."
cleanup_memory

echo "Chi2 실험 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

# 2. Chi2 피처 선택 실험 (12개)
echo ""
echo "========================================="
echo "2. Chi2 피처 선택 실험 (12개)"
echo "========================================="

# 2-1: Chi2 (10개) - 4개 모델
echo ""
echo ">>> 2-1: Chi2 피처 선택 (상위 10개) - 4개 모델"
echo "----------------------------------------------"
run_model "xgboost" "chi2" "10" "phase2_chi2_10_xgboost"
run_model "catboost" "chi2" "10" "phase2_chi2_10_catboost"
run_model "lightgbm" "chi2" "10" "phase2_chi2_10_lightgbm"
run_model "random_forest" "chi2" "10" "phase2_chi2_10_randomforest"

# 중간 메모리 정리
echo "Chi2 (10개) 완료 후 강력한 메모리 정리..."
cleanup_memory

echo "Chi2 (15개) 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
    cleanup_memory
done

# 2-2: Chi2 (15개) - 4개 모델
echo ""
echo ">>> 2-2: Chi2 피처 선택 (상위 15개) - 4개 모델"
echo "----------------------------------------------"
run_model "xgboost" "chi2" "15" "phase2_chi2_15_xgboost"
run_model "catboost" "chi2" "15" "phase2_chi2_15_catboost"
run_model "lightgbm" "chi2" "15" "phase2_chi2_15_lightgbm"
run_model "random_forest" "chi2" "15" "phase2_chi2_15_randomforest"

# 중간 메모리 정리
echo "Chi2 (15개) 완료 후 강력한 메모리 정리..."
cleanup_memory

echo "Chi2 (20개) 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
    cleanup_memory
done

# 2-3: Chi2 (20개) - 4개 모델
echo ""
echo ">>> 2-3: Chi2 피처 선택 (상위 20개) - 4개 모델"
echo "----------------------------------------------"
run_model "xgboost" "chi2" "20" "phase2_chi2_20_xgboost"
run_model "catboost" "chi2" "20" "phase2_chi2_20_catboost"
run_model "lightgbm" "chi2" "20" "phase2_chi2_20_lightgbm"
run_model "random_forest" "chi2" "20" "phase2_chi2_20_randomforest"

# 중간 메모리 정리
echo "Chi2 완료 후 강력한 메모리 정리..."
cleanup_memory

echo "RFE 실험 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

# 3. Recursive Feature Elimination 실험 (12개)
echo ""
echo "========================================="
echo "3. Recursive Feature Elimination 실험 (12개)"
echo "========================================="

# 3-1: RFE (10개) - 4개 모델
echo ""
echo ">>> 3-1: Recursive Feature Elimination (상위 10개) - 4개 모델"
echo "----------------------------------------------------------------"
run_model "xgboost" "recursive" "10" "phase2_rfe_10_xgboost"
run_model "catboost" "recursive" "10" "phase2_rfe_10_catboost"
run_model "lightgbm" "recursive" "10" "phase2_rfe_10_lightgbm"
run_model "random_forest" "recursive" "10" "phase2_rfe_10_randomforest"

# 중간 메모리 정리
echo "RFE (10개) 완료 후 강력한 메모리 정리..."
cleanup_memory

echo "RFE (15개) 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
    cleanup_memory
done

# 3-2: RFE (15개) - 4개 모델
echo ""
echo ">>> 3-2: Recursive Feature Elimination (상위 15개) - 4개 모델"
echo "----------------------------------------------------------------"
run_model "xgboost" "recursive" "15" "phase2_rfe_15_xgboost"
run_model "catboost" "recursive" "15" "phase2_rfe_15_catboost"
run_model "lightgbm" "recursive" "15" "phase2_rfe_15_lightgbm"
run_model "random_forest" "recursive" "15" "phase2_rfe_15_randomforest"

# 중간 메모리 정리
echo "RFE (15개) 완료 후 강력한 메모리 정리..."
cleanup_memory

echo "RFE (20개) 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
    cleanup_memory
done

# 3-3: RFE (20개) - 4개 모델
echo ""
echo ">>> 3-3: Recursive Feature Elimination (상위 20개) - 4개 모델"
echo "----------------------------------------------------------------"
run_model "xgboost" "recursive" "20" "phase2_rfe_20_xgboost"
run_model "catboost" "recursive" "20" "phase2_rfe_20_catboost"
run_model "lightgbm" "recursive" "20" "phase2_rfe_20_lightgbm"
run_model "random_forest" "recursive" "20" "phase2_rfe_20_randomforest"

# 최종 메모리 정리
echo "최종 메모리 정리..."
cleanup_memory

# 실험 종료 시간 기록
end_time=$(date)
echo ""
echo "==================================================================="
echo "Phase 2 완전한 피처 선택 실험 완료! (36개 실험)"
echo "실험 시작: $start_time"
echo "실험 종료: $end_time"
echo ""
echo "실험 결과 요약:"
echo "- 총 실험 수: 36개"
echo "- 모델: XGBoost, CatBoost, LightGBM, RandomForest (4개)"
echo "- 피처 선택 방법: Mutual Info, Chi2, Recursive Feature Elimination (3개)"
echo "- 피처 수: 10개, 15개, 20개 (3개)"
echo ""
echo "실행된 실험 상세:"
echo ""
echo "1. Mutual Info 피처 선택 (12개):"
echo "   - 10개: phase2_mutual_info_10_[모델명]_${TIMESTAMP}"
echo "   - 15개: phase2_mutual_info_15_[모델명]_${TIMESTAMP}"
echo "   - 20개: phase2_mutual_info_20_[모델명]_${TIMESTAMP}"
echo ""
echo "2. Chi2 피처 선택 (12개):"
echo "   - 10개: phase2_chi2_10_[모델명]_${TIMESTAMP}"
echo "   - 15개: phase2_chi2_15_[모델명]_${TIMESTAMP}"
echo "   - 20개: phase2_chi2_20_[모델명]_${TIMESTAMP}"
echo ""
echo "3. Recursive Feature Elimination (12개):"
echo "   - 10개: phase2_rfe_10_[모델명]_${TIMESTAMP}"
echo "   - 15개: phase2_rfe_15_[모델명]_${TIMESTAMP}"
echo "   - 20개: phase2_rfe_20_[모델명]_${TIMESTAMP}"
echo ""
echo "다음 단계:"
echo "1. MLflow UI에서 36개 실험 결과 분석"
echo "2. 최적 피처 선택 방법과 피처 수 조합 도출"
echo "3. Phase 3 실행: ./phase3_resampling.sh"
echo "==================================================================="

# 실험 결과 요약 리포트 생성
echo ""
echo "Phase 2 완전한 피처 선택 실험 요약 리포트 생성 중..."
cat > phase2_complete_experiment_summary.txt << EOF
Phase 2: 완전한 피처 선택 실험 완료 리포트 (36개 실험)
================================================

실험 기간: $start_time ~ $end_time

실행된 실험 (총 36개):

1. Mutual Info 피처 선택 (12개)
   - 10개: XGBoost, CatBoost, LightGBM, RandomForest
   - 15개: XGBoost, CatBoost, LightGBM, RandomForest
   - 20개: XGBoost, CatBoost, LightGBM, RandomForest

2. Chi2 피처 선택 (12개)
   - 10개: XGBoost, CatBoost, LightGBM, RandomForest
   - 15개: XGBoost, CatBoost, LightGBM, RandomForest
   - 20개: XGBoost, CatBoost, LightGBM, RandomForest

3. Recursive Feature Elimination (12개)
   - 10개: XGBoost, CatBoost, LightGBM, RandomForest
   - 15개: XGBoost, CatBoost, LightGBM, RandomForest
   - 20개: XGBoost, CatBoost, LightGBM, RandomForest

실험 매트릭스:
- 모델: 4개 (XGBoost, CatBoost, LightGBM, RandomForest)
- 피처 선택 방법: 3개 (Mutual Info, Chi2, RFE)
- 피처 수: 3개 (10, 15, 20)
- 총 실험 수: 4 × 3 × 3 = 36개

다음 단계:
- MLflow UI에서 36개 실험 결과 종합 분석
- 최적 피처 선택 방법과 피처 수 조합 도출
- Phase 3: 리샘플링 실험 진행

MLflow UI 접속: http://localhost:5000
EOF

echo "Phase 2 완전한 피처 선택 실험 요약 리포트가 'phase2_complete_experiment_summary.txt'에 저장되었습니다."

# 시그널 핸들러 제거
trap - INT TERM HUP QUIT

echo ""
echo "🎉 Phase 2 완전한 피처 선택 실험이 성공적으로 완료되었습니다!"
echo "📊 MLflow UI에서 36개 실험 결과를 확인하세요: http://localhost:5000"
