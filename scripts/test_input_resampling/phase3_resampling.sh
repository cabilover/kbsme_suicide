#!/bin/bash

# Phase 3: Resampling 방법 체계적 비교
# 목적: 극도 불균형 데이터(849:1)에 최적화된 리샘플링 기법 발견

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
echo "Phase 3: Resampling 방법 체계적 비교 실험 시작"
echo "- 4개 모델 × 7개 리샘플링 기법 비교"
echo "- 리샘플링 기법: none, smote, borderline_smote, adasyn,"
echo "  under_sampling, hybrid, time_series_adapted"
echo "- 극도 불균형 데이터 (자살 시도: 0.12%, 849:1 비율) 대응"
echo "- 메모리 안정화 설정 적용"
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
echo "========================================="
echo "Phase 3-1: 전체 리샘플링 기법 비교"
echo "4개 모델 × 7개 리샘플링 기법"
echo "========================================="

echo ""
echo ">>> 3-1-1: XGBoost 리샘플링 비교"
echo "--------------------------------"
run_resampling_model "xgboost" "phase3_resampling_comparison_xgboost" "--resampling-comparison --resampling-methods none smote borderline_smote adasyn under_sampling hybrid time_series_adapted --n-trials 60"

echo ""
echo ">>> 3-1-2: CatBoost 리샘플링 비교"
echo "---------------------------------"
run_resampling_model "catboost" "phase3_resampling_comparison_catboost" "--resampling-comparison --resampling-methods none smote borderline_smote adasyn under_sampling hybrid time_series_adapted --n-trials 60"

echo ""
echo ">>> 3-1-3: LightGBM 리샘플링 비교"
echo "---------------------------------"
run_resampling_model "lightgbm" "phase3_resampling_comparison_lightgbm" "--resampling-comparison --resampling-methods none smote borderline_smote adasyn under_sampling hybrid time_series_adapted --n-trials 60"

echo ""
echo ">>> 3-1-4: Random Forest 리샘플링 비교"
echo "--------------------------------------"
run_resampling_model "random_forest" "phase3_resampling_comparison_random_forest" "--resampling-comparison --resampling-methods none smote borderline_smote adasyn under_sampling hybrid time_series_adapted --n-trials 60"

# Phase 간 강력한 메모리 정리
echo "Phase 3-1과 Phase 3-2 사이 강력한 메모리 정리..."
cleanup_memory

echo "Phase 3-2 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

echo ""
echo "========================================="
echo "Phase 3-2: 시계열 특화 리샘플링 실험"
echo "========================================="

echo ""
echo ">>> 3-2-1: 시계열 적응형 리샘플링 - XGBoost"
echo "------------------------------------------"
run_resampling_model "xgboost" "phase3_time_series_resampling_xgboost" "--resampling-method time_series_adapted --n-trials 50"

echo ""
echo ">>> 3-2-2: 시계열 적응형 리샘플링 - CatBoost"
echo "--------------------------------------------"
run_resampling_model "catboost" "phase3_time_series_resampling_catboost" "--resampling-method time_series_adapted --n-trials 50"

# Phase 간 강력한 메모리 정리
echo "Phase 3-2와 Phase 3-3 사이 강력한 메모리 정리..."
cleanup_memory

echo "Phase 3-3 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

echo ""
echo "========================================="
echo "Phase 3-3: 특정 리샘플링 기법 집중 분석"
echo "========================================="

echo ""
echo ">>> 3-3-1: SMOTE 하이퍼파라미터 튜닝"
echo "------------------------------------"
# SMOTE k_neighbors 및 sampling_strategy 최적화
run_resampling_model "xgboost" "phase3_smote_tuning_xgboost" "--resampling-method smote --n-trials 50"

echo ""
echo ">>> 3-3-2: Borderline SMOTE 하이퍼파라미터 튜닝"
echo "-----------------------------------------------"
run_resampling_model "catboost" "phase3_borderline_smote_tuning_catboost" "--resampling-method borderline_smote --n-trials 50"

echo ""
echo ">>> 3-3-3: ADASYN 하이퍼파라미터 튜닝"
echo "-------------------------------------"
run_resampling_model "lightgbm" "phase3_adasyn_tuning_lightgbm" "--resampling-method adasyn --n-trials 50"

# Phase 간 강력한 메모리 정리
echo "Phase 3-3과 Phase 3-4 사이 강력한 메모리 정리..."
cleanup_memory

echo "Phase 3-4 시작 전 5분 대기..."
for i in {1..5}; do
    echo "대기 중... ($i/5)"
    sleep 60
    check_memory
done

echo ""
echo "========================================="
echo "Phase 3-4: 하이브리드 접근 방법 실험"
echo "========================================="

echo ""
echo ">>> 3-4-1: SMOTE + Under Sampling 조합"
echo "--------------------------------------"
run_resampling_model "xgboost" "phase3_hybrid_resampling_xgboost" "--resampling-method hybrid --n-trials 50"

echo ""
echo ">>> 3-4-2: 리샘플링 + 피처선택 조합"
echo "-----------------------------------"
run_resampling_model "catboost" "phase3_smote_feature_selection_catboost" "--resampling-method smote --feature-selection --feature-selection-method mutual_info --feature-selection-k 12 --n-trials 50"

# 최종 메모리 정리
echo "최종 메모리 정리..."
cleanup_memory

# 실험 종료 시간 기록
end_time=$(date)
echo ""
echo "==================================================================="
echo "Phase 3 실험 완료!"
echo "실험 시작: $start_time"
echo "실험 종료: $end_time"
echo ""
echo "실험 결과 요약:"
echo "- 전체 리샘플링 비교: 4개 모델 × 7개 기법 = 28개 실험"
echo "- 시계열 특화 리샘플링: 2개 실험"
echo "- 특정 기법 집중 분석: 3개 실험"
echo "- 하이브리드 접근: 2개 실험"
echo "- 총 35개 리샘플링 실험 완료"
echo ""
echo "실행된 실험:"
echo "전체 리샘플링 비교:"
echo "  - phase3_resampling_comparison_xgboost_${TIMESTAMP}"
echo "  - phase3_resampling_comparison_catboost_${TIMESTAMP}"
echo "  - phase3_resampling_comparison_lightgbm_${TIMESTAMP}"
echo "  - phase3_resampling_comparison_random_forest_${TIMESTAMP}"
echo "시계열 특화:"
echo "  - phase3_time_series_resampling_xgboost_${TIMESTAMP}"
echo "  - phase3_time_series_resampling_catboost_${TIMESTAMP}"
echo "특정 기법 분석:"
echo "  - phase3_smote_tuning_xgboost_${TIMESTAMP}"
echo "  - phase3_borderline_smote_tuning_catboost_${TIMESTAMP}"
echo "  - phase3_adasyn_tuning_lightgbm_${TIMESTAMP}"
echo "하이브리드 접근:"
echo "  - phase3_hybrid_resampling_xgboost_${TIMESTAMP}"
echo "  - phase3_smote_feature_selection_catboost_${TIMESTAMP}"
echo ""
echo "다음 단계:"
echo "1. MLflow UI에서 리샘플링 기법별 성능 비교"
echo "2. 각 모델별 최적 리샘플링 기법 도출"
echo "3. Phase 4 실행: ./phase4_deep_analysis.sh"
echo "==================================================================="

# 리샘플링 실험 요약 리포트 생성
echo ""
echo "Phase 3 리샘플링 실험 요약 리포트 생성 중..."
cat > phase3_resampling_summary.txt << EOF
Phase 3: Resampling 방법 체계적 비교 완료 리포트
=============================================

실험 기간: $start_time ~ $end_time

극도 불균형 데이터 특성:
- 자살 시도 비율: 0.12% (849:1)
- 전체 데이터: 1,569,071 행

실행된 실험 카테고리:
1. 전체 리샘플링 비교 (28개 실험)
   - XGBoost: phase3_resampling_comparison_xgboost_${TIMESTAMP}
   - CatBoost: phase3_resampling_comparison_catboost_${TIMESTAMP}
   - LightGBM: phase3_resampling_comparison_lightgbm_${TIMESTAMP}
   - Random Forest: phase3_resampling_comparison_random_forest_${TIMESTAMP}

2. 시계열 특화 리샘플링 (2개 실험)
   - XGBoost: phase3_time_series_resampling_xgboost_${TIMESTAMP}
   - CatBoost: phase3_time_series_resampling_catboost_${TIMESTAMP}

3. 특정 기법 집중 분석 (3개 실험)
   - SMOTE: phase3_smote_tuning_xgboost_${TIMESTAMP}
   - Borderline SMOTE: phase3_borderline_smote_tuning_catboost_${TIMESTAMP}
   - ADASYN: phase3_adasyn_tuning_lightgbm_${TIMESTAMP}

4. 하이브리드 접근 (2개 실험)
   - Hybrid: phase3_hybrid_resampling_xgboost_${TIMESTAMP}
   - SMOTE + Feature Selection: phase3_smote_feature_selection_catboost_${TIMESTAMP}

총 35개 리샘플링 실험 완료

분석 포인트:
- 각 모델별 최적 리샘플링 기법 확인
- 극도 불균형에서의 Precision/Recall 트레이드오프
- F1-score, PR-AUC, MCC 등 불균형 특화 지표 비교
- 시계열 데이터에서의 리샘플링 효과

MLflow UI 접속: http://localhost:5000
EOF

echo "Phase 3 요약 리포트가 'phase3_resampling_summary.txt'에 저장되었습니다."

# 중간 결과 확인 옵션
read -p "중간 결과를 MLflow UI에서 확인하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "MLflow UI 실행 중... (http://localhost:5000)"
    mlflow ui --host 0.0.0.0 --port 5000 &
    echo "백그라운드에서 MLflow UI가 실행되었습니다."
    echo "Phase 3 결과를 확인한 후 Phase 4를 진행하세요."
fi

# 시그널 핸들러 제거
trap - INT TERM