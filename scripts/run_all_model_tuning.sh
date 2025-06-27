#!/bin/bash

# 전체 모델 튜닝 테스트 자동화 스크립트
# KBSMC 자살 예측 프로젝트

# set -e 제거: 오류 발생 시에도 계속 진행
# set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 파일 설정
LOG_DIR="results/test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/test_run_${TIMESTAMP}.log"

# 로깅 함수
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
}

# 테스트 실행 함수
run_test() {
    local test_name="$1"
    local command="$2"
    local timeout_minutes="${3:-30}"
    
    log "🚀 시작: $test_name"
    log "명령어: $command"
    
    # 타임아웃과 함께 명령어 실행
    if timeout "${timeout_minutes}m" bash -c "$command" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "완료: $test_name"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_error "타임아웃: $test_name (${timeout_minutes}분 초과)"
        else
            log_error "실패: $test_name (종료 코드: $exit_code)"
        fi
        return 1
    fi
}

# 결과 파일 확인 함수
check_results() {
    local test_name="$1"
    local results_dir="results"
    
    # 최신 experiment_results 파일 확인
    local latest_result=$(ls -t "$results_dir"/experiment_results_*.txt 2>/dev/null | head -1)
    
    if [ -n "$latest_result" ]; then
        log_success "결과 파일 생성됨: $latest_result"
        
        # 파일 크기 확인
        local file_size=$(stat -c%s "$latest_result")
        if [ $file_size -gt 1000 ]; then
            log_success "결과 파일 크기 적절: ${file_size} bytes"
        else
            log_warning "결과 파일이 너무 작음: ${file_size} bytes"
        fi
        
        # MLflow 링크 확인
        if grep -q "MLflow 링크:" "$latest_result"; then
            log_success "MLflow 링크 포함됨"
        else
            log_warning "MLflow 링크 누락"
        fi
        
        # 튜닝 범위 확인
        if grep -q "튜닝 범위:" "$latest_result"; then
            log_success "튜닝 범위 정보 포함됨"
        else
            log_warning "튜닝 범위 정보 누락"
        fi
        
        return 0
    else
        log_error "결과 파일을 찾을 수 없음"
        return 1
    fi
}

# 메인 테스트 실행
main() {
    log "🎯 KBSMC 자살 예측 프로젝트 - 전체 모델 튜닝 테스트 시작"
    log "로그 파일: $LOG_FILE"
    log "⚠️  오류 발생 시에도 다음 테스트로 계속 진행됩니다."
    
    local failed_tests=()
    local total_tests=0
    local passed_tests=0
    
    # Phase 1: 기본 검증 (빠른 테스트)
    log "📋 Phase 1: 기본 검증 (빠른 테스트)"
    
    # XGBoost 빠른 테스트
    ((total_tests++))
    if run_test "XGBoost 빠른 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type hyperparameter_tuning --nrows 1000 --n-trials 3 --cv-folds 2 --verbose 2" 10; then
        ((passed_tests++))
        check_results "XGBoost 빠른 테스트"
    else
        failed_tests+=("XGBoost 빠른 테스트")
    fi
    
    # CatBoost 빠른 테스트
    ((total_tests++))
    if run_test "CatBoost 빠른 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type catboost --experiment-type hyperparameter_tuning --nrows 1000 --n-trials 3 --cv-folds 2 --verbose 2" 10; then
        ((passed_tests++))
        check_results "CatBoost 빠른 테스트"
    else
        failed_tests+=("CatBoost 빠른 테스트")
    fi
    
    # LightGBM 빠른 테스트
    ((total_tests++))
    if run_test "LightGBM 빠른 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type lightgbm --experiment-type hyperparameter_tuning --nrows 1000 --n-trials 3 --cv-folds 2 --verbose 2" 10; then
        ((passed_tests++))
        check_results "LightGBM 빠른 테스트"
    else
        failed_tests+=("LightGBM 빠른 테스트")
    fi
    
    # Random Forest 빠른 테스트
    ((total_tests++))
    if run_test "Random Forest 빠른 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type random_forest --experiment-type hyperparameter_tuning --nrows 1000 --n-trials 3 --cv-folds 2 --verbose 2" 10; then
        ((passed_tests++))
        check_results "Random Forest 빠른 테스트"
    else
        failed_tests+=("Random Forest 빠른 테스트")
    fi
    
    # Phase 2: 분할 전략 테스트
    log "📋 Phase 2: 분할 전략 테스트"
    
    # Group K-Fold 테스트
    ((total_tests++))
    if run_test "Group K-Fold 분할 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type hyperparameter_tuning --split-strategy group_kfold --cv-folds 3 --nrows 2000 --n-trials 5 --verbose 2" 15; then
        ((passed_tests++))
        check_results "Group K-Fold 분할 테스트"
    else
        failed_tests+=("Group K-Fold 분할 테스트")
    fi
    
    # Time Series Walk Forward 테스트
    ((total_tests++))
    if run_test "Time Series Walk Forward 분할 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type catboost --experiment-type hyperparameter_tuning --split-strategy time_series_walk_forward --cv-folds 3 --nrows 2000 --n-trials 5 --verbose 2" 15; then
        ((passed_tests++))
        check_results "Time Series Walk Forward 분할 테스트"
    else
        failed_tests+=("Time Series Walk Forward 분할 테스트")
    fi
    
    # Time Series Group K-Fold 테스트
    ((total_tests++))
    if run_test "Time Series Group K-Fold 분할 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type lightgbm --experiment-type hyperparameter_tuning --split-strategy time_series_group_kfold --cv-folds 3 --nrows 2000 --n-trials 5 --verbose 2" 15; then
        ((passed_tests++))
        check_results "Time Series Group K-Fold 분할 테스트"
    else
        failed_tests+=("Time Series Group K-Fold 분할 테스트")
    fi
    
    # Phase 3: 리샘플링 테스트
    log "📋 Phase 3: 리샘플링 테스트"
    
    # XGBoost 리샘플링 비교
    ((total_tests++))
    if run_test "XGBoost 리샘플링 비교 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type resampling --resampling-comparison --nrows 2000 --n-trials 5 --cv-folds 2 --verbose 2" 20; then
        ((passed_tests++))
        check_results "XGBoost 리샘플링 비교 테스트"
    else
        failed_tests+=("XGBoost 리샘플링 비교 테스트")
    fi
    
    # CatBoost 리샘플링 비교
    ((total_tests++))
    if run_test "CatBoost 리샘플링 비교 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type catboost --experiment-type resampling --resampling-comparison --nrows 2000 --n-trials 5 --cv-folds 2 --verbose 2" 20; then
        ((passed_tests++))
        check_results "CatBoost 리샘플링 비교 테스트"
    else
        failed_tests+=("CatBoost 리샘플링 비교 테스트")
    fi
    
    # Phase 4: 평가 지표별 테스트
    log "📋 Phase 4: 평가 지표별 테스트"
    
    # PR-AUC 최적화 테스트
    ((total_tests++))
    if run_test "PR-AUC 최적화 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type hyperparameter_tuning --primary-metric pr_auc --nrows 2000 --n-trials 5 --cv-folds 3 --verbose 2" 15; then
        ((passed_tests++))
        check_results "PR-AUC 최적화 테스트"
    else
        failed_tests+=("PR-AUC 최적화 테스트")
    fi
    
    # F1-Score 최적화 테스트
    ((total_tests++))
    if run_test "F1-Score 최적화 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type lightgbm --experiment-type hyperparameter_tuning --primary-metric f1 --nrows 2000 --n-trials 5 --cv-folds 3 --verbose 2" 15; then
        ((passed_tests++))
        check_results "F1-Score 최적화 테스트"
    else
        failed_tests+=("F1-Score 최적화 테스트")
    fi
    
    # ROC-AUC 최적화 테스트
    ((total_tests++))
    if run_test "ROC-AUC 최적화 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type catboost --experiment-type hyperparameter_tuning --primary-metric roc_auc --nrows 2000 --n-trials 5 --cv-folds 3 --verbose 2" 15; then
        ((passed_tests++))
        check_results "ROC-AUC 최적화 테스트"
    else
        failed_tests+=("ROC-AUC 최적화 테스트")
    fi
    
    # Phase 5: 고급 기능 테스트
    log "📋 Phase 5: 고급 기능 테스트"
    
    # Early Stopping 테스트
    ((total_tests++))
    if run_test "Early Stopping 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type hyperparameter_tuning --early-stopping --early-stopping-rounds 10 --nrows 2000 --n-trials 5 --cv-folds 3 --verbose 2" 15; then
        ((passed_tests++))
        check_results "Early Stopping 테스트"
    else
        failed_tests+=("Early Stopping 테스트")
    fi
    
    # 피처 선택 테스트
    ((total_tests++))
    if run_test "피처 선택 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type random_forest --experiment-type hyperparameter_tuning --feature-selection --feature-selection-method mutual_info --feature-selection-k 5 --nrows 2000 --n-trials 5 --cv-folds 3 --verbose 2" 15; then
        ((passed_tests++))
        check_results "피처 선택 테스트"
    else
        failed_tests+=("피처 선택 테스트")
    fi
    
    # 타임아웃 설정 테스트
    ((total_tests++))
    if run_test "타임아웃 설정 테스트" \
        "python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type hyperparameter_tuning --timeout 120 --nrows 2000 --n-trials 5 --cv-folds 2 --verbose 2" 10; then
        ((passed_tests++))
        check_results "타임아웃 설정 테스트"
    else
        failed_tests+=("타임아웃 설정 테스트")
    fi
    
    # 최종 결과 요약
    log "🎯 테스트 완료 요약"
    log "총 테스트 수: $total_tests"
    log "성공한 테스트: $passed_tests"
    log "실패한 테스트: $(($total_tests - $passed_tests))"
    
    if [ ${#failed_tests[@]} -eq 0 ]; then
        log_success "🎉 모든 테스트가 성공적으로 완료되었습니다!"
        exit 0
    else
        log_error "❌ 실패한 테스트가 있습니다:"
        for test in "${failed_tests[@]}"; do
            log_error "  - $test"
        done
        log_warning "⚠️  일부 테스트가 실패했지만 전체 테스트는 완료되었습니다."
        exit 1
    fi
}

# 스크립트 실행
main "$@"