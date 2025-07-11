#!/bin/bash

# 메모리 최적화된 개별 모델 실행 스크립트
# 각 모델을 독립적으로 실행하여 메모리 문제 해결

set -e

# Argument 파싱
N_TRIALS=${1:-3}  # 기본값: 3

# 타임스탬프 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "=== 개별 모델 실행 시작 ==="
echo "시작 시간: $(date)"
echo "타임스탬프: ${TIMESTAMP}"
echo "n_trials: ${N_TRIALS}"
echo "전체 데이터셋 사용"

# 메모리 제한 설정 (GB)
MEMORY_LIMIT=50
echo "메모리 제한 설정: ${MEMORY_LIMIT}GB"
echo "모든 모델 n_jobs 통일: 4 (메모리 안정성 확보)"

# 메모리 상태 확인 함수
check_memory() {
    echo "메모리 사용량:"
    free -h
}

# 강화된 메모리 정리 함수
cleanup_memory() {
    echo "강화된 메모리 정리 시작..."
    
    # Python 가비지 컬렉션
    python3 -c "
import gc
import sys
gc.collect()
print(f'Python 메모리 정리 완료. 참조 카운트: {sys.getrefcount({})}')
"
    
    # 시스템 캐시 정리 (권한 오류 무시)
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "시스템 캐시 정리 권한 없음 (무시됨)"
    
    echo "메모리 정리 완료"
}

# 초기 메모리 상태 확인
echo "초기 메모리 상태:"
check_memory

# Phase 2: 기본 모델들 (n_jobs=4로 감소)
echo ""
echo "========================================"
echo "Phase 2: 기본 모델 실행 시작"
echo "========================================"

# CatBoost (기본)
echo ""
echo "=== Phase 2 Step 1: catboost (기본) 하이퍼파라미터 튜닝 시작 ==="
echo "시작 시간: $(date)"
python scripts/run_hyperparameter_tuning.py \
    --model-type catboost \
    --experiment-type hyperparameter_tuning \
    --experiment-name hyperparameter_tuning \
    --n-trials ${N_TRIALS} \
    --n-jobs 4 \
    > logs/catboost_basic_${TIMESTAMP}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ catboost 성공적으로 완료"
else
    echo "❌ catboost 실패 (종료 코드: $?)"
    echo "⚠️ 실패했지만 다음 단계로 진행합니다."
fi

echo "완료 시간: $(date)"
echo "catboost 완료 후 메모리 상태:"
check_memory
cleanup_memory

# 3분 대기
echo "메모리 정리를 위해 3분 대기..."
for i in {1..3}; do
    echo "대기 중... ($i/3)"
    sleep 60
    check_memory
done

# Random Forest (기본)
echo ""
echo "=== Phase 2 Step 2: random_forest (기본) 하이퍼파라미터 튜닝 시작 ==="
echo "시작 시간: $(date)"
python scripts/run_hyperparameter_tuning.py \
    --model-type random_forest \
    --experiment-type hyperparameter_tuning \
    --experiment-name hyperparameter_tuning \
    --n-trials ${N_TRIALS} \
    --n-jobs 4 \
    > logs/random_forest_basic_${TIMESTAMP}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ random_forest 성공적으로 완료"
else
    echo "❌ random_forest 실패 (종료 코드: $?)"
    echo "⚠️ 실패했지만 다음 단계로 진행합니다."
fi

echo "완료 시간: $(date)"
echo "random_forest 완료 후 메모리 상태:"
check_memory
cleanup_memory

# 3분 대기
echo "메모리 정리를 위해 3분 대기..."
for i in {1..3}; do
    echo "대기 중... ($i/3)"
    sleep 60
    check_memory
done

# XGBoost (기본)
echo ""
echo "=== Phase 2 Step 3: xgboost (기본) 하이퍼파라미터 튜닝 시작 ==="
echo "시작 시간: $(date)"
python scripts/run_hyperparameter_tuning.py \
    --model-type xgboost \
    --experiment-type hyperparameter_tuning \
    --experiment-name hyperparameter_tuning \
    --n-trials ${N_TRIALS} \
    --n-jobs 4 \
    > logs/xgboost_basic_${TIMESTAMP}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ xgboost 성공적으로 완료"
else
    echo "❌ xgboost 실패 (종료 코드: $?)"
    echo "⚠️ 실패했지만 다음 단계로 진행합니다."
fi

echo "완료 시간: $(date)"
echo "xgboost 완료 후 메모리 상태:"
check_memory
cleanup_memory

# 3분 대기
echo "메모리 정리를 위해 3분 대기..."
for i in {1..3}; do
    echo "대기 중... ($i/3)"
    sleep 60
    check_memory
done

# LightGBM (기본)
echo ""
echo "=== Phase 2 Step 4: lightgbm (기본) 하이퍼파라미터 튜닝 시작 ==="
echo "시작 시간: $(date)"
python scripts/run_hyperparameter_tuning.py \
    --model-type lightgbm \
    --experiment-type hyperparameter_tuning \
    --experiment-name hyperparameter_tuning \
    --n-trials ${N_TRIALS} \
    --n-jobs 4 \
    > logs/lightgbm_basic_${TIMESTAMP}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ lightgbm 성공적으로 완료"
else
    echo "❌ lightgbm 실패 (종료 코드: $?)"
    echo "⚠️ 실패했지만 다음 단계로 진행합니다."
fi

echo "완료 시간: $(date)"
echo "lightgbm 완료 후 메모리 상태:"
check_memory
cleanup_memory

# 3분 대기
echo "메모리 정리를 위해 3분 대기..."
for i in {1..3}; do
    echo "대기 중... ($i/3)"
    sleep 60
    check_memory
done

echo ""
echo "=== Phase 2 완료 ==="
echo "완료 시간: $(date)"

# Phase 2와 Phase 3 사이 강력한 메모리 정리
echo ""
echo "Phase 2와 Phase 3 사이 강력한 메모리 정리..."
cleanup_memory

# 15분 대기
echo "Phase 3 시작 전 15분 대기..."
for i in {1..15}; do
    echo "대기 중... ($i/15)"
    sleep 60
    check_memory
done

# Phase 3: SMOTE 리샘플링 모델들
echo ""
echo "========================================"
echo "Phase 3: SMOTE 리샘플링 모델 실행 시작"
echo "========================================"

# XGBoost (SMOTE)
echo ""
echo "=== Phase 3 Step 1: xgboost (SMOTE 리샘플링) 하이퍼파라미터 튜닝 시작 ==="
echo "시작 시간: $(date)"
python scripts/run_hyperparameter_tuning.py \
    --model-type xgboost \
    --experiment-type resampling \
    --experiment-name resampling \
    --n-trials ${N_TRIALS} \
    --n-jobs 4 \
    > logs/xgboost_smote_${TIMESTAMP}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ xgboost 성공적으로 완료"
else
    echo "❌ xgboost 실패 (종료 코드: $?)"
    echo "⚠️ 실패했지만 다음 단계로 진행합니다."
fi

echo "완료 시간: $(date)"
echo "xgboost 완료 후 메모리 상태:"
check_memory
cleanup_memory

# 3분 대기
echo "메모리 정리를 위해 3분 대기..."
for i in {1..3}; do
    echo "대기 중... ($i/3)"
    sleep 60
    check_memory
done

# LightGBM (SMOTE)
echo ""
echo "=== Phase 3 Step 2: lightgbm (SMOTE 리샘플링) 하이퍼파라미터 튜닝 시작 ==="
echo "시작 시간: $(date)"
python scripts/run_hyperparameter_tuning.py \
    --model-type lightgbm \
    --experiment-type resampling \
    --experiment-name resampling \
    --n-trials ${N_TRIALS} \
    --n-jobs 4 \
    > logs/lightgbm_smote_${TIMESTAMP}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ lightgbm 성공적으로 완료"
else
    echo "❌ lightgbm 실패 (종료 코드: $?)"
    echo "⚠️ 실패했지만 다음 단계로 진행합니다."
fi

echo "완료 시간: $(date)"
echo "lightgbm 완료 후 메모리 상태:"
check_memory
cleanup_memory

# 3분 대기
echo "메모리 정리를 위해 3분 대기..."
for i in {1..3}; do
    echo "대기 중... ($i/3)"
    sleep 60
    check_memory
done

# CatBoost (SMOTE)
echo ""
echo "=== Phase 3 Step 3: catboost (SMOTE 리샘플링) 하이퍼파라미터 튜닝 시작 ==="
echo "시작 시간: $(date)"
python scripts/run_hyperparameter_tuning.py \
    --model-type catboost \
    --experiment-type resampling \
    --experiment-name resampling \
    --n-trials ${N_TRIALS} \
    --n-jobs 4 \
    > logs/catboost_smote_${TIMESTAMP}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ catboost 성공적으로 완료"
else
    echo "❌ catboost 실패 (종료 코드: $?)"
    echo "⚠️ 실패했지만 다음 단계로 진행합니다."
fi

echo "완료 시간: $(date)"
echo "catboost 완료 후 메모리 상태:"
check_memory
cleanup_memory

# 3분 대기
echo "메모리 정리를 위해 3분 대기..."
for i in {1..3}; do
    echo "대기 중... ($i/3)"
    sleep 60
    check_memory
done

# Random Forest (SMOTE)
echo ""
echo "=== Phase 3 Step 4: random_forest (SMOTE 리샘플링) 하이퍼파라미터 튜닝 시작 ==="
echo "시작 시간: $(date)"
python scripts/run_hyperparameter_tuning.py \
    --model-type random_forest \
    --experiment-type resampling \
    --experiment-name resampling \
    --n-trials ${N_TRIALS} \
    --n-jobs 4 \
    > logs/random_forest_smote_${TIMESTAMP}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ random_forest 성공적으로 완료"
else
    echo "❌ random_forest 실패 (종료 코드: $?)"
    echo "⚠️ 실패했지만 다음 단계로 진행합니다."
fi

echo "완료 시간: $(date)"
echo "random_forest 완료 후 메모리 상태:"
check_memory
cleanup_memory

# 최종 메모리 정리
echo ""
echo "최종 메모리 정리..."
cleanup_memory

echo ""
echo "========================================"
echo "전체 실행 완료"
echo "========================================"
echo "완료 시간: $(date)"

echo ""
echo "실행 완료! 로그 파일을 확인하세요:"
echo "로그 디렉토리: logs/"
echo "타임스탬프: ${TIMESTAMP}" 