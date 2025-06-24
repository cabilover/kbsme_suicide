#!/bin/bash

# 전체 데이터에 대해 모델별 하이퍼파라미터 튜닝을 순차적으로 실행
# 로그: 시작/종료 시간, 모델명

set -e
MODELS=(xgboost lightgbm random_forest catboost)
for MODEL in "${MODELS[@]}"; do
    echo "========================================="
    echo "[START] $MODEL tuning: $(date)"
    echo "========================================="
    python scripts/run_hyperparameter_tuning.py --model-type $MODEL --experiment-type hyperparameter_tuning
    echo "========================================="
    echo "[END] $MODEL tuning: $(date)"
    echo "========================================="
    echo ""
done
echo "모든 모델 튜닝이 완료되었습니다. (종료: $(date))"