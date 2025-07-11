#!/usr/bin/env python3
"""
Optuna 시각화 MLflow 로깅 테스트 스크립트

이 스크립트는 Optuna 시각화가 MLflow에 제대로 로깅되는지 테스트합니다.
"""

import sys
from pathlib import Path
import optuna
import mlflow
import logging
import numpy as np
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.hyperparameter_tuning import log_optuna_visualizations_to_mlflow, log_optuna_dashboard_to_mlflow

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def objective(trial):
    """테스트용 목적 함수"""
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    z = trial.suggest_int('z', 1, 100)
    
    # 간단한 목적 함수
    return (x - 2) ** 2 + (y + 3) ** 2 + z * 0.1


def create_test_study():
    """테스트용 Optuna study를 생성합니다."""
    logger.info("테스트용 Optuna study 생성 시작")
    
    # Study 생성
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 20번의 trial 실행
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    logger.info(f"Study 생성 완료 - 최고 성능: {study.best_value:.4f}")
    logger.info(f"최적 파라미터: {study.best_params}")
    
    return study


def test_optuna_mlflow_logging():
    """Optuna 시각화 MLflow 로깅을 테스트합니다."""
    logger.info("=== Optuna 시각화 MLflow 로깅 테스트 시작 ===")
    
    # MLflow 실험 설정
    experiment_name = "optuna_visualization_test"
    mlflow.set_experiment(experiment_name)
    
    # 테스트용 study 생성
    study = create_test_study()
    
    # MLflow run 시작
    with mlflow.start_run(run_name="optuna_visualization_test"):
        try:
            # Optuna 시각화를 MLflow에 로깅
            logger.info("Optuna 시각화 MLflow 로깅 시작")
            log_optuna_visualizations_to_mlflow(study)
            
            # Optuna 대시보드 데이터를 MLflow에 로깅
            logger.info("Optuna 대시보드 데이터 MLflow 로깅 시작")
            log_optuna_dashboard_to_mlflow(study)
            
            # 추가 메트릭 로깅
            mlflow.log_metric("test_best_value", study.best_value)
            mlflow.log_metric("test_n_trials", len(study.trials))
            
            # 파라미터 로깅
            for param, value in study.best_params.items():
                mlflow.log_param(f"best_{param}", value)
            
            logger.info("Optuna 시각화 MLflow 로깅 테스트 완료")
            
        except Exception as e:
            logger.error(f"Optuna 시각화 MLflow 로깅 테스트 실패: {e}")
            raise


if __name__ == "__main__":
    try:
        test_optuna_mlflow_logging()
        print("✅ Optuna 시각화 MLflow 로깅 테스트 성공!")
        print("MLflow UI에서 결과를 확인하세요: http://localhost:5000")
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        sys.exit(1) 