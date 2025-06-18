#!/usr/bin/env python3
"""
실험 실행 메인 스크립트

이 스크립트는 전체 ML 파이프라인을 실행하며, 데이터 분할부터 모델 학습 및 평가까지 포함합니다.
"""

import argparse
import logging
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.splits import (
    load_config, 
    split_test_set, 
    get_cv_splits, 
    validate_splits, 
    log_splits_info
)
from src.training import run_cross_validation

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mlflow_experiment(experiment_name: str = "kbsmc_suicide_prediction"):
    """
    MLflow 실험을 설정합니다.
    
    Args:
        experiment_name: 실험 이름
        
    Returns:
        experiment_id: 실험 ID
    """
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    logger.info(f"MLflow 실험 설정: {experiment_name} (ID: {experiment_id})")
    return experiment_id


def log_experiment_params(config: Dict[str, Any]):
    """
    실험 파라미터를 MLflow에 로깅합니다.
    
    Args:
        config: 설정 딕셔너리
    """
    # 데이터 분할 파라미터
    data_split_params = {
        "test_ids_ratio": config['data_split']['test_ids_ratio'],
        "random_state": config['data_split']['random_state']
    }
    
    # 검증 전략 파라미터
    validation_params = {
        "validation_strategy": config['validation']['strategy'],
        "val_ids_ratio": config['validation']['val_ids_ratio']
    }
    
    # 전처리 파라미터
    preprocessing_params = {
        "numerical_imputation_strategy": config['preprocessing']['numerical_imputation']['strategy'],
        "categorical_imputation_strategy": config['preprocessing']['categorical_imputation']['strategy'],
        "categorical_encoding_method": config['preprocessing']['categorical_encoding']['method'],
        "time_series_fallback_strategy": config['preprocessing']['time_series_imputation']['fallback_strategy']
    }
    
    # 피처 엔지니어링 파라미터
    feature_params = {
        "feature_validation_scope": config['features']['validation']['scope'],
        "feature_validation_sample_size": config['features']['validation']['sample_size'],
        "enable_lagged_features": config['features']['enable_lagged_features'],
        "enable_rolling_stats": config['features']['enable_rolling_stats']
    }
    
    # 모델 파라미터
    model_params = {
        "n_estimators": config['model']['xgboost']['n_estimators'],
        "max_depth": config['model']['xgboost']['max_depth'],
        "learning_rate": config['model']['xgboost']['learning_rate'],
        "early_stopping_rounds": config['model']['xgboost']['early_stopping_rounds']
    }
    
    # 학습 파라미터
    training_params = {
        "early_stopping": config['training']['early_stopping'],
        "patience": config['training']['patience'],
        "handle_imbalance": config['training']['handle_imbalance'],
        "save_best_model": config['training']['save_best_model'],
        "model_save_path": config['training']['model_save_path']
    }
    
    # 모든 파라미터 로깅
    all_params = {
        **data_split_params,
        **validation_params,
        **preprocessing_params,
        **feature_params,
        **model_params,
        **training_params
    }
    
    mlflow.log_params(all_params)
    logger.info("실험 파라미터 로깅 완료")


def evaluate_final_model(test_df: pd.DataFrame, cv_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    최종 테스트 세트에서 모델을 평가합니다.
    
    Args:
        test_df: 테스트 데이터프레임
        cv_results: 교차 검증 결과
        config: 설정 딕셔너리
        
    Returns:
        최종 평가 결과
    """
    from src.preprocessing import transform_data
    from src.feature_engineering import transform_features, get_feature_columns, get_target_columns, get_target_columns_from_data
    from src.evaluation import calculate_all_metrics, print_evaluation_summary
    
    logger.info("=== 최종 테스트 세트 평가 ===")
    
    if not cv_results.get('best_model'):
        logger.warning("최고 성능 모델이 없습니다. 최종 평가를 건너뜁니다.")
        return {}
    
    best_fold = cv_results['best_fold']
    best_model = cv_results['best_model']
    
    # 최고 성능 폴드의 전처리 파이프라인과 피처 정보 사용
    best_fold_result = cv_results['fold_results'][best_fold - 1]
    preprocessor = best_fold_result['preprocessor']
    feature_info = best_fold_result['feature_info']
    
    # 테스트 데이터 전처리
    test_processed = transform_data(test_df, preprocessor, config)
    test_engineered = transform_features(test_processed, feature_info, config)
    
    # 피처와 타겟 분리
    feature_columns = get_feature_columns(test_engineered, config)
    target_columns = get_target_columns_from_data(test_engineered, config)
    
    X_test = test_engineered[feature_columns]
    y_test = test_engineered[target_columns]
    
    # 예측 수행
    test_predictions = best_model.predict(X_test)
    
    # 확률 예측 (분류 문제의 경우)
    test_proba = None
    if hasattr(best_model, 'predict_proba'):
        test_proba = best_model.predict_proba(X_test)
    
    # 평가 지표 계산
    final_metrics = calculate_all_metrics(y_test, test_predictions, test_proba, config)
    
    # 결과 출력
    print_evaluation_summary(
        final_metrics, 
        feature_validation_results=cv_results.get('feature_validation_results'),
        early_stopping_used=cv_results.get('early_stopping_used', False),
        cv_results=cv_results
    )
    
    # MLflow에 최종 결과 로깅
    for target, target_info in final_metrics.items():
        for metric_name, value in target_info['metrics'].items():
            mlflow.log_metric(f"final_test_{target}_{metric_name}", value)
    
    # 최종 테스트 결과를 아티팩트로 저장
    import json
    import tempfile
    from pathlib import Path
    
    final_results = {
        "best_fold": best_fold,
        "best_score": cv_results.get('best_score'),
        "test_metrics": final_metrics,
        "test_samples": len(test_df),
        "test_ids_count": len(test_df[config['time_series']['id_column']].unique())
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(final_results, f, indent=2, default=str)
        mlflow.log_artifact(f.name, "final_test_results.json")
    
    logger.info("최종 테스트 세트 평가 완료")
    return final_metrics


def run_full_experiment(config_path: str, data_path: str, nrows: int = None):
    """
    전체 ML 파이프라인 실험을 실행합니다.
    
    Args:
        config_path: 설정 파일 경로
        data_path: 데이터 파일 경로
        nrows: 테스트용으로 읽을 행 수 (None이면 전체)
        
    Returns:
        bool: 실험 성공 여부
    """
    # 설정 로드
    config = load_config(config_path)
    
    # MLflow 실험 시작
    with mlflow.start_run(run_name="full_ml_pipeline") as parent_run:
        # 실험 파라미터 로깅
        log_experiment_params(config)
        
        # 데이터 로드
        logger.info(f"데이터 로드 중: {data_path}")
        if nrows:
            df = pd.read_csv(data_path, nrows=nrows)
            logger.info(f"테스트용 데이터 로드: {len(df):,} 행")
        else:
            df = pd.read_csv(data_path)
            logger.info(f"전체 데이터 로드: {len(df):,} 행")
        
        # 데이터 기본 정보 로깅
        mlflow.log_params({
            "total_samples": len(df),
            "total_ids": df[config['time_series']['id_column']].nunique(),
            "year_range": f"{df[config['time_series']['year_column']].min()}-{df[config['time_series']['year_column']].max()}"
        })
        
        # 1. 테스트 세트 분리
        logger.info("=== 1단계: 테스트 세트 분리 ===")
        train_val_df, test_df, test_ids = split_test_set(df, config)
        
        # 2. 분할 검증
        logger.info("=== 2단계: 분할 검증 ===")
        if not validate_splits(train_val_df, test_df, config):
            logger.error("분할 검증 실패!")
            return False
        
        # 3. 분할 정보 로깅
        logger.info("=== 3단계: 분할 정보 로깅 ===")
        splits_info = log_splits_info(train_val_df, test_df, config)
        
        # MLflow에 분할 정보 로깅
        mlflow.log_metrics({
            "train_val_samples": splits_info['train_val_samples'],
            "test_samples": splits_info['test_samples'],
            "train_val_ids": splits_info['train_val_ids'],
            "test_ids": splits_info['test_ids'],
            "test_ratio_actual": splits_info['test_samples'] / splits_info['total_samples']
        })
        
        # 4. 교차 검증 및 모델 학습
        logger.info("=== 4단계: 교차 검증 및 모델 학습 ===")
        cv_results = run_cross_validation(train_val_df, config)
        
        if not cv_results or not cv_results.get('fold_results'):
            logger.error("교차 검증 실패!")
            return False
        
        # 5. 최종 테스트 세트 평가
        logger.info("=== 5단계: 최종 테스트 세트 평가 ===")
        final_metrics = evaluate_final_model(test_df, cv_results, config)
        
        # 6. 결과 요약 및 로깅
        logger.info("=== 6단계: 결과 요약 ===")
        
        # 집계 메트릭 로깅
        aggregate_metrics = cv_results.get('aggregate_metrics', {})
        for metric_name, value in aggregate_metrics.items():
            mlflow.log_metric(f"final_{metric_name}", value)
        
        # 테스트 세트 정보를 아티팩트로 저장
        test_set_info = {
            "test_samples": len(test_df),
            "test_ids_count": len(test_df[config['time_series']['id_column']].unique()),
            "test_years_range": (
                test_df[config['time_series']['year_column']].min(),
                test_df[config['time_series']['year_column']].max()
            ),
            "note": "이 테스트 세트는 최종 모델 평가 시에만 사용됩니다."
        }
        
        import json
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_set_info, f, indent=2, default=str)
            mlflow.log_artifact(f.name, "test_set_info.json")
        
        # 7. 실험 완료
        logger.info("=== 7단계: 실험 완료 ===")
        logger.info("전체 ML 파이프라인 실험이 성공적으로 완료되었습니다.")
        logger.info(f"생성된 폴드 수: {len(cv_results['fold_results'])}")
        logger.info(f"테스트 세트 크기: {len(test_df):,} 행")
        logger.info(f"최고 성능 폴드: {cv_results.get('best_fold', 'N/A')}")
        
        # 주요 성능 지표 출력
        if aggregate_metrics:
            logger.info("=== 주요 성능 지표 ===")
            for metric_name, value in aggregate_metrics.items():
                if 'mean' in metric_name:
                    logger.info(f"{metric_name}: {value:.4f}")
        
        return True


def main():
    """
    메인 함수
    """
    parser = argparse.ArgumentParser(description="KBSMC 자살 예측 실험 실행")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/processed/processed_data_with_features.csv",
        help="데이터 파일 경로"
    )
    parser.add_argument(
        "--nrows", 
        type=int, 
        default=10000,
        help="테스트용으로 읽을 행 수 (None이면 전체)"
    )
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default="kbsmc_suicide_prediction",
        help="MLflow 실험 이름"
    )
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.config).exists():
        logger.error(f"설정 파일이 존재하지 않습니다: {args.config}")
        return False
    
    if not Path(args.data).exists():
        logger.error(f"데이터 파일이 존재하지 않습니다: {args.data}")
        return False
    
    # MLflow 실험 설정
    setup_mlflow_experiment(args.experiment_name)
    
    # 실험 실행
    success = run_full_experiment(args.config, args.data, args.nrows)
    
    if success:
        logger.info("실험이 성공적으로 완료되었습니다.")
        logger.info("MLflow UI에서 결과를 확인하세요: mlflow ui")
        return True
    else:
        logger.error("실험 실행 중 오류가 발생했습니다.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 