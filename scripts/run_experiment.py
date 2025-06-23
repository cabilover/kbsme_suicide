#!/usr/bin/env python3
"""
실험 실행 메인 스크립트

이 스크립트는 전체 ML 파이프라인을 실행하며, 데이터 분할부터 모델 학습 및 평가까지 포함합니다.
리샘플링 실험 기능도 포함되어 있습니다.
"""

import argparse
import logging
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from typing import Dict, Any, List

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
    
    # XGBoost 모델 파라미터 (상세 로깅)
    xgboost_config = config['model']['xgboost']
    model_params = {
        # 기본 파라미터
        "n_estimators": xgboost_config['n_estimators'],
        "max_depth": xgboost_config['max_depth'],
        "learning_rate": xgboost_config['learning_rate'],
        "subsample": xgboost_config['subsample'],
        "colsample_bytree": xgboost_config['colsample_bytree'],
        "min_child_weight": xgboost_config['min_child_weight'],
        "scale_pos_weight": xgboost_config['scale_pos_weight'],
        "early_stopping_rounds": xgboost_config['early_stopping_rounds'],
        "verbose": xgboost_config['verbose'],
        
        # Focal Loss 파라미터
        "use_focal_loss": xgboost_config.get('use_focal_loss', False),
        "focal_loss_alpha": xgboost_config.get('focal_loss', {}).get('alpha', 0.25),
        "focal_loss_gamma": xgboost_config.get('focal_loss', {}).get('gamma', 2.0),
        
        # 파라미터 소스 추적
        "param_source": "config_file",
        "config_file_path": "configs/default_config.yaml"
    }
    
    # 학습 파라미터
    training_params = {
        "early_stopping": config['training']['early_stopping'],
        "patience": config['training']['patience'],
        "handle_imbalance": config['training']['handle_imbalance'],
        "save_best_model": config['training']['save_best_model'],
        "model_save_path": config['training']['model_save_path']
    }
    
    # 리샘플링 파라미터
    resampling_config = config.get('resampling', {})
    resampling_params = {
        "resampling_enabled": resampling_config.get('enabled', False),
        "resampling_method": resampling_config.get('method', 'none'),
        "resampling_random_state": resampling_config.get('random_state', 42)
    }
    
    # 리샘플링 기법별 파라미터
    if resampling_config.get('enabled', False):
        method = resampling_config.get('method', 'none')
        if method == 'smote':
            resampling_params['smote_k_neighbors'] = resampling_config.get('smote_k_neighbors', 5)
        elif method == 'borderline_smote':
            resampling_params['borderline_smote_k_neighbors'] = resampling_config.get('borderline_smote_k_neighbors', 5)
        elif method == 'adasyn':
            resampling_params['adasyn_k_neighbors'] = resampling_config.get('adasyn_k_neighbors', 5)
        elif method == 'under_sampling':
            resampling_params['under_sampling_strategy'] = resampling_config.get('under_sampling_strategy', 'random')
        elif method == 'hybrid':
            resampling_params['hybrid_strategy'] = resampling_config.get('hybrid_strategy', 'smote_tomek')
    
    # 모든 파라미터 로깅
    all_params = {
        **data_split_params,
        **validation_params,
        **preprocessing_params,
        **feature_params,
        **model_params,
        **training_params,
        **resampling_params
    }
    
    mlflow.log_params(all_params)
    logger.info("실험 파라미터 로깅 완료")
    
    # XGBoost 파라미터 상세 출력 (디버깅용)
    logger.info("=== XGBoost 파라미터 확인 ===")
    for key, value in model_params.items():
        if key.startswith('param_source') or key.startswith('config_file'):
            continue
        logger.info(f"  {key}: {value}")
    
    # 리샘플링 파라미터 출력
    if resampling_params['resampling_enabled']:
        logger.info("=== 리샘플링 파라미터 확인 ===")
        for key, value in resampling_params.items():
            logger.info(f"  {key}: {value}")


def run_resampling_comparison_experiment(train_val_df: pd.DataFrame, test_df: pd.DataFrame, 
                                       config: Dict[str, Any], resampling_methods: List[str]) -> Dict[str, Any]:
    """
    다양한 리샘플링 기법을 비교하는 실험을 실행합니다.
    
    Args:
        train_val_df: 훈련/검증 데이터
        test_df: 테스트 데이터
        config: 기본 설정
        resampling_methods: 테스트할 리샘플링 기법 리스트
        
    Returns:
        비교 실험 결과
    """
    logger.info("=== 리샘플링 비교 실험 시작 ===")
    
    comparison_results = {
        'methods': {},
        'best_method': None,
        'best_score': -np.inf,
        'summary': {}
    }
    
    for method in resampling_methods:
        logger.info(f"\n--- {method.upper()} 실험 시작 ---")
        
        # 설정 복사 및 리샘플링 설정 업데이트
        method_config = config.copy()
        method_config['resampling'] = {
            'enabled': method != 'none',
            'method': method if method != 'none' else 'smote',  # none이 아닌 경우에만 설정
            'random_state': 42
        }
        
        # 리샘플링 기법별 파라미터 설정
        if method == 'smote':
            method_config['resampling']['smote_k_neighbors'] = 5
        elif method == 'borderline_smote':
            method_config['resampling']['borderline_smote_k_neighbors'] = 5
        elif method == 'adasyn':
            method_config['resampling']['adasyn_k_neighbors'] = 5
        elif method == 'under_sampling':
            method_config['resampling']['under_sampling_strategy'] = 'random'
        elif method == 'hybrid':
            method_config['resampling']['hybrid_strategy'] = 'smote_tomek'
        
        # MLflow 중첩 실행
        with mlflow.start_run(run_name=f"resampling_{method}", nested=True):
            # 실험 파라미터 로깅
            log_experiment_params(method_config)
            
            # 교차 검증 실행
            cv_results = run_cross_validation(train_val_df, method_config)
            
            if not cv_results or not cv_results.get('fold_results'):
                logger.warning(f"{method} 실험 실패")
                continue
            
            # 최종 테스트 세트 평가
            final_metrics = evaluate_final_model(test_df, cv_results, method_config)
            
            # 결과 수집
            method_results = {
                'cv_results': cv_results,
                'final_metrics': final_metrics,
                'config': method_config
            }
            
            comparison_results['methods'][method] = method_results
            
            # 주요 성능 지표 추출 (F1-Score 기준)
            primary_metric = 'suicide_a_next_year_f1'
            if final_metrics and primary_metric in str(final_metrics):
                # final_metrics에서 해당 지표 찾기
                score = 0.0
                for target, target_info in final_metrics.items():
                    if 'f1' in target_info.get('metrics', {}):
                        score = target_info['metrics']['f1']
                        break
                
                if score > comparison_results['best_score']:
                    comparison_results['best_score'] = score
                    comparison_results['best_method'] = method
            
            logger.info(f"{method} 실험 완료")
    
    # 비교 결과 요약
    logger.info("\n=== 리샘플링 비교 결과 요약 ===")
    for method, results in comparison_results['methods'].items():
        final_metrics = results['final_metrics']
        if final_metrics:
            for target, target_info in final_metrics.items():
                if 'f1' in target_info.get('metrics', {}):
                    f1_score = target_info['metrics']['f1']
                    logger.info(f"{method}: {target} F1-Score = {f1_score:.4f}")
    
    logger.info(f"최고 성능 기법: {comparison_results['best_method']} (F1-Score: {comparison_results['best_score']:.4f})")
    
    return comparison_results


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
    from src.evaluation import (
        calculate_all_metrics, print_evaluation_summary,
        evaluate_with_advanced_metrics, create_comprehensive_evaluation_report,
        save_advanced_evaluation_plots, analyze_fold_performance_distribution,
        calculate_confidence_intervals, analyze_fold_variability
    )
    
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
    
    # 기본 평가 지표 계산
    final_metrics = calculate_all_metrics(y_test, test_predictions, test_proba, config)
    
    # 고급 평가 기능 수행
    logger.info("=== 고급 평가 분석 ===")
    
    # 폴드별 성능 분석
    if cv_results.get('fold_results'):
        fold_results = cv_results['fold_results']
        
        # 폴드별 성능 분포 분석
        performance_distribution = analyze_fold_performance_distribution(fold_results)
        logger.info("폴드별 성능 분포 분석:")
        for target, metrics in performance_distribution.items():
            logger.info(f"\n{target}:")
            for metric, stats in metrics.items():
                logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 폴드 간 변동성 분석
        variability = analyze_fold_variability(fold_results)
        logger.info("\n폴드 간 변동성 분석:")
        for target, metrics in variability.items():
            logger.info(f"\n{target}:")
            for metric, var_info in metrics.items():
                stability = var_info['stability']
                cv = var_info['coefficient_of_variation']
                logger.info(f"  {metric}: {stability} (CV: {cv:.3f})")
        
        # 신뢰구간 계산
        logger.info("\n신뢰구간 분석:")
        for target in fold_results[0].get('metrics', {}).keys():
            for metric in ['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy']:
                if metric in fold_results[0]['metrics'][target]:
                    values = [fold['metrics'][target].get(metric, 0) for fold in fold_results]
                    values = [v for v in values if v is not None]
                    
                    if len(values) > 1:
                        ci = calculate_confidence_intervals(values, 0.95)
                        logger.info(f"  {target}_{metric}: {ci['mean']:.4f} ({ci['lower']:.4f} - {ci['upper']:.4f})")
        
        # MLflow에 고급 메트릭 로깅
        if performance_distribution:
            for target, metrics in performance_distribution.items():
                for metric, stats in metrics.items():
                    mlflow.log_metric(f"fold_analysis_{target}_{metric}_mean", stats['mean'])
                    mlflow.log_metric(f"fold_analysis_{target}_{metric}_std", stats['std'])
                    mlflow.log_metric(f"fold_analysis_{target}_{metric}_min", stats['min'])
                    mlflow.log_metric(f"fold_analysis_{target}_{metric}_max", stats['max'])
        
        # 변동성 메트릭 로깅
        if variability:
            for target, metrics in variability.items():
                for metric, var_info in metrics.items():
                    mlflow.log_metric(f"fold_variability_{target}_{metric}_cv", var_info['coefficient_of_variation'])
    
    # 종합 평가 리포트 생성
    comprehensive_report = create_comprehensive_evaluation_report(
        final_metrics, 
        cv_results.get('fold_results'),
        config
    )
    
    # 권장사항 출력
    if comprehensive_report.get('recommendations'):
        logger.info("\n=== 모델 개선 권장사항 ===")
        for i, recommendation in enumerate(comprehensive_report['recommendations'], 1):
            logger.info(f"{i}. {recommendation}")
    
    # 고급 평가 플롯 저장
    if config.get('evaluation', {}).get('save_plots', True):
        for target in target_columns:
            if target in y_test.columns and target in test_predictions.columns:
                try:
                    # 확률 예측이 있는 경우에만 플롯 생성
                    if test_proba is not None and target in test_proba:
                        save_advanced_evaluation_plots(
                            y_test[target], 
                            test_predictions[target], 
                            test_proba[target],
                            target, 
                            config
                        )
                except Exception as e:
                    logger.warning(f"{target} 플롯 생성 실패: {e}")
    
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
        "test_ids_count": len(test_df[config['time_series']['id_column']].unique()),
        "comprehensive_report": comprehensive_report
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(final_results, f, indent=2, default=str)
        mlflow.log_artifact(f.name, "final_test_results.json")
    
    logger.info("최종 테스트 세트 평가 완료")
    return final_metrics


def run_full_experiment(config_path: str, data_path: str, nrows: int = None, 
                       resampling_comparison: bool = False, resampling_methods: List[str] = None):
    """
    전체 ML 파이프라인 실험을 실행합니다.
    
    Args:
        config_path: 설정 파일 경로
        data_path: 데이터 파일 경로
        nrows: 테스트용으로 읽을 행 수 (None이면 전체)
        resampling_comparison: 리샘플링 비교 실험 여부
        resampling_methods: 비교할 리샘플링 기법 리스트
        
    Returns:
        bool: 실험 성공 여부
    """
    # 설정 로드
    config = load_config(config_path)
    
    # MLflow 실험 시작
    experiment_name = "resampling_comparison" if resampling_comparison else "kbsmc_suicide_prediction"
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
        
        if resampling_comparison:
            # 리샘플링 비교 실험
            logger.info("=== 4단계: 리샘플링 비교 실험 ===")
            if resampling_methods is None:
                resampling_methods = ['none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid']
            
            comparison_results = run_resampling_comparison_experiment(
                train_val_df, test_df, config, resampling_methods
            )
            
            # 비교 결과 로깅
            mlflow.log_param("resampling_comparison", True)
            mlflow.log_param("resampling_methods", str(resampling_methods))
            mlflow.log_param("best_resampling_method", comparison_results.get('best_method', 'none'))
            mlflow.log_metric("best_resampling_score", comparison_results.get('best_score', 0.0))
            
            logger.info("리샘플링 비교 실험 완료")
            return True
        else:
            # 일반 실험
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
    parser.add_argument(
        "--resampling-comparison",
        action="store_true",
        help="리샘플링 기법 비교 실험 실행"
    )
    parser.add_argument(
        "--resampling-methods",
        nargs="+",
        choices=['none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid'],
        default=['none', 'smote', 'borderline_smote', 'adasyn'],
        help="비교할 리샘플링 기법들"
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
    experiment_name = "resampling_comparison" if args.resampling_comparison else args.experiment_name
    setup_mlflow_experiment(experiment_name)
    
    # 실험 실행
    success = run_full_experiment(
        args.config, 
        args.data, 
        args.nrows,
        args.resampling_comparison,
        args.resampling_methods
    )
    
    if success:
        if args.resampling_comparison:
            logger.info("리샘플링 비교 실험이 성공적으로 완료되었습니다.")
        else:
            logger.info("실험이 성공적으로 완료되었습니다.")
        logger.info("MLflow UI에서 결과를 확인하세요: mlflow ui")
        return True
    else:
        logger.error("실험 실행 중 오류가 발생했습니다.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 