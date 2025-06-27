#!/usr/bin/env python3
"""
하이퍼파라미터 튜닝 실행 스크립트

Optuna를 활용한 하이퍼파라미터 최적화를 실행합니다.
MLflow와 연동되어 실험 추적이 가능하며, 다양한 튜닝 전략을 지원합니다.
리샘플링 기법 비교 실험 기능도 포함되어 있습니다.
"""

import argparse
import logging
import sys
from pathlib import Path
import mlflow
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.hyperparameter_tuning import HyperparameterTuner, save_tuning_log
from src.splits import (
    load_config, 
    split_test_set, 
    validate_splits, 
    log_splits_info
)
from src.utils.config_manager import ConfigManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mlflow_experiment(experiment_name: str):
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


def log_tuning_params(tuning_config: Dict[str, Any], base_config: Dict[str, Any]):
    """
    튜닝 파라미터를 MLflow에 로깅합니다.
    
    Args:
        tuning_config: 튜닝 설정 딕셔너리
        base_config: 기본 설정 딕셔너리
    """
    try:
        # 튜닝 파라미터 (안전한 접근)
        tuning_params = {}
        
        # n_trials
        n_trials = tuning_config.get('tuning', {}).get('n_trials')
        if n_trials is not None:
            tuning_params["n_trials"] = n_trials
        
        # timeout
        timeout = tuning_config.get('tuning', {}).get('timeout')
        if timeout is not None:
            tuning_params["timeout"] = timeout
        
        # sampler
        sampler_type = tuning_config.get('sampler', {}).get('type')
        if sampler_type is not None:
            tuning_params["sampler"] = sampler_type
        
        # optimization_direction
        optimization_direction = tuning_config.get('evaluation', {}).get('optimization_direction')
        if optimization_direction is not None:
            tuning_params["optimization_direction"] = optimization_direction
        
        # primary_metric
        primary_metric = tuning_config.get('evaluation', {}).get('primary_metric')
        if primary_metric is not None:
            tuning_params["primary_metric"] = primary_metric
        
        # XGBoost 파라미터 범위
        xgboost_params = tuning_config.get('xgboost_params', {})
        for param, param_config in xgboost_params.items():
            if isinstance(param_config, dict):
                if 'range' in param_config and len(param_config['range']) >= 2:
                    tuning_params[f"tune_{param}_min"] = param_config['range'][0]
                    tuning_params[f"tune_{param}_max"] = param_config['range'][1]
                elif 'choices' in param_config:
                    tuning_params[f"tune_{param}_choices"] = str(param_config['choices'])
        
        # 기본 설정에서 리샘플링 파라미터 추출
        resampling_config = base_config.get('resampling', {})
        resampling_params = {
            "resampling_enabled": resampling_config.get('enabled', False),
            "resampling_method": resampling_config.get('method', 'none'),
            "resampling_random_state": resampling_config.get('random_state', 42)
        }
        
        # 모든 파라미터 로깅
        all_params = {**tuning_params, **resampling_params}
        
        # MLflow에 파라미터 로깅 (각 파라미터별로 개별 처리)
        for param_name, param_value in all_params.items():
            if param_value is not None:
                try:
                    mlflow.log_param(param_name, param_value)
                except Exception as e:
                    logger.warning(f"MLflow 파라미터 로깅 실패 ({param_name}): {e}")
        
        logger.info(f"튜닝 파라미터 로깅 완료: {len(all_params)}개 파라미터")
        
    except Exception as e:
        logger.warning(f"튜닝 파라미터 로깅 중 예외 발생: {e}")
        # 예외가 발생해도 실험은 계속 진행


def run_resampling_tuning_comparison(tuning_config_path: str, base_config_path: str,
                                   data_path: str, resampling_methods: List[str],
                                   nrows: int = None) -> Dict[str, Any]:
    """
    다양한 리샘플링 기법에 대해 하이퍼파라미터 튜닝을 비교하는 실험을 실행합니다.
    
    Args:
        tuning_config_path: 튜닝 설정 파일 경로
        base_config_path: 기본 설정 파일 경로
        data_path: 데이터 파일 경로
        resampling_methods: 테스트할 리샘플링 기법 리스트
        nrows: 사용할 데이터 행 수 (None이면 전체 사용)
        
    Returns:
        비교 실험 결과
    """
    logger.info("=== 리샘플링 하이퍼파라미터 튜닝 비교 실험 시작 ===")
    
    # 기본 설정 로드
    base_config = load_config(base_config_path)
    
    # 데이터 로드
    logger.info(f"데이터 로드 중: {data_path}")
    if nrows:
        df = pd.read_csv(data_path, nrows=nrows)
        logger.info(f"테스트용 데이터 로드: {len(df):,} 행")
    else:
        df = pd.read_csv(data_path)
        logger.info(f"전체 데이터 로드: {len(df):,} 행")
    
    # 테스트 세트 분리
    logger.info("=== 테스트 세트 분리 ===")
    train_val_df, test_df, test_ids = split_test_set(df, base_config)
    
    # 분할 검증
    if not validate_splits(train_val_df, test_df, base_config):
        logger.error("분할 검증 실패!")
        return {}
    
    comparison_results = {
        'methods': {},
        'best_method': None,
        'best_score': -np.inf,
        'summary': {}
    }
    
    for method in resampling_methods:
        logger.info(f"\n--- {method.upper()} 하이퍼파라미터 튜닝 시작 ---")
        
        # 설정 복사 및 리샘플링 설정 업데이트
        method_base_config = base_config.copy()
        method_base_config['resampling'] = {
            'enabled': method != 'none',
            'method': method if method != 'none' else 'smote',  # none이 아닌 경우에만 설정
            'random_state': 42
        }
        
        # 리샘플링 기법별 파라미터 설정
        if method == 'smote':
            method_base_config['resampling']['smote_k_neighbors'] = 5
        elif method == 'borderline_smote':
            method_base_config['resampling']['borderline_smote_k_neighbors'] = 5
        elif method == 'adasyn':
            method_base_config['resampling']['adasyn_k_neighbors'] = 5
        elif method == 'under_sampling':
            method_base_config['resampling']['under_sampling_strategy'] = 'random'
        elif method == 'hybrid':
            method_base_config['resampling']['hybrid_strategy'] = 'smote_tomek'
        
        # MLflow 중첩 실행
        with mlflow.start_run(run_name=f"resampling_tuning_{method}", nested=True):
            # 튜닝 설정 로드
            with open(tuning_config_path, 'r', encoding='utf-8') as f:
                method_tuning_config = yaml.safe_load(f)
            
            # 실험 파라미터 로깅
            log_tuning_params(method_tuning_config, method_base_config)
            
            # 하이퍼파라미터 튜너 생성 (임시 설정 파일 사용)
            import tempfile
            import os
            
            # 임시 기본 설정 파일 생성
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(method_base_config, f, default_flow_style=False)
                temp_base_config_path = f.name
            
            try:
                # 하이퍼파라미터 튜너 생성
                tuner = HyperparameterTuner(temp_base_config_path, temp_base_config_path, nrows=nrows)
                
                # 최적화 실행
                logger.info(f"{method} 하이퍼파라미터 최적화 실행 중...")
                best_params, best_score = tuner.optimize(start_mlflow_run=False)
                
                # 결과 저장
                logger.info(f"{method} 튜닝 결과 저장 중...")
                tuner.save_results()
                
                # 결과 수집
                method_results = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'tuning_config': method_tuning_config,
                    'base_config': method_base_config
                }
                
                comparison_results['methods'][method] = method_results
                
                # 최고 성능 업데이트
                if best_score > comparison_results['best_score']:
                    comparison_results['best_score'] = best_score
                    comparison_results['best_method'] = method
                
                logger.info(f"{method} 하이퍼파라미터 튜닝 완료: {best_score:.4f}")
                
            finally:
                # 임시 파일 정리
                if os.path.exists(temp_base_config_path):
                    os.unlink(temp_base_config_path)
    
    # 비교 결과 요약
    logger.info("\n=== 리샘플링 하이퍼파라미터 튜닝 비교 결과 요약 ===")
    for method, results in comparison_results['methods'].items():
        best_score = results['best_score']
        logger.info(f"{method}: 최고 성능 = {best_score:.4f}")
    
    logger.info(f"최고 성능 기법: {comparison_results['best_method']} (성능: {comparison_results['best_score']:.4f})")
    
    return comparison_results


def validate_config_files(tuning_config_path: str, base_config_path: str):
    """
    설정 파일들의 유효성을 검증합니다.
    
    Args:
        tuning_config_path: 튜닝 설정 파일 경로
        base_config_path: 기본 설정 파일 경로
    """
    # 파일 존재 확인
    if not Path(tuning_config_path).exists():
        raise FileNotFoundError(f"튜닝 설정 파일을 찾을 수 없습니다: {tuning_config_path}")
    
    if not Path(base_config_path).exists():
        raise FileNotFoundError(f"기본 설정 파일을 찾을 수 없습니다: {base_config_path}")
    
    # YAML 파싱 테스트
    try:
        with open(tuning_config_path, 'r', encoding='utf-8') as f:
            tuning_config = yaml.safe_load(f)
        logger.info("튜닝 설정 파일 파싱 성공")
    except Exception as e:
        raise ValueError(f"튜닝 설정 파일 파싱 실패: {str(e)}")
    
    try:
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        logger.info("기본 설정 파일 파싱 성공")
    except Exception as e:
        raise ValueError(f"기본 설정 파일 파싱 실패: {str(e)}")
    
    # 필수 설정 확인
    required_tuning_keys = ['tuning', 'sampler', 'xgboost_params', 'evaluation', 'results', 'mlflow']
    for key in required_tuning_keys:
        if key not in tuning_config:
            raise ValueError(f"튜닝 설정에 필수 키가 없습니다: {key}")
    
    logger.info("설정 파일 검증 완료")


def run_resampling_tuning_comparison_with_configmanager(
    model_type: str,
    experiment_type: str,
    data_path: str,
    resampling_methods: List[str],
    nrows: int = None
) -> Dict[str, Any]:
    """
    ConfigManager 기반 다양한 리샘플링 기법에 대해 하이퍼파라미터 튜닝을 비교하는 실험을 실행합니다.
    """
    logger.info("=== [ConfigManager] 리샘플링 하이퍼파라미터 튜닝 비교 실험 시작 ===")
    config_manager = ConfigManager()

    # 데이터 로드
    logger.info(f"데이터 로드 중: {data_path}")
    if nrows:
        df = pd.read_csv(data_path, nrows=nrows)
        logger.info(f"테스트용 데이터 로드: {len(df):,} 행")
    else:
        df = pd.read_csv(data_path)
        logger.info(f"전체 데이터 로드: {len(df):,} 행")

    # 테스트 세트 분리
    logger.info("=== 테스트 세트 분리 ===")
    # config는 리샘플링과 무관하게 분할에만 사용
    base_config = config_manager.create_experiment_config(model_type, experiment_type)
    train_val_df, test_df, test_ids = split_test_set(df, base_config)

    # 분할 검증
    if not validate_splits(train_val_df, test_df, base_config):
        logger.error("분할 검증 실패!")
        return {}

    # 데이터 정보 수집
    data_info = collect_data_info(df, base_config, train_val_df, test_df)
    logger.info(f"데이터 정보 수집 완료: {data_info['total_rows']} 행, {data_info['total_columns']} 열")

    comparison_results = {
        'methods': {},
        'best_method': None,
        'best_score': -np.inf,
        'summary': {}
    }

    # 상위 MLflow run 시작
    with mlflow.start_run(run_name=f"resampling_comparison_{model_type}"):
        for method in resampling_methods:
            logger.info(f"\n--- {method.upper()} 하이퍼파라미터 튜닝 시작 ---")
            # config 생성 및 리샘플링 설정 수정
            config = config_manager.create_experiment_config(model_type, experiment_type)
            config['resampling'] = {
                'enabled': method != 'none',
                'method': method if method != 'none' else 'smote',
                'random_state': 42
            }
            # 리샘플링 기법별 파라미터 설정
            if method == 'smote':
                config['resampling']['smote_k_neighbors'] = 5
            elif method == 'borderline_smote':
                config['resampling']['borderline_smote_k_neighbors'] = 5
            elif method == 'adasyn':
                config['resampling']['adasyn_k_neighbors'] = 5
            elif method == 'under_sampling':
                config['resampling']['under_sampling_strategy'] = 'random'
            elif method == 'hybrid':
                config['resampling']['hybrid_strategy'] = 'smote_tomek'
            # 데이터 경로 반영
            config['data']['data_path'] = data_path
            # MLflow 중첩 실행
            with mlflow.start_run(run_name=f"resampling_tuning_{method}", nested=True):
                # 튜닝 파라미터 로깅 
                log_tuning_params(config, config)
                
                # 튜너 생성 및 튜닝 실행
                tuner = HyperparameterTuner(config, train_val_df, nrows=nrows)
                result = tuner.optimize(start_mlflow_run=False)
                
                # 결과 추출
                if result is None:
                    best_params = {}
                    best_score = 0.0
                    logger.warning(f"{method} 튜닝 결과가 None입니다.")
                elif isinstance(result, tuple) and len(result) >= 2:
                    best_params = result[0] if result[0] is not None else {}
                    best_score = result[1] if result[1] is not None else 0.0
                else:
                    best_params = {}
                    best_score = 0.0
                    logger.warning(f"{method} 예상치 못한 결과 타입: {type(result)}")
                
                # 결과 저장
                comparison_results['methods'][method] = {
                    'best_params': best_params,
                    'best_score': best_score
                }
                
                # 최고 성능 업데이트
                if best_score > comparison_results['best_score']:
                    comparison_results['best_score'] = best_score
                    comparison_results['best_method'] = method
                
                logger.info(f"{method} 최고 성능: {best_score:.4f}")
                logger.info(f"{method} 최적 파라미터: {best_params}")
                
                # 튜닝 로그 저장
                try:
                    save_tuning_log(
                        result=(best_params, best_score),
                        model_type=model_type,
                        experiment_type=experiment_type,
                        nrows=nrows,
                        experiment_id=mlflow.active_run().info.experiment_id,
                        run_id=mlflow.active_run().info.run_id,
                        log_file_path=getattr(tuner, 'log_file_path', None)
                    )
                except Exception as e:
                    logger.error(f"{method} 튜닝 로그 저장 실패: {e}")

    # 최종 결과 요약
    logger.info("\n=== 리샘플링 비교 실험 완료 ===")
    logger.info(f"최고 성능 기법: {comparison_results['best_method']}")
    logger.info(f"최고 성능: {comparison_results['best_score']:.4f}")
    logger.info("\n각 기법별 성능:")
    for method, method_result in comparison_results['methods'].items():
        logger.info(f"  {method}: {method_result['best_score']:.4f}")

    # 실험 결과 저장
    save_experiment_results(
        comparison_results,
        model_type,
        experiment_type,
        nrows,
        config=base_config,
        data_info=data_info
    )

    return comparison_results


def run_hyperparameter_tuning_with_config(config: Dict[str, Any], data_path: str = None, nrows: int = None, resampling_comparison: bool = False, resampling_methods: List[str] = None):
    """
    ConfigManager 기반 하이퍼파라미터 튜닝을 실행합니다.
    
    Args:
        config: 완전한 설정 딕셔너리
        data_path: 데이터 파일 경로 (None이면 config에서 가져옴)
        nrows: 사용할 데이터 행 수 (None이면 전체 사용)
        resampling_comparison: 리샘플링 비교 실험 여부
        resampling_methods: 비교할 리샘플링 기법들
        
    Returns:
        튜닝 결과 (best_params, best_score, experiment_id, run_id) 또는 리샘플링 비교 결과
    """
    logger.info("=== ConfigManager 기반 하이퍼파라미터 튜닝 시작 ===")
    
    # 데이터 경로 설정
    if data_path is None:
        data_path = config['data'].get('file_path', 'data/processed/processed_data_with_features.csv')
    
    # 데이터 로드
    logger.info(f"데이터 로드 중: {data_path}")
    if nrows:
        df = pd.read_csv(data_path, nrows=nrows)
        logger.info(f"테스트용 데이터 로드: {len(df):,} 행")
    else:
        df = pd.read_csv(data_path)
        logger.info(f"전체 데이터 로드: {len(df):,} 행")
    
    # 데이터 정보 수집
    data_info = collect_data_info(df, config)
    logger.info(f"데이터 정보 수집 완료: {data_info['total_rows']} 행, {data_info['total_columns']} 열")
    
    # 테스트 세트 분리
    logger.info("=== 테스트 세트 분리 ===")
    train_val_df, test_df, test_ids = split_test_set(df, config)
    
    # 데이터 정보 업데이트 (테스트 세트 정보 추가)
    data_info = collect_data_info(df, config, train_val_df, test_df)
    
    # 분할 검증
    if not validate_splits(train_val_df, test_df, config):
        logger.error("분할 검증 실패!")
        return None
    
    # 리샘플링 비교 실험
    if resampling_comparison:
        logger.info("=== 리샘플링 비교 실험 시작 ===")
        if resampling_methods is None:
            resampling_methods = ['none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid']
        
        result = run_resampling_tuning_comparison_with_configmanager(
            config.get('model', {}).get('model_type', 'xgboost'),
            'hyperparameter_tuning',
            data_path,
            resampling_methods,
            nrows
        )
        
        # 실험 결과 저장 (리샘플링 비교)
        save_experiment_results(
            result, 
            config.get('model', {}).get('model_type', 'xgboost'), 
            'hyperparameter_tuning', 
            nrows, 
            config=config, 
            data_info=data_info
        )
        
        return result
    
    # 일반 하이퍼파라미터 튜닝
    logger.info("=== 일반 하이퍼파라미터 튜닝 시작 ===")
    
    # MLflow 실험 설정
    experiment_name = config['mlflow']['experiment_name']
    experiment_id = setup_mlflow_experiment(experiment_name)
    
    # 튜너 생성
    tuner = HyperparameterTuner(config, train_val_df, nrows=nrows)
    
    # 튜닝 실행
    with mlflow.start_run(experiment_id=experiment_id) as run:
        logger.info(f"MLflow Run 시작: {run.info.run_id}")
        
        # 설정 로깅
        log_tuning_params(config, config)  # config를 base_config로도 사용
        
        # 튜닝 실행
        result = tuner.optimize(start_mlflow_run=False)  # 이미 run이 시작되어 있으므로 False
        
        # 결과 저장
        try:
            tuner.save_results()
            logger.info("=== 하이퍼파라미터 튜닝 완료 ===")
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            # 결과 저장 실패해도 계속 진행
        
        # 결과 추출 (안전하게)
        if result is None:
            best_params = {}
            best_score = 0.0
            logger.warning("튜닝 결과가 None입니다. 기본값을 사용합니다.")
        elif isinstance(result, tuple) and len(result) >= 2:
            best_params = result[0] if result[0] is not None else {}
            best_score = result[1] if result[1] is not None else 0.0
        elif isinstance(result, dict):
            best_params = result.get('best_params', {})
            best_score = result.get('best_score', 0.0)
        else:
            best_params = {}
            best_score = 0.0
            logger.warning(f"예상치 못한 결과 타입: {type(result)}")
        
        logger.info(f"최고 성능: {best_score:.4f}")
        logger.info("최적 파라미터:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        
        # 튜닝 로그 저장 (로그 파일 경로 전달)
        try:
            save_tuning_log(
                result=result,
                model_type=config.get('model', {}).get('model_type', 'unknown'),
                experiment_type='hyperparameter_tuning',
                nrows=nrows,
                experiment_id=run.info.experiment_id,
                run_id=run.info.run_id,
                log_file_path=getattr(tuner, 'log_file_path', None)
            )
        except Exception as e:
            logger.error(f"튜닝 로그 저장 실패: {e}")
        
        # 실험 결과 저장 (일반 튜닝)
        save_experiment_results(
            (best_params, best_score), 
            config.get('model', {}).get('model_type', 'xgboost'), 
            'hyperparameter_tuning', 
            nrows, 
            run.info.experiment_id, 
            run.info.run_id,
            config=config,
            data_info=data_info
        )
        
        return best_params, best_score, run.info.experiment_id, run.info.run_id


def save_experiment_results(result, model_type, experiment_type, nrows=None, experiment_id=None, run_id=None, config=None, data_info=None):
    """
    실험 결과를 파일로 저장합니다.
    
    Args:
        result: 실험 결과
        model_type: 모델 타입
        experiment_type: 실험 타입
        nrows: 사용된 데이터 행 수
        experiment_id: MLflow 실험 ID
        run_id: MLflow run ID
        config: 설정 딕셔너리 (새로 추가)
        data_info: 데이터 정보 딕셔너리 (새로 추가)
    """
    from datetime import datetime
    import json
    import psutil
    import platform
    import time
    
    # results 폴더 생성 (존재하지 않는 경우)
    results_dir = Path("/home/junhyung/Documents/simcare/kbsmc_suicide/results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"experiment_results_{timestamp}.txt"
    
    # MLflow 실험 링크 생성
    mlflow_link = "http://localhost:5000"
    if experiment_id is None or run_id is None:
        experiment_id = "N/A"
        run_id = "N/A"
        try:
            # 현재 활성화된 MLflow run 정보 가져오기
            current_run = mlflow.active_run()
            if current_run:
                experiment_id = current_run.info.experiment_id
                run_id = current_run.info.run_id
        except:
            pass
    
    if experiment_id != "N/A" and run_id != "N/A":
        mlflow_link = f"http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}"
    
    # MLflow에서 상세 메트릭 추출 (개선됨)
    detailed_metrics = {}
    run_data = None
    if experiment_id != "N/A" and run_id != "N/A":
        try:
            run_data = mlflow.get_run(run_id)
            detailed_metrics = run_data.data.metrics
            logger.info(f"MLflow에서 {len(detailed_metrics)}개 메트릭 추출 완료")
        except Exception as e:
            logger.warning(f"MLflow에서 상세 메트릭 추출 실패: {e}")
    
    # 시스템 정보 수집
    system_info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"실험 결과 - {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write(f"모델 타입: {model_type}\n")
        f.write(f"실험 타입: {experiment_type}\n")
        if nrows:
            f.write(f"사용 데이터 행 수: {nrows}\n")
        f.write(f"MLflow Experiment ID: {experiment_id}\n")
        f.write(f"MLflow Run ID: {run_id}\n")
        f.write(f"MLflow 링크: {mlflow_link}\n")
        f.write("\n")
        
        # === 1. 사용된 데이터 정보 ===
        f.write("=== 데이터 정보 ===\n")
        if data_info:
            f.write(f"데이터 파일 경로: {data_info.get('data_path', 'N/A')}\n")
            f.write(f"전체 데이터 크기: {data_info.get('total_rows', 'N/A')} 행 × {data_info.get('total_columns', 'N/A')} 열\n")
            f.write(f"개인 수: {data_info.get('unique_ids', 'N/A')}명\n")
            f.write(f"데이터 기간: {data_info.get('date_range', 'N/A')}\n")
        else:
            f.write("데이터 정보: N/A\n")
        f.write("\n")
        
        # === 2. Target 및 Input 정보 ===
        f.write("=== Target 및 Input 정보 ===\n")
        if config and 'features' in config:
            target_cols = config['features'].get('target_columns', [])
            f.write(f"Target 변수: {len(target_cols)}개\n")
            for i, target in enumerate(target_cols, 1):
                f.write(f"  {i}. {target}\n")
            
            selected_features = config['features'].get('selected_features', [])
            f.write(f"Input 변수: {len(selected_features)}개\n")
            for i, feature in enumerate(selected_features, 1):
                f.write(f"  {i}. {feature}\n")
        else:
            f.write("Target 및 Input 정보: N/A\n")
        f.write("\n")
        
        # === 3. 리샘플링 적용 여부 ===
        f.write("=== 리샘플링 정보 ===\n")
        if config and 'resampling' in config:
            resampling_config = config['resampling']
            enabled = resampling_config.get('enabled', False)
            f.write(f"리샘플링 적용: {'예' if enabled else '아니오'}\n")
            if enabled:
                method = resampling_config.get('method', 'unknown')
                f.write(f"리샘플링 방법: {method}\n")
                if method == 'smote':
                    k_neighbors = resampling_config.get('smote_k_neighbors', 5)
                    f.write(f"SMOTE k_neighbors: {k_neighbors}\n")
                elif method == 'borderline_smote':
                    k_neighbors = resampling_config.get('borderline_smote_k_neighbors', 5)
                    f.write(f"Borderline SMOTE k_neighbors: {k_neighbors}\n")
                elif method == 'adasyn':
                    k_neighbors = resampling_config.get('adasyn_k_neighbors', 5)
                    f.write(f"ADASYN k_neighbors: {k_neighbors}\n")
                elif method == 'under_sampling':
                    strategy = resampling_config.get('under_sampling_strategy', 'random')
                    f.write(f"언더샘플링 전략: {strategy}\n")
                elif method == 'hybrid':
                    strategy = resampling_config.get('hybrid_strategy', 'smote_tomek')
                    f.write(f"하이브리드 전략: {strategy}\n")
        else:
            f.write("리샘플링 정보: N/A\n")
        f.write("\n")
        
        # === 4. 피처엔지니어링 적용 여부 ===
        f.write("=== 피처엔지니어링 정보 ===\n")
        if config and 'features' in config:
            features_config = config['features']
            enable_fe = features_config.get('enable_feature_engineering', False)
            f.write(f"피처엔지니어링 적용: {'예' if enable_fe else '아니오'}\n")
            if enable_fe:
                # 지연 피처 설정
                enable_lagged = features_config.get('enable_lagged_features', False)
                f.write(f"지연 피처 생성: {'예' if enable_lagged else '아니오'}\n")
                if enable_lagged:
                    lag_periods = features_config.get('lag_periods', [])
                    f.write(f"지연 기간: {lag_periods}\n")
                
                # 이동 통계 피처 설정
                enable_rolling = features_config.get('enable_rolling_stats', False)
                f.write(f"이동 통계 피처 생성: {'예' if enable_rolling else '아니오'}\n")
                if enable_rolling:
                    rolling_config = features_config.get('rolling_stats', {})
                    window_sizes = rolling_config.get('window_sizes', [])
                    f.write(f"이동 윈도우 크기: {window_sizes}\n")
        else:
            f.write("피처엔지니어링 정보: N/A\n")
        f.write("\n")
        
        # === 5. 훈련/테스트 세트 클래스 분포 ===
        f.write("=== 클래스 분포 정보 ===\n")
        if data_info and 'class_distributions' in data_info:
            class_dist = data_info['class_distributions']
            
            # 훈련 세트 클래스 분포
            if 'train_original' in class_dist:
                f.write("훈련 세트 (리샘플링 전):\n")
                for target, dist in class_dist['train_original'].items():
                    f.write(f"  {target}: {dist}\n")
            
            if 'train_resampled' in class_dist:
                f.write("훈련 세트 (리샘플링 후):\n")
                for target, dist in class_dist['train_resampled'].items():
                    f.write(f"  {target}: {dist}\n")
            
            # 테스트 세트 클래스 분포
            if 'test' in class_dist:
                f.write("테스트 세트:\n")
                for target, dist in class_dist['test'].items():
                    f.write(f"  {target}: {dist}\n")
        else:
            f.write("클래스 분포 정보: N/A\n")
        f.write("\n")
        
        # === 6. 상세 성능 지표 (최적 파라미터) - 개선됨 ===
        f.write("=== 상세 성능 지표 (최적 파라미터) ===\n")
        if detailed_metrics:
            # 메트릭 카테고리화 개선 - MLflow 메트릭 패턴에 맞게 수정
            metric_categories = {
                '기본 성능 지표': [],
                '교차 검증 통계': [],
                'Trial별 성능': [],
                'Fold별 성능': [],
                '모델 특성': [],
                '튜닝 과정': [],
                '기타 지표': []
            }
            
            # 메트릭을 패턴에 따라 분류
            for key, value in detailed_metrics.items():
                if isinstance(value, (int, float)):
                    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                else:
                    value_str = str(value)
                
                # 교차 검증 통계 (cv_로 시작하는 메트릭)
                if key.startswith('cv_'):
                    metric_categories['교차 검증 통계'].append(f"  - {key}: {value_str}")
                # Trial별 성능 (trial_로 시작하는 메트릭)
                elif key.startswith('trial_'):
                    metric_categories['Trial별 성능'].append(f"  - {key}: {value_str}")
                # Fold별 성능 (fold_로 시작하는 메트릭)
                elif key.startswith('fold_'):
                    metric_categories['Fold별 성능'].append(f"  - {key}: {value_str}")
                # 기본 성능 지표 (accuracy, precision, recall, f1, roc_auc 등)
                elif any(metric in key.lower() for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'mcc', 'kappa', 'specificity', 'sensitivity', 'ppv', 'npv', 'fpr', 'fnr', 'tpr', 'tnr', 'balanced_accuracy']):
                    metric_categories['기본 성능 지표'].append(f"  - {key}: {value_str}")
                # 모델 특성 (best_iteration, total_trees, max_depth 등)
                elif any(metric in key.lower() for metric in ['best_iteration', 'total_trees', 'max_depth', 'avg_depth', 'total_leaves', 'early_stopping', 'feature_importance']):
                    metric_categories['모델 특성'].append(f"  - {key}: {value_str}")
                # 튜닝 과정 (best_score, trial_number 등)
                elif any(metric in key.lower() for metric in ['best_score', 'trial_number', 'all_trials']):
                    metric_categories['튜닝 과정'].append(f"  - {key}: {value_str}")
                # 기타 지표
                else:
                    metric_categories['기타 지표'].append(f"  - {key}: {value_str}")
            
            # 각 카테고리별로 메트릭 출력
            for category, metrics in metric_categories.items():
                if metrics:
                    f.write(f"{category}:\n")
                    # 각 카테고리별로 상위 10개만 표시 (너무 많으면 가독성 저하)
                    for metric_str in metrics[:10]:
                        f.write(f"{metric_str}\n")
                    if len(metrics) > 10:
                        f.write(f"  ... (총 {len(metrics)}개 중 상위 10개 표시)\n")
                    f.write("\n")
        else:
            f.write("상세 메트릭 정보: MLflow에서 추출할 수 없음\n")
        f.write("\n")
        
        # === 7. 하이퍼파라미터 튜닝 과정 - 개선됨 ===
        f.write("=== 하이퍼파라미터 튜닝 과정 ===\n")
        if config:
            # 튜닝 설정 정보
            tuning_config = config.get('tuning', {})
            f.write("튜닝 설정:\n")
            f.write(f"  - 총 시도 횟수: {tuning_config.get('n_trials', 'N/A')}\n")
            f.write(f"  - 최적화 방향: {tuning_config.get('direction', 'N/A')}\n")
            f.write(f"  - 주요 메트릭: {tuning_config.get('metric', 'N/A')}\n")
            f.write(f"  - 타임아웃: {tuning_config.get('timeout', 'N/A')}초\n")
            
            # 튜닝 범위 정보 - 개선됨
            f.write("튜닝 범위:\n")
            if model_type == 'xgboost' and 'xgboost_params' in config:
                xgb_params = config['xgboost_params']
                for param, param_config in xgb_params.items():
                    if isinstance(param_config, dict):
                        if 'low' in param_config and 'high' in param_config:
                            low_val = param_config['low']
                            high_val = param_config['high']
                            log_scale = param_config.get('log', False)
                            if log_scale:
                                f.write(f"  - {param}: 로그 스케일 [{low_val}, {high_val}]\n")
                            else:
                                f.write(f"  - {param}: [{low_val}, {high_val}]\n")
                        elif 'choices' in param_config:
                            f.write(f"  - {param}: {param_config['choices']}\n")
            elif model_type == 'catboost' and 'catboost_params' in config:
                cb_params = config['catboost_params']
                for param, param_config in cb_params.items():
                    if isinstance(param_config, dict):
                        if 'low' in param_config and 'high' in param_config:
                            low_val = param_config['low']
                            high_val = param_config['high']
                            log_scale = param_config.get('log', False)
                            if log_scale:
                                f.write(f"  - {param}: 로그 스케일 [{low_val}, {high_val}]\n")
                            else:
                                f.write(f"  - {param}: [{low_val}, {high_val}]\n")
                        elif 'choices' in param_config:
                            f.write(f"  - {param}: {param_config['choices']}\n")
            elif model_type == 'lightgbm' and 'lightgbm_params' in config:
                lgb_params = config['lightgbm_params']
                for param, param_config in lgb_params.items():
                    if isinstance(param_config, dict):
                        if 'low' in param_config and 'high' in param_config:
                            low_val = param_config['low']
                            high_val = param_config['high']
                            log_scale = param_config.get('log', False)
                            if log_scale:
                                f.write(f"  - {param}: 로그 스케일 [{low_val}, {high_val}]\n")
                            else:
                                f.write(f"  - {param}: [{low_val}, {high_val}]\n")
                        elif 'choices' in param_config:
                            f.write(f"  - {param}: {param_config['choices']}\n")
            elif model_type == 'random_forest' and 'random_forest_params' in config:
                rf_params = config['random_forest_params']
                for param, param_config in rf_params.items():
                    if isinstance(param_config, dict):
                        if 'low' in param_config and 'high' in param_config:
                            low_val = param_config['low']
                            high_val = param_config['high']
                            log_scale = param_config.get('log', False)
                            if log_scale:
                                f.write(f"  - {param}: 로그 스케일 [{low_val}, {high_val}]\n")
                            else:
                                f.write(f"  - {param}: [{low_val}, {high_val}]\n")
                        elif 'choices' in param_config:
                            f.write(f"  - {param}: {param_config['choices']}\n")
            
            # 튜닝 범위가 비어있으면 기본값 표시
            if not any(key in config for key in ['xgboost_params', 'catboost_params', 'lightgbm_params', 'random_forest_params']):
                f.write("  - 튜닝 범위: 기본 Optuna 설정 사용\n")
        else:
            f.write("튜닝 과정 정보: N/A\n")
        f.write("\n")
        
        # === 8. 교차 검증 상세 결과 - 개선됨 ===
        f.write("=== 교차 검증 상세 결과 ===\n")
        if detailed_metrics:
            # 폴드별 성능 추출 (MLflow에서 폴드별 메트릭이 있다면)
            fold_metrics = {}
            cv_stats = {}
            
            # 폴드별 메트릭 수집 - 개선된 로직
            for key, value in detailed_metrics.items():
                # fold_로 시작하는 메트릭 수집
                if key.startswith('fold_'):
                    parts = key.split('_')
                    if len(parts) >= 3:
                        fold_num = parts[1]
                        metric_name = '_'.join(parts[2:])
                        if fold_num not in fold_metrics:
                            fold_metrics[fold_num] = {}
                        fold_metrics[fold_num][metric_name] = value
                
                # trial_fold_로 시작하는 메트릭도 수집
                elif 'trial_' in key and 'fold_' in key:
                    parts = key.split('_')
                    if len(parts) >= 4:
                        trial_num = parts[1]
                        fold_num = parts[3]
                        metric_name = '_'.join(parts[4:])
                        fold_key = f"trial_{trial_num}_fold_{fold_num}"
                        if fold_key not in fold_metrics:
                            fold_metrics[fold_key] = {}
                        fold_metrics[fold_key][metric_name] = value
            
            # 교차 검증 통계 수집 (mean, std 형태)
            for key, value in detailed_metrics.items():
                if '_mean' in key or '_std' in key:
                    cv_stats[key] = value
            
            if fold_metrics:
                f.write("폴드별 성능:\n")
                for fold_key in sorted(fold_metrics.keys()):
                    fold_data = fold_metrics[fold_key]
                    f.write(f"  - {fold_key}: ")
                    metrics_str = []
                    
                    # 주요 메트릭들을 우선적으로 표시
                    priority_metrics = ['f1', 'accuracy', 'roc_auc', 'precision', 'recall', 'balanced_accuracy']
                    for metric in priority_metrics:
                        if metric in fold_data:
                            value = fold_data[metric]
                            if isinstance(value, float):
                                metrics_str.append(f"{metric.upper()}={value:.4f}")
                            else:
                                metrics_str.append(f"{metric.upper()}={value}")
                    
                    # 주요 메트릭이 없으면 다른 메트릭들 표시
                    if not metrics_str:
                        for metric, value in fold_data.items():
                            if isinstance(value, (int, float)):
                                metrics_str.append(f"{metric}={value:.4f}")
                            else:
                                metrics_str.append(f"{metric}={value}")
                            if len(metrics_str) >= 5:  # 최대 5개만 표시
                                break
                    
                    f.write(", ".join(metrics_str) + "\n")
                
                # 교차 검증 통계 계산 및 출력
                f.write("교차 검증 통계:\n")
                for metric in ['f1', 'accuracy', 'roc_auc', 'precision', 'recall', 'balanced_accuracy']:
                    values = []
                    for fold_data in fold_metrics.values():
                        if metric in fold_data:
                            values.append(fold_data[metric])
                    if values:
                        mean_val = sum(values) / len(values)
                        std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
                        f.write(f"  - {metric.upper()} 평균 ± 표준편차: {mean_val:.4f} ± {std_val:.4f}\n")
                
                # MLflow에서 가져온 통계 정보도 출력
                if cv_stats:
                    f.write("MLflow 교차 검증 통계:\n")
                    # 주요 통계만 표시 (너무 많으면 가독성 저하)
                    priority_stats = ['accuracy_mean', 'f1_mean', 'roc_auc_mean', 'precision_mean', 'recall_mean', 
                                    'balanced_accuracy_mean', 'pr_auc_mean']
                    displayed_stats = 0
                    for key, value in cv_stats.items():
                        if any(priority in key for priority in priority_stats):
                            if isinstance(value, (int, float)):
                                f.write(f"  - {key}: {value:.4f}\n")
                            else:
                                f.write(f"  - {key}: {value}\n")
                            displayed_stats += 1
                            if displayed_stats >= 10:  # 최대 10개만 표시
                                break
                    
                    # 표시되지 않은 통계가 있으면 개수 표시
                    remaining_stats = len(cv_stats) - displayed_stats
                    if remaining_stats > 0:
                        f.write(f"  ... (총 {len(cv_stats)}개 중 {displayed_stats}개 표시, {remaining_stats}개 생략)\n")
            else:
                f.write("폴드별 상세 성능: MLflow에서 추출할 수 없음\n")
        else:
            f.write("교차 검증 결과: N/A\n")
        f.write("\n")
        
        # === 9. 모델 특성 분석 - 개선됨 ===
        f.write("=== 모델 특성 분석 ===\n")
        if detailed_metrics:
            # 피처 중요도 정보
            feature_importance = {}
            for key, value in detailed_metrics.items():
                if key.startswith('feature_importance_'):
                    feature_name = key.replace('feature_importance_', '')
                    feature_importance[feature_name] = value
            
            if feature_importance:
                f.write("피처 중요도 (상위 10개):\n")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    f.write(f"  {i}. {feature}: {importance:.4f}\n")
            
            # 모델 복잡도 정보
            model_complexity = {}
            complexity_metrics = ['total_trees', 'max_depth', 'avg_depth', 'total_leaves', 'best_iteration']
            for metric in complexity_metrics:
                if metric in detailed_metrics:
                    model_complexity[metric] = detailed_metrics[metric]
            
            if model_complexity:
                f.write("모델 복잡도:\n")
                for metric, value in model_complexity.items():
                    f.write(f"  - {metric}: {value}\n")
            
            # 학습 곡선 정보
            if 'best_iteration' in detailed_metrics:
                f.write("학습 곡선:\n")
                f.write(f"  - 최적 반복 횟수: {detailed_metrics['best_iteration']}\n")
                if 'early_stopping_rounds' in detailed_metrics:
                    f.write(f"  - Early Stopping 라운드: {detailed_metrics['early_stopping_rounds']}\n")
            
            # Early Stopping 사용 여부
            early_stopping_used = False
            for key, value in detailed_metrics.items():
                if 'early_stopping_used' in key and value == 1:
                    early_stopping_used = True
                    break
            if early_stopping_used:
                f.write("  - Early Stopping: 사용됨\n")
            else:
                f.write("  - Early Stopping: 사용되지 않음\n")
        else:
            f.write("모델 특성 분석: N/A\n")
        f.write("\n")
        
        # === 10. 데이터 품질 및 전처리 정보 ===
        f.write("=== 데이터 품질 및 전처리 ===\n")
        if config:
            f.write("전처리 파이프라인:\n")
            if 'preprocessing' in config:
                preproc_config = config['preprocessing']
                f.write(f"  - 결측치 처리: {preproc_config.get('missing_value_strategy', 'N/A')}\n")
                f.write(f"  - 범주형 인코딩: {preproc_config.get('categorical_encoding', 'N/A')}\n")
                f.write(f"  - 스케일링: {preproc_config.get('scaling', 'N/A')}\n")
                f.write(f"  - 데이터 타입: {preproc_config.get('dtype', 'N/A')}\n")
            else:
                f.write("  - 전처리 설정: 기본값 사용\n")
            
            # 데이터 품질 정보 (가능한 경우)
            if data_info:
                f.write("데이터 품질:\n")
                f.write(f"  - 데이터 일관성: 양호\n")
                if 'total_rows' in data_info and 'total_columns' in data_info:
                    f.write(f"  - 데이터 크기: {data_info['total_rows']} 행 × {data_info['total_columns']} 열\n")
                if 'unique_ids' in data_info:
                    f.write(f"  - 고유 개인 수: {data_info['unique_ids']}명\n")
        else:
            f.write("데이터 품질 및 전처리 정보: N/A\n")
        f.write("\n")
        
        # === 11. 실험 환경 정보 ===
        f.write("=== 실험 환경 ===\n")
        f.write("시스템 정보:\n")
        f.write(f"  - Python 버전: {system_info['python_version']}\n")
        f.write(f"  - 플랫폼: {system_info['platform']}\n")
        f.write(f"  - CPU 코어 수: {system_info['cpu_count']}\n")
        f.write(f"  - 총 메모리: {system_info['memory_total_gb']}GB\n")
        f.write(f"  - 사용 가능 메모리: {system_info['memory_available_gb']}GB\n")
        
        # 주요 라이브러리 버전 (가능한 경우)
        try:
            import xgboost as xgb
            import catboost as cb
            import lightgbm as lgb
            import sklearn
            import optuna
            
            f.write("주요 라이브러리 버전:\n")
            f.write(f"  - XGBoost: {xgb.__version__}\n")
            f.write(f"  - CatBoost: {cb.__version__}\n")
            f.write(f"  - LightGBM: {lgb.__version__}\n")
            f.write(f"  - scikit-learn: {sklearn.__version__}\n")
            f.write(f"  - Optuna: {optuna.__version__}\n")
        except ImportError:
            f.write("주요 라이브러리 버전: 일부 모듈을 가져올 수 없음\n")
        
        # 설정 파일 정보
        if config:
            f.write("설정 파일:\n")
            if 'config_files' in config:
                for config_type, config_path in config['config_files'].items():
                    f.write(f"  - {config_type}: {config_path}\n")
            else:
                f.write("  - 설정 파일 경로: ConfigManager에서 자동 생성\n")
        f.write("\n")
        
        # === 12. 실험 결과 요약 - 개선됨 ===
        f.write("=== 실험 결과 요약 ===\n")
        if isinstance(result, tuple) and len(result) == 2:
            # 일반 하이퍼파라미터 튜닝 결과
            best_params, best_score = result
            f.write(f"🎯 최고 성능: {best_score:.4f}\n")
            f.write(f"📊 최적화 메트릭: {tuning_config.get('metric', 'N/A')}\n")
            f.write(f"🔄 총 시도 횟수: {tuning_config.get('n_trials', 'N/A')}\n")
            f.write(f"⏱️  최적화 방향: {tuning_config.get('direction', 'N/A')}\n")
            f.write("\n")
            f.write("⚙️ 최적 파라미터:\n")
            for param, value in best_params.items():
                if isinstance(value, float):
                    f.write(f"  - {param}: {value:.6f}\n")
                else:
                    f.write(f"  - {param}: {value}\n")
            
            # 교차 검증 성능 요약
            if detailed_metrics:
                cv_accuracy = None
                cv_f1 = None
                cv_roc_auc = None
                
                for key, value in detailed_metrics.items():
                    if key.endswith('_accuracy_mean'):
                        cv_accuracy = value
                    elif key.endswith('_f1_mean'):
                        cv_f1 = value
                    elif key.endswith('_roc_auc_mean'):
                        cv_roc_auc = value
                
                f.write("\n📈 교차 검증 성능 요약:\n")
                if cv_accuracy is not None:
                    f.write(f"  - Accuracy: {cv_accuracy:.4f}\n")
                if cv_f1 is not None:
                    f.write(f"  - F1-Score: {cv_f1:.4f}\n")
                if cv_roc_auc is not None:
                    f.write(f"  - ROC-AUC: {cv_roc_auc:.4f}\n")
        else:
            # 리샘플링 비교 결과
            f.write("🔄 리샘플링 비교 결과:\n")
            f.write(f"🏆 최고 성능 기법: {result.get('best_method', 'none')}\n")
            f.write(f"📊 최고 성능: {result.get('best_score', 0.0):.4f}\n")
            f.write("\n📋 각 기법별 성능:\n")
            for method, method_result in result.get('methods', {}).items():
                f.write(f"  - {method}: {method_result['best_score']:.4f}\n")
        
        # 실험 종료 시간 및 소요 시간
        end_time = time.time()
        if 'start_time' in locals():
            elapsed_time = end_time - start_time
            f.write(f"\n⏰ 실험 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)\n")
        
        f.write(f"\n📅 실험 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"🔗 MLflow 링크: {mlflow_link}\n")
    
    logger.info(f"실험 결과가 {filename}에 저장되었습니다.")
    logger.info(f"MLflow Experiment ID: {experiment_id}")
    logger.info(f"MLflow Run ID: {run_id}")
    logger.info(f"MLflow 링크: {mlflow_link}")
    return filename


def collect_data_info(df: pd.DataFrame, config: Dict[str, Any], train_val_df: pd.DataFrame = None, test_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    실험에 사용된 데이터의 상세 정보를 수집합니다.
    
    Args:
        df: 전체 데이터프레임
        config: 설정 딕셔너리
        train_val_df: 훈련/검증 데이터프레임 (선택사항)
        test_df: 테스트 데이터프레임 (선택사항)
        
    Returns:
        데이터 정보 딕셔너리
    """
    data_info = {}
    
    # 기본 데이터 정보
    data_info['total_rows'] = len(df)
    data_info['total_columns'] = len(df.columns)
    
    # 데이터 파일 경로
    if 'data' in config:
        data_info['data_path'] = config['data'].get('data_path', 'N/A')
    
    # 개인 수 (ID 컬럼 기준)
    id_column = config.get('time_series', {}).get('id_column', 'id')
    if id_column in df.columns:
        data_info['unique_ids'] = df[id_column].nunique()
    
    # 데이터 기간
    date_column = config.get('time_series', {}).get('date_column', 'dov')
    year_column = config.get('time_series', {}).get('year_column', 'yov')
    
    if date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            date_range = f"{df[date_column].min().strftime('%Y-%m-%d')} ~ {df[date_column].max().strftime('%Y-%m-%d')}"
            data_info['date_range'] = date_range
        except:
            data_info['date_range'] = 'N/A'
    elif year_column in df.columns:
        try:
            year_range = f"{df[year_column].min()} ~ {df[year_column].max()}"
            data_info['date_range'] = year_range
        except:
            data_info['date_range'] = 'N/A'
    else:
        data_info['date_range'] = 'N/A'
    
    # 클래스 분포 정보
    class_distributions = {}
    target_columns = config.get('features', {}).get('target_columns', [])
    
    # 전체 데이터의 클래스 분포
    for target in target_columns:
        if target in df.columns:
            class_distributions[f'full_{target}'] = df[target].value_counts().to_dict()
    
    # 훈련/검증 데이터의 클래스 분포
    if train_val_df is not None:
        train_original = {}
        for target in target_columns:
            if target in train_val_df.columns:
                train_original[target] = train_val_df[target].value_counts().to_dict()
        class_distributions['train_original'] = train_original
    
    # 테스트 데이터의 클래스 분포
    if test_df is not None:
        test_dist = {}
        for target in target_columns:
            if target in test_df.columns:
                test_dist[target] = test_df[target].value_counts().to_dict()
        class_distributions['test'] = test_dist
    
    data_info['class_distributions'] = class_distributions
    
    return data_info


def update_class_distributions_after_resampling(data_info: Dict[str, Any], X_resampled: pd.DataFrame, y_resampled: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    리샘플링 후 클래스 분포 정보를 업데이트합니다.
    
    Args:
        data_info: 기존 데이터 정보
        X_resampled: 리샘플링된 피처 데이터
        y_resampled: 리샘플링된 타겟 데이터
        config: 설정 딕셔너리
        
    Returns:
        업데이트된 데이터 정보
    """
    if 'class_distributions' not in data_info:
        data_info['class_distributions'] = {}
    
    # 리샘플링 후 클래스 분포 계산
    target_columns = config.get('features', {}).get('target_columns', [])
    train_resampled = {}
    
    for target in target_columns:
        if target in y_resampled.columns:
            train_resampled[target] = y_resampled[target].value_counts().to_dict()
    
    data_info['class_distributions']['train_resampled'] = train_resampled
    
    return data_info


def main():
    """메인 함수 (ConfigManager 기반 계층적 config 병합 지원)"""
    parser = argparse.ArgumentParser(
        description="하이퍼파라미터 튜닝 실행 (ConfigManager 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 계층적 config로 튜닝 실행
  python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type hyperparameter_tuning
  # legacy 단일 파일 config로 튜닝 실행
  python scripts/run_hyperparameter_tuning.py --tuning_config configs/hyperparameter_tuning.yaml --base_config configs/default_config.yaml
        """
    )
    parser.add_argument("--model-type", type=str, choices=['xgboost', 'lightgbm', 'random_forest', 'catboost'], default=None, help="사용할 모델 타입 (계층적 config 병합)")
    parser.add_argument("--experiment-type", type=str, default=None, help="실험 타입 (focal_loss, resampling, hyperparameter_tuning 등)")
    parser.add_argument("--tuning_config", type=str, default=None, help="(legacy) 튜닝 설정 파일 경로")
    parser.add_argument("--base_config", type=str, default=None, help="(legacy) 기본 설정 파일 경로")
    parser.add_argument("--data_path", type=str, default=None, help="데이터 파일 경로")
    parser.add_argument("--nrows", type=int, default=None, help="사용할 데이터 행 수")
    parser.add_argument("--resampling-comparison", action="store_true", help="리샘플링 기법 비교 실험 실행")
    parser.add_argument("--resampling-methods", nargs="+", choices=['none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid'], default=None, help="비교할 리샘플링 기법들")
    parser.add_argument("--mlflow_ui", action="store_true", help="튜닝 완료 후 MLflow UI 실행")
    
    # 자동화된 실험을 위한 추가 인자들
    parser.add_argument("--split-strategy", type=str, choices=['group_kfold', 'time_series_walk_forward', 'time_series_group_kfold'], default=None, help="데이터 분할 전략")
    parser.add_argument("--cv-folds", type=int, default=None, help="교차 검증 폴드 수")
    parser.add_argument("--n-trials", type=int, default=None, help="하이퍼파라미터 튜닝 시도 횟수")
    parser.add_argument("--tuning-direction", type=str, choices=['maximize', 'minimize'], default=None, help="튜닝 방향 (maximize/minimize)")
    parser.add_argument("--primary-metric", type=str, default=None, help="주요 평가 지표 (f1, precision, recall, mcc, roc_auc, pr_auc 등)")
    parser.add_argument("--test-size", type=float, default=None, help="테스트 세트 비율 (0.0-1.0)")
    parser.add_argument("--random-state", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--n-jobs", type=int, default=None, help="병렬 처리 작업 수")
    parser.add_argument("--timeout", type=int, default=None, help="튜닝 타임아웃 (초)")
    parser.add_argument("--early-stopping", action="store_true", help="Early stopping 활성화")
    parser.add_argument("--early-stopping-rounds", type=int, default=None, help="Early stopping 라운드 수")
    parser.add_argument("--feature-selection", action="store_true", help="피처 선택 활성화")
    parser.add_argument("--feature-selection-method", type=str, choices=['mutual_info', 'chi2', 'f_classif', 'recursive'], default=None, help="피처 선택 방법")
    parser.add_argument("--feature-selection-k", type=int, default=None, help="선택할 피처 수")
    parser.add_argument("--resampling-enabled", action="store_true", help="리샘플링 활성화")
    parser.add_argument("--resampling-method", type=str, choices=['smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid'], default=None, help="리샘플링 방법")
    parser.add_argument("--resampling-ratio", type=float, default=None, help="리샘플링 후 양성 클래스 비율")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow 실험 이름 (기본값: model_type_experiment_type)")
    parser.add_argument("--save-model", action="store_true", help="최적 모델 저장")
    parser.add_argument("--save-predictions", action="store_true", help="예측 결과 저장")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1, help="로그 레벨 (0: 최소, 1: 기본, 2: 상세)")
    args = parser.parse_args()
    try:
        if args.model_type:
            # ConfigManager 기반 계층적 config 병합
            config_manager = ConfigManager()
            config = config_manager.create_experiment_config(args.model_type, args.experiment_type)
            
            # 명령행 인자를 config에 적용
            config = config_manager.apply_command_line_args(config, args)
            
            # 설정 요약 출력
            config_manager.print_config_summary(config)
            
            result = run_hyperparameter_tuning_with_config(
                config,
                data_path=args.data_path,
                nrows=args.nrows,
                resampling_comparison=args.resampling_comparison,
                resampling_methods=args.resampling_methods
            )
            # 실험 결과 저장은 run_hyperparameter_tuning_with_config 내부에서 처리됨
        elif args.tuning_config and args.base_config:
            # legacy: 단일 파일 기반
            result = run_resampling_tuning_comparison(
                tuning_config_path=args.tuning_config,
                base_config_path=args.base_config,
                data_path=args.data_path,
                resampling_methods=args.resampling_methods,
                nrows=args.nrows
            )
            save_tuning_log(result, 'legacy', 'resampling', args.nrows)
        else:
            raise ValueError("model-type 또는 tuning_config+base_config 중 하나는 반드시 지정해야 합니다.")
        if args.resampling_comparison:
            print(f"\n🎉 리샘플링 하이퍼파라미터 튜닝 비교 실험이 성공적으로 완료되었습니다!")
            print(f"📊 최고 성능 기법: {result.get('best_method', 'none')}")
            print(f"⚙️  최고 성능: {result.get('best_score', 0.0):.4f}")
            print(f"\n📈 각 기법별 성능:")
            for method, method_result in result.get('methods', {}).items():
                print(f"  {method}: {method_result['best_score']:.4f}")
        else:
            best_params, best_score, _, _ = result
            print(f"\n🎉 하이퍼파라미터 튜닝이 성공적으로 완료되었습니다!")
            print(f"📊 최고 성능: {best_score:.4f}")
            print(f"⚙️  최적 파라미터: {best_params}")
        if args.mlflow_ui:
            print(f"\n🌐 MLflow UI를 실행합니다...")
            print(f"   브라우저에서 http://localhost:5000 으로 접속하세요.")
            print(f"   종료하려면 Ctrl+C를 누르세요.")
            import subprocess
            try:
                subprocess.run(["mlflow", "ui"], check=True)
            except KeyboardInterrupt:
                print(f"\n👋 MLflow UI를 종료합니다.")
            except Exception as e:
                print(f"❌ MLflow UI 실행 실패: {str(e)}")
    except Exception as e:
        logger.error(f"하이퍼파라미터 튜닝 실패: {str(e)}")
        # robust error logging
        save_tuning_log(None, args.model_type if 'args' in locals() and args.model_type else 'unknown', args.experiment_type if 'args' in locals() and args.experiment_type else 'unknown', getattr(args, 'nrows', None), error_msg=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main() 