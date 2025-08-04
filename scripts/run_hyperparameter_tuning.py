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
import os
from datetime import datetime
from pathlib import Path
import mlflow
import yaml
import pandas as pd
import numpy as np
import optuna
from typing import Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.hyperparameter_tuning import HyperparameterTuner
from src.splits import (
    load_config, 
    split_test_set, 
    validate_splits, 
    log_splits_info
)
from src.utils.config_manager import ConfigManager
from src.utils import setup_logging, setup_experiment_logging, log_experiment_summary, experiment_logging_context
from src.utils.experiment_results import save_experiment_results

# 로깅 설정
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def setup_mlflow_experiment(experiment_name: str):
    """
    MLflow 실험을 설정합니다.
    
    Args:
        experiment_name: 실험 이름
        
    Returns:
        experiment_id: 실험 ID
    """
    from src.utils.mlflow_manager import setup_mlflow_experiment_safely
    
    experiment_id = setup_mlflow_experiment_safely(experiment_name)
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
        
        # 기본 설정에서 리샘플링 파라미터 추출 (개선됨)
        resampling_config = base_config.get('resampling', {})
        resampling_params = {
            "resampling_enabled": resampling_config.get('enabled', False),
            "resampling_method": resampling_config.get('method', 'none'),
            "resampling_random_state": resampling_config.get('random_state', 42)
        }
        
        # 리샘플링 기법별 상세 파라미터 추가
        method = resampling_config.get('method', 'none')
        if method == 'smote':
            resampling_params.update({
                "smote_k_neighbors": resampling_config.get('smote_k_neighbors', 5),
                "smote_sampling_strategy": resampling_config.get('smote_sampling_strategy', 0.1)
            })
        elif method == 'borderline_smote':
            resampling_params.update({
                "borderline_smote_k_neighbors": resampling_config.get('borderline_smote_k_neighbors', 5),
                "borderline_smote_sampling_strategy": resampling_config.get('borderline_smote_sampling_strategy', 0.1),
                "borderline_smote_m_neighbors": resampling_config.get('borderline_smote_m_neighbors', 10)
            })
        elif method == 'adasyn':
            resampling_params.update({
                "adasyn_k_neighbors": resampling_config.get('adasyn_k_neighbors', 5),
                "adasyn_sampling_strategy": resampling_config.get('adasyn_sampling_strategy', 0.1)
            })
        elif method == 'time_series_adapted':
            time_series_config = resampling_config.get('time_series_adapted', {})
            resampling_params.update({
                "time_weight": time_series_config.get('time_weight', 0.3),
                "temporal_window": time_series_config.get('temporal_window', 3),
                "seasonality_weight": time_series_config.get('seasonality_weight', 0.2),
                "pattern_preservation": time_series_config.get('pattern_preservation', True),
                "trend_preservation": time_series_config.get('trend_preservation', True)
            })
        
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





def run_hyperparameter_tuning_with_config(config: Dict[str, Any], data_path: str = None, nrows: int = None):
    """
    설정 기반 하이퍼파라미터 튜닝을 실행합니다.
    
    Args:
        config: 설정 딕셔너리
        data_path: 데이터 파일 경로
        nrows: 사용할 데이터 행 수
        resampling_comparison: 리샘플링 비교 실험 여부 (사용하지 않음)
        resampling_methods: 리샘플링 방법들 (사용하지 않음)
        
    Returns:
        튜닝 결과 튜플 (best_params, best_score, study, tuner)
    """
    import time
    start_time = time.time()
    
    # 실험 정보 추출
    model_type = config.get('model', {}).get('model_type', 'unknown')
    experiment_type = config.get('experiment_type', 'hyperparameter_tuning')
    
    # 새로운 로깅 시스템 적용
    with experiment_logging_context(
        experiment_type=experiment_type,
        model_type=model_type,
        log_level="INFO",
        capture_console=True
    ) as log_file_path:
        
        logger.info(f"=== 하이퍼파라미터 튜닝 시작 ===")
        logger.info(f"모델 타입: {model_type}")
        logger.info(f"실험 타입: {experiment_type}")
        logger.info(f"로그 파일: {log_file_path}")
        
        try:
            # MLflow 실험 무결성 검증 및 복구
            from src.utils.mlflow_manager import MLflowExperimentManager
            manager = MLflowExperimentManager()
            
            # 현재 실험 상태 확인
            logger.info("MLflow 실험 상태 확인 중...")
            manager.print_experiment_summary()
            
            # Orphaned 실험 정리
            logger.info("Orphaned 실험 정리 중...")
            deleted_experiments = manager.cleanup_orphaned_experiments(backup=True)
            if deleted_experiments:
                logger.info(f"정리된 orphaned 실험: {deleted_experiments}")
            
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
            
            # MLflow 실험 설정
            experiment_name = config['mlflow']['experiment_name']
            experiment_id = setup_mlflow_experiment(experiment_name)
            
            # 실험 무결성 검증
            if not manager.validate_experiment_integrity(experiment_id):
                logger.warning(f"실험 {experiment_id} 무결성 검증 실패, 복구 시도 중...")
                if manager.repair_experiment(experiment_id):
                    logger.info(f"실험 {experiment_id} 복구 완료")
                else:
                    logger.error(f"실험 {experiment_id} 복구 실패")
            
            # 튜너 생성
            tuner = HyperparameterTuner(config, train_val_df, nrows=nrows)
            
            # 튜닝 실행 (안전한 MLflow run 관리)
            from src.utils.mlflow_manager import safe_mlflow_run
            
            with safe_mlflow_run(experiment_id=experiment_id) as run:
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
                
                # === 고급 로깅 기능 추가 ===
                from src.utils.mlflow_logging import log_all_advanced_metrics
                
                # 모든 고급 로깅 기능 실행
                logging_results = log_all_advanced_metrics(tuner, config, run)
                
                # 로깅 결과 요약
                successful_features = [k for k, v in logging_results.items() if v]
                failed_features = [k for k, v in logging_results.items() if not v]
                
                if successful_features:
                    logger.info(f"고급 로깅 성공: {successful_features}")
                if failed_features:
                    logger.warning(f"고급 로깅 실패: {failed_features}")
                
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
                
                # 실행 시간 계산
                execution_time = time.time() - start_time
                
                # 실험 요약 로깅
                n_trials = len(tuner.study.trials) if hasattr(tuner, 'study') and tuner.study else 0
                log_experiment_summary(
                    experiment_type=experiment_type,
                    model_type=model_type,
                    best_score=best_score,
                    best_params=best_params,
                    execution_time=execution_time,
                    n_trials=n_trials,
                    data_info=data_info,
                    log_file_path=log_file_path
                )
                
                logger.info(f"=== 하이퍼파라미터 튜닝 완료 ===")
                logger.info(f"최고 성능: {best_score:.6f}")
                logger.info(f"실행 시간: {execution_time:.2f}초")
                logger.info(f"시도 횟수: {n_trials}")
                
                # 실험 결과 저장
                try:
                    save_experiment_results(
                        result=(best_params, best_score),
                        model_type=model_type,
                        experiment_type=experiment_type,
                        nrows=nrows,
                        experiment_id=run.info.experiment_id,
                        run_id=run.info.run_id,
                        config=config,
                        data_info=data_info
                    )
                    logger.info("실험 결과 저장 완료")
                except Exception as e:
                    logger.error(f"실험 결과 저장 실패: {e}")
                
                return best_params, best_score, run.info.experiment_id, run.info.run_id
                
        except Exception as e:
            logger.error(f"하이퍼파라미터 튜닝 중 오류 발생: {e}")
            raise


def save_tuning_log(result, model_type, experiment_type, nrows=None, experiment_id=None, run_id=None, log_file_path=None):
    """
    튜닝 로그를 파일로 저장합니다.
    
    Args:
        result: 튜닝 결과 (best_params, best_score) 또는 튜플
        model_type: 모델 타입
        experiment_type: 실험 타입
        nrows: 사용한 데이터 행 수
        experiment_id: MLflow 실험 ID
        run_id: MLflow Run ID
        log_file_path: 로그 파일 경로 (None이면 기본 경로 사용)
    """
    try:
        # 결과 파싱
        if isinstance(result, tuple) and len(result) >= 2:
            best_params = result[0] if result[0] is not None else {}
            best_score = result[1] if result[1] is not None else 0.0
        elif isinstance(result, dict):
            best_params = result.get('best_params', {})
            best_score = result.get('best_score', 0.0)
        else:
            best_params = {}
            best_score = 0.0
        
        # 로그 파일 경로 설정
        if log_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = f"logs/tuning_log_{model_type}_{experiment_type}_{timestamp}.txt"
        
        # 로그 디렉토리 생성
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # 로그 내용 작성
        log_content = f"""
=== 튜닝 로그 ===
모델 타입: {model_type}
실험 타입: {experiment_type}
데이터 행 수: {nrows}
MLflow 실험 ID: {experiment_id}
MLflow Run ID: {run_id}
최고 성능: {best_score:.6f}
최적 파라미터:
"""
        
        for param, value in best_params.items():
            log_content += f"  {param}: {value}\n"
        
        log_content += f"\n생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # 파일 저장
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        logger.info(f"튜닝 로그 저장 완료: {log_file_path}")
        
    except Exception as e:
        logger.error(f"튜닝 로그 저장 실패: {e}")





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
                nrows=args.nrows
            )
            # 실험 결과 저장은 run_hyperparameter_tuning_with_config 내부에서 처리됨
        elif args.tuning_config and args.base_config:
            # legacy: 단일 파일 기반 (리샘플링 기능 제거됨)
            raise NotImplementedError("Legacy 모드는 지원하지 않습니다. ConfigManager를 사용해주세요.")
        else:
            raise ValueError("model-type 또는 tuning_config+base_config 중 하나는 반드시 지정해야 합니다.")
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
        # robust error logging (MLflow를 통해 자동으로 로깅됨)
        logger.error(f"실험 실패 - 모델: {args.model_type if 'args' in locals() and args.model_type else 'unknown'}, 실험: {args.experiment_type if 'args' in locals() and args.experiment_type else 'unknown'}")
        sys.exit(1)


if __name__ == "__main__":
    main() 