#!/usr/bin/env python3
"""
리샘플링 실험 스크립트

하이퍼파라미터 튜닝과 동일한 구조를 유지하면서 리샘플링 방법을 튜닝 범위에 포함하는 실험을 수행합니다.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import mlflow
import optuna
from datetime import datetime

# 프로젝트 모듈 import
from src.utils.config_manager import ConfigManager
from src.utils import setup_logging, setup_experiment_logging, log_experiment_summary, experiment_logging_context
from src.hyperparameter_tuning import HyperparameterTuner
from src.splits import split_test_set, validate_splits


# 로깅 설정
logger = logging.getLogger(__name__)


def setup_mlflow_experiment(experiment_name: str):
    """
    MLflow 실험을 설정합니다.
    
    Args:
        experiment_name: 실험 이름
        
    Returns:
        실험 ID
    """
    from src.utils.mlflow_manager import setup_mlflow_experiment_safely
    
    experiment_id = setup_mlflow_experiment_safely(experiment_name)
    logger.info(f"MLflow 실험 설정: {experiment_name} (ID: {experiment_id})")
    return experiment_id


def log_tuning_params(config: Dict[str, Any], base_config: Dict[str, Any]):
    """
    튜닝 파라미터를 MLflow에 로깅합니다.
    
    Args:
        config: 튜닝 설정
        base_config: 기본 설정
    """
    # 리샘플링 관련 파라미터 로깅
    resampling_config = config.get('imbalanced_data', {}).get('resampling', {})
    
    # 리샘플링 활성화 여부
    mlflow.log_param("resampling_enabled", resampling_config.get('enabled', False))
    
    if resampling_config.get('enabled', False):
        # 리샘플링 방법
        mlflow.log_param("resampling_method", resampling_config.get('method', 'none'))
        
        # SMOTE 관련 파라미터
        smote_config = resampling_config.get('smote', {})
        mlflow.log_param("smote_k_neighbors", smote_config.get('k_neighbors', 5))
        mlflow.log_param("smote_sampling_strategy", smote_config.get('sampling_strategy', 'auto'))
        
        # Borderline SMOTE 관련 파라미터
        borderline_smote_config = resampling_config.get('borderline_smote', {})
        mlflow.log_param("borderline_smote_k_neighbors", borderline_smote_config.get('k_neighbors', 5))
        mlflow.log_param("borderline_smote_m_neighbors", borderline_smote_config.get('m_neighbors', 10))
        mlflow.log_param("borderline_smote_sampling_strategy", borderline_smote_config.get('sampling_strategy', 'auto'))
        
        # ADASYN 관련 파라미터
        adasyn_config = resampling_config.get('adasyn', {})
        mlflow.log_param("adasyn_k_neighbors", adasyn_config.get('k_neighbors', 5))
        mlflow.log_param("adasyn_n_neighbors", adasyn_config.get('n_neighbors', 5))
        mlflow.log_param("adasyn_sampling_strategy", adasyn_config.get('sampling_strategy', 'auto'))
        
        # Under Sampling 관련 파라미터
        under_sampling_config = resampling_config.get('under_sampling', {})
        mlflow.log_param("under_sampling_strategy", under_sampling_config.get('strategy', 'random'))
        mlflow.log_param("under_sampling_sampling_strategy", under_sampling_config.get('sampling_strategy', 'auto'))
    
    # 클래스 가중치 관련 파라미터
    mlflow.log_param("class_weights", config.get('imbalanced_data', {}).get('class_weights', 'Balanced'))
    mlflow.log_param("auto_class_weights", config.get('imbalanced_data', {}).get('auto_class_weights', 'Balanced'))
    mlflow.log_param("scale_pos_weight", config.get('imbalanced_data', {}).get('scale_pos_weight', 1.0))
    
    logger.info(f"튜닝 파라미터 로깅 완료: {len(resampling_config)}개 파라미터")


def run_resampling_experiment_with_config(
    config: Dict[str, Any], 
    data_path: str = None, 
    nrows: int = None
) -> Tuple[Dict[str, Any], float, Any, Any]:
    """
    설정 기반 리샘플링 실험을 실행합니다.
    
    Args:
        config: 설정 딕셔너리
        data_path: 데이터 파일 경로
        nrows: 사용할 데이터 행 수
        
    Returns:
        튜닝 결과 튜플 (best_params, best_score, study, tuner)
    """
    import time
    start_time = time.time()
    
    # 실험 정보 추출
    model_type = config.get('model', {}).get('model_type', 'unknown')
    experiment_type = config.get('experiment_type', 'resampling_experiment')
    
    # 새로운 로깅 시스템 적용
    with experiment_logging_context(
        experiment_type=experiment_type,
        model_type=model_type,
        log_level="INFO",
        capture_console=True
    ) as log_file_path:
        
        logger.info(f"=== 리샘플링 실험 시작 ===")
        logger.info(f"모델 타입: {model_type}")
        logger.info(f"실험 타입: {experiment_type}")
        logger.info(f"로그 파일: {log_file_path}")
        
        try:
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
            
            # 튜너 생성 (리샘플링 실험용)
            tuner = ResamplingHyperparameterTuner(config, train_val_df, nrows=nrows)
            
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
                    logger.info("=== 리샘플링 실험 완료 ===")
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
                
                logger.info(f"=== 리샘플링 실험 완료 ===")
                logger.info(f"최고 성능: {best_score:.6f}")
                logger.info(f"실행 시간: {execution_time:.2f}초")
                logger.info(f"시도 횟수: {n_trials}")
                
                return best_params, best_score, run.info.experiment_id, run.info.run_id
                
        except Exception as e:
            logger.error(f"리샘플링 실험 중 오류 발생: {e}")
            raise


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


class ResamplingHyperparameterTuner(HyperparameterTuner):
    """
    리샘플링 하이퍼파라미터 튜너
    
    기존 HyperparameterTuner를 확장하여 리샘플링 방법도 튜닝 범위에 포함합니다.
    """
    
    def __init__(self, config: Dict[str, Any], train_val_df: pd.DataFrame, nrows: int = None):
        """
        리샘플링 하이퍼파라미터 튜너 초기화
        
        Args:
            config: 설정 딕셔너리
            train_val_df: 훈련/검증 데이터
            nrows: 사용할 데이터 행 수
        """
        super().__init__(config, train_val_df, nrows=nrows)
        
        # 리샘플링 방법 목록 (설정에서 지정된 방법이 있으면 해당 방법만 사용)
        config_resampling_method = config.get('imbalanced_data', {}).get('resampling', {}).get('method')
        if config_resampling_method and config_resampling_method != 'auto':
            self.resampling_methods = [config_resampling_method]
            logger.info(f"지정된 리샘플링 방법만 사용: {config_resampling_method}")
        else:
            self.resampling_methods = [
                'none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid'
            ]
        
        logger.info(f"리샘플링 하이퍼파라미터 튜너 초기화 완료")
        logger.info(f"  - 지원하는 리샘플링 방법: {self.resampling_methods}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        하이퍼파라미터를 제안합니다 (리샘플링 방법 포함).
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            제안된 하이퍼파라미터 딕셔너리
        """
        # 기본 하이퍼파라미터 제안
        params = super().suggest_hyperparameters(trial)
        
        # 리샘플링 방법 제안
        resampling_method = trial.suggest_categorical('resampling_method', self.resampling_methods)
        params['resampling_method'] = resampling_method
        
        # 리샘플링 방법별 파라미터 제안
        if resampling_method == 'smote':
            params['smote_k_neighbors'] = trial.suggest_int('smote_k_neighbors', 3, 10)
            params['smote_sampling_strategy'] = trial.suggest_float('smote_sampling_strategy', 0.05, 0.3)
        
        elif resampling_method == 'borderline_smote':
            params['borderline_smote_k_neighbors'] = trial.suggest_int('borderline_smote_k_neighbors', 3, 10)
            params['borderline_smote_m_neighbors'] = trial.suggest_int('borderline_smote_m_neighbors', 5, 15)
            params['borderline_smote_sampling_strategy'] = trial.suggest_float('borderline_smote_sampling_strategy', 0.05, 0.3)
        
        elif resampling_method == 'adasyn':
            params['adasyn_k_neighbors'] = trial.suggest_int('adasyn_k_neighbors', 3, 10)
            params['adasyn_n_neighbors'] = trial.suggest_int('adasyn_n_neighbors', 3, 10)
            params['adasyn_sampling_strategy'] = trial.suggest_float('adasyn_sampling_strategy', 0.05, 0.3)
        
        elif resampling_method == 'under_sampling':
            params['under_sampling_strategy'] = trial.suggest_categorical('under_sampling_strategy', ['random', 'tomek_links', 'edited_nearest_neighbours'])
            params['under_sampling_sampling_strategy'] = trial.suggest_float('under_sampling_sampling_strategy', 0.1, 0.5)
        
        elif resampling_method == 'hybrid':
            # Hybrid는 SMOTE + Under Sampling의 조합
            params['hybrid_smote_k_neighbors'] = trial.suggest_int('hybrid_smote_k_neighbors', 3, 10)
            params['hybrid_smote_sampling_strategy'] = trial.suggest_float('hybrid_smote_sampling_strategy', 0.05, 0.3)
            params['hybrid_under_sampling_strategy'] = trial.suggest_categorical('hybrid_under_sampling_strategy', ['random', 'tomek_links'])
            params['hybrid_under_sampling_sampling_strategy'] = trial.suggest_float('hybrid_under_sampling_sampling_strategy', 0.1, 0.5)
        
        return params
    
    def apply_resampling_parameters_to_config(self, config: Dict[str, Any], trial_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        리샘플링 파라미터를 설정에 적용합니다.
        
        Args:
            config: 원본 설정
            trial_params: trial에서 제안된 파라미터
            
        Returns:
            리샘플링 파라미터가 적용된 설정
        """
        # 설정 복사
        updated_config = config.copy()
        
        # 리샘플링 활성화
        resampling_method = trial_params.get('resampling_method', 'none')
        
        if resampling_method != 'none':
            # 리샘플링 활성화
            if 'imbalanced_data' not in updated_config:
                updated_config['imbalanced_data'] = {}
            if 'resampling' not in updated_config['imbalanced_data']:
                updated_config['imbalanced_data']['resampling'] = {}
            
            updated_config['imbalanced_data']['resampling']['enabled'] = True
            updated_config['imbalanced_data']['resampling']['method'] = resampling_method
            
            # 리샘플링 방법별 파라미터 적용
            if resampling_method == 'smote':
                if 'smote' not in updated_config['imbalanced_data']['resampling']:
                    updated_config['imbalanced_data']['resampling']['smote'] = {}
                
                updated_config['imbalanced_data']['resampling']['smote']['k_neighbors'] = trial_params.get('smote_k_neighbors', 5)
                updated_config['imbalanced_data']['resampling']['smote']['sampling_strategy'] = trial_params.get('smote_sampling_strategy', 'auto')
            
            elif resampling_method == 'borderline_smote':
                if 'borderline_smote' not in updated_config['imbalanced_data']['resampling']:
                    updated_config['imbalanced_data']['resampling']['borderline_smote'] = {}
                
                updated_config['imbalanced_data']['resampling']['borderline_smote']['k_neighbors'] = trial_params.get('borderline_smote_k_neighbors', 5)
                updated_config['imbalanced_data']['resampling']['borderline_smote']['m_neighbors'] = trial_params.get('borderline_smote_m_neighbors', 10)
                updated_config['imbalanced_data']['resampling']['borderline_smote']['sampling_strategy'] = trial_params.get('borderline_smote_sampling_strategy', 'auto')
            
            elif resampling_method == 'adasyn':
                if 'adasyn' not in updated_config['imbalanced_data']['resampling']:
                    updated_config['imbalanced_data']['resampling']['adasyn'] = {}
                
                updated_config['imbalanced_data']['resampling']['adasyn']['k_neighbors'] = trial_params.get('adasyn_k_neighbors', 5)
                updated_config['imbalanced_data']['resampling']['adasyn']['n_neighbors'] = trial_params.get('adasyn_n_neighbors', 5)
                updated_config['imbalanced_data']['resampling']['adasyn']['sampling_strategy'] = trial_params.get('adasyn_sampling_strategy', 'auto')
            
            elif resampling_method == 'under_sampling':
                if 'under_sampling' not in updated_config['imbalanced_data']['resampling']:
                    updated_config['imbalanced_data']['resampling']['under_sampling'] = {}
                
                updated_config['imbalanced_data']['resampling']['under_sampling']['strategy'] = trial_params.get('under_sampling_strategy', 'random')
                updated_config['imbalanced_data']['resampling']['under_sampling']['sampling_strategy'] = trial_params.get('under_sampling_sampling_strategy', 'auto')
            
            elif resampling_method == 'hybrid':
                # Hybrid는 SMOTE + Under Sampling의 조합
                if 'hybrid' not in updated_config['imbalanced_data']['resampling']:
                    updated_config['imbalanced_data']['resampling']['hybrid'] = {}
                
                updated_config['imbalanced_data']['resampling']['hybrid']['smote_k_neighbors'] = trial_params.get('hybrid_smote_k_neighbors', 5)
                updated_config['imbalanced_data']['resampling']['hybrid']['smote_sampling_strategy'] = trial_params.get('hybrid_smote_sampling_strategy', 'auto')
                updated_config['imbalanced_data']['resampling']['hybrid']['under_sampling_strategy'] = trial_params.get('hybrid_under_sampling_strategy', 'random')
                updated_config['imbalanced_data']['resampling']['hybrid']['under_sampling_sampling_strategy'] = trial_params.get('hybrid_under_sampling_sampling_strategy', 'auto')
        else:
            # 리샘플링 비활성화
            if 'imbalanced_data' in updated_config and 'resampling' in updated_config['imbalanced_data']:
                updated_config['imbalanced_data']['resampling']['enabled'] = False
        
        return updated_config
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective 함수 (리샘플링 파라미터 포함).
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            평가 점수
        """
        try:
            # 하이퍼파라미터 제안 (리샘플링 포함)
            trial_params = self.suggest_hyperparameters(trial)
            
            # 리샘플링 파라미터를 설정에 적용
            updated_config = self.apply_resampling_parameters_to_config(self.config, trial_params)
            
            # 기본 하이퍼파라미터 추출
            model_params = {}
            for key, value in trial_params.items():
                if not key.startswith(('resampling_', 'smote_', 'borderline_smote_', 'adasyn_', 'under_sampling_', 'hybrid_')):
                    model_params[key] = value
            
            # 모델 타입 확인
            model_type = self.config.get('model', {}).get('model_type', 'catboost')
            
            # MLflow 파라미터 로깅
            for param, value in trial_params.items():
                mlflow.log_param(param, value)
            
            # 교차 검증 수행
            from src.training import cross_validate_model
            
            # 교차 검증 실행
            cv_results = cross_validate_model(
                data=self.data,
                config=updated_config,
                model_params=model_params,
                model_type=model_type,
                n_splits=self.config.get('validation', {}).get('num_cv_folds', 5),
                random_state=self.config.get('validation', {}).get('random_state', 42)
            )
            
            # 주요 메트릭 추출
            primary_metric = self.config.get('evaluation', {}).get('primary_metric', 'f1')
            if primary_metric in cv_results:
                score = cv_results[primary_metric]
            else:
                # 기본값으로 f1 사용
                score = cv_results.get('f1', 0.0)
            
            # MLflow 메트릭 로깅
            for metric, value in cv_results.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    mlflow.log_metric(metric, value)
            
            logger.info(f"Trial {trial.number} 완료: {primary_metric} = {score:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} 실패: {e}")
            return 0.0


def main():
    """메인 함수 (ConfigManager 기반 계층적 config 병합 지원)"""
    parser = argparse.ArgumentParser(
        description="리샘플링 실험 실행 (ConfigManager 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 계층적 config로 리샘플링 실험 실행
  python scripts/run_resampling_experiment.py --model-type xgboost --experiment-type resampling_experiment
  # legacy 단일 파일 config로 리샘플링 실험 실행
  python scripts/run_resampling_experiment.py --tuning_config configs/resampling_experiment.yaml --base_config configs/default_config.yaml
        """
    )
    parser.add_argument("--model-type", type=str, choices=['xgboost', 'lightgbm', 'random_forest', 'catboost'], default=None, help="사용할 모델 타입 (계층적 config 병합)")
    parser.add_argument("--experiment-type", type=str, default=None, help="실험 타입 (resampling_experiment 등)")
    parser.add_argument("--tuning_config", type=str, default=None, help="(legacy) 튜닝 설정 파일 경로")
    parser.add_argument("--base_config", type=str, default=None, help="(legacy) 기본 설정 파일 경로")
    parser.add_argument("--data_path", type=str, default=None, help="데이터 파일 경로")
    parser.add_argument("--nrows", type=int, default=None, help="사용할 데이터 행 수")
    parser.add_argument("--mlflow_ui", action="store_true", help="실험 완료 후 MLflow UI 실행")
    
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
    parser.add_argument("--resampling-method", type=str, choices=['none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid'], default=None, help="특정 리샘플링 방법 지정 (지정하지 않으면 튜닝에서 자동 선택)")
    args = parser.parse_args()
    
    try:
        if args.model_type:
            # ConfigManager 기반 계층적 config 병합
            config_manager = ConfigManager()
            config = config_manager.create_experiment_config(args.model_type, args.experiment_type or 'resampling_experiment')
            
            # 명령행 인자를 config에 적용
            config = config_manager.apply_command_line_args(config, args)
            
            # 리샘플링 방법이 지정된 경우 설정에 적용
            if args.resampling_method:
                if 'imbalanced_data' not in config:
                    config['imbalanced_data'] = {}
                if 'resampling' not in config['imbalanced_data']:
                    config['imbalanced_data']['resampling'] = {}
                config['imbalanced_data']['resampling']['method'] = args.resampling_method
                logger.info(f"지정된 리샘플링 방법: {args.resampling_method}")
            
            # 설정 요약 출력
            config_manager.print_config_summary(config)
            
            result = run_resampling_experiment_with_config(
                config,
                data_path=args.data_path,
                nrows=args.nrows
            )
            # 실험 결과 저장은 run_resampling_experiment_with_config 내부에서 처리됨
        elif args.tuning_config and args.base_config:
            # legacy: 단일 파일 기반
            raise NotImplementedError("Legacy 모드는 지원하지 않습니다. ConfigManager를 사용해주세요.")
        else:
            raise ValueError("model-type 또는 tuning_config+base_config 중 하나는 반드시 지정해야 합니다.")
        
        best_params, best_score, _, _ = result
        print(f"\n🎉 리샘플링 실험이 성공적으로 완료되었습니다!")
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
        logger.error(f"리샘플링 실험 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main() 