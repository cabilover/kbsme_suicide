"""
하이퍼파라미터 튜닝 모듈

Optuna를 활용한 하이퍼파라미터 최적화를 구현합니다.
MLflow와 연동되어 실험 추적이 가능하며, 기존 ML 파이프라인과 완벽 호환됩니다.
"""

import optuna
import mlflow
import logging
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import time
import numbers
import gc
import psutil
from src.utils import safe_float_conversion, is_valid_number, safe_float, setup_logging

# 프로젝트 루트를 Python 경로에 추가
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.splits import load_config, split_test_set, get_cv_splits
from src.preprocessing import transform_data, fit_preprocessing_pipeline
from src.feature_engineering import transform_features, get_feature_columns, get_target_columns_from_data
from src.models import ModelFactory
from src.evaluation import calculate_all_metrics

# 로깅 설정
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def monitor_memory_usage():
    """현재 메모리 사용량을 모니터링하고 로깅합니다."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        logger.info(f"메모리 사용량: {memory_info.rss / 1024 / 1024:.1f}MB ({memory_percent:.1f}%)")
        return memory_info.rss / 1024 / 1024  # MB 단위
    except Exception as e:
        logger.warning(f"메모리 모니터링 실패: {e}")
        return None


def cleanup_memory():
    """메모리를 강제로 정리합니다."""
    try:
        # Python 가비지 컬렉션
        collected = gc.collect()
        logger.info(f"가비지 컬렉션 완료: {collected}개 객체 정리")
        
        # 메모리 사용량 확인
        memory_usage = monitor_memory_usage()
        
        return memory_usage
    except Exception as e:
        logger.warning(f"메모리 정리 실패: {e}")
        return None


def safe_cleanup_memory():
    """안전한 메모리 정리 - 중요한 객체는 보호"""
    try:
        # 현재 메모리 상태 확인
        before_memory = monitor_memory_usage()
        
        # 가비지 컬렉션만 수행 (강제 정리는 하지 않음)
        collected = gc.collect()
        
        # 메모리 정리 후 상태 확인
        after_memory = monitor_memory_usage()
        
        if before_memory and after_memory:
            freed_memory = before_memory - after_memory
            logger.info(f"안전한 메모리 정리: {collected}개 객체 정리, {freed_memory:.1f}MB 해제")
        
        return after_memory
    except Exception as e:
        logger.warning(f"안전한 메모리 정리 실패: {e}")
        return None


class HyperparameterTuner:
    """
    Optuna를 사용한 하이퍼파라미터 튜닝 클래스
    
    이 클래스는 MLflow와 연동되어 실험 추적 및 결과 저장을 지원합니다.
    """
    
    def __init__(self, config: Dict[str, Any], data: Optional[pd.DataFrame] = None, data_path: Optional[str] = None, nrows: Optional[int] = None):
        """
        하이퍼파라미터 튜너를 초기화합니다.
        
        Args:
            config: 설정 딕셔너리
            data: 데이터프레임 (직접 전달하는 경우)
            data_path: 데이터 파일 경로 (data가 None인 경우 사용)
            nrows: 사용할 데이터 행 수 (None이면 전체)
        """
        self.config = config
        self.nrows = nrows
        
        # 데이터 경로 설정
        if data_path is not None:
            self.data_path = data_path
        elif 'data' in config and 'file_path' in config['data']:
            self.data_path = config['data']['file_path']
        else:
            self.data_path = 'data/processed/processed_data_with_features.csv'  # 기본값
        
        # 로그 파일 설정 (MLflow를 통한 로깅 사용)
        self.log_file_path = None
        
        # MLflow 설정 (실험 시작 시 자동으로 설정됨)
        pass
        
        # 데이터 로딩
        if data is not None:
            self.data = data
            logger.info(f"전달받은 데이터 사용: {self.data.shape}")
        else:
            self.data = self.load_data()
        
        # 모델 타입 확인
        self.model_type = config.get('model', {}).get('model_type', 'xgboost')
        logger.info(f"모델 타입: {self.model_type}")
        
        # 튜닝 설정
        self.tuning_config = config
        self.n_trials = self.tuning_config.get('tuning', {}).get('n_trials', 100)
        self.timeout = self.tuning_config.get('tuning', {}).get('timeout', 3600)
        
        logger.info(f"튜닝 설정 - 시도 횟수: {self.n_trials}, 타임아웃: {self.timeout}초")
        
        # Optuna study 및 결과 저장용
        self.study = None
        self.best_params = None
        self.best_score = None
        
        logger.info("하이퍼파라미터 튜너 초기화 완료")
        logger.info(f"  - 튜닝 설정: {self.tuning_config}")
        logger.info(f"  - 튜닝 횟수: {self.n_trials}")
        logger.info(f"  - 타임아웃: {self.timeout}초")
    
    def _get_base_config_path(self) -> str:
        """(더 이상 사용하지 않음)"""
        raise NotImplementedError("_get_base_config_path는 계층적 config 체계에서 사용하지 않습니다.")
    
    def _load_tuning_config(self) -> Dict[str, Any]:
        """튜닝 설정 파일을 로드합니다."""
        with open(self.tuning_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _merge_configs(self) -> Dict[str, Any]:
        """기본 설정과 튜닝 설정을 병합합니다."""
        # 기본 설정 로드
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 튜닝 설정에서 오버라이드 설정이 있으면 적용
        if self.tuning_config.get('override_base_config', {}).get('enabled', False):
            override_config = self.tuning_config['override_base_config']
            self._deep_merge(base_config, override_config)  # 반환값 받지 않음
        
        return base_config
    
    def _deep_merge(self, base_dict: Dict, override_dict: Dict):
        """중첩된 딕셔너리를 재귀적으로 병합합니다."""
        for key, value in override_dict.items():
            if key == 'enabled':
                continue
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """설정에 따라 Optuna sampler를 생성합니다."""
        sampler_config = self.tuning_config['sampler']
        sampler_type = sampler_config['type']
        
        if sampler_type == "tpe":
            tpe_config = sampler_config.get('tpe', {})
            return optuna.samplers.TPESampler(
                seed=self.config['data_split']['random_state'],
                n_startup_trials=tpe_config.get('n_startup_trials', 10),
                n_ei_candidates=tpe_config.get('n_ei_candidates', 24),
                multivariate=tpe_config.get('multivariate', True),
                group=tpe_config.get('group', True)
            )
        
        elif sampler_type == "random":
            return optuna.samplers.RandomSampler(
                seed=self.config['data_split']['random_state']
            )
        
        elif sampler_type == "cmaes":
            cmaes_config = sampler_config.get('cmaes', {})
            x0 = cmaes_config.get('x0', {})
            sigma0 = cmaes_config.get('sigma0', 0.1)
            return optuna.samplers.CmaEsSampler(
                seed=self.config['data_split']['random_state'],
                x0=x0,
                sigma0=sigma0
            )
        
        elif sampler_type == "qmc":
            return optuna.samplers.QMCSampler(
                seed=self.config['data_split']['random_state']
            )
        
        else:
            raise ValueError(f"지원하지 않는 sampler 타입: {sampler_type}")
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Optuna trial에서 하이퍼파라미터를 제안합니다.
        Args:
            trial: Optuna trial 객체
        Returns:
            제안된 하이퍼파라미터 딕셔너리 (모델 파라미터만)
        """
        params = {}
        # 모델 타입 확인
        model_type = self.config.get('model', {}).get('model_type', 'xgboost')
        # 모델 타입별 파라미터 설정
        if model_type == 'xgboost':
            model_params = self.tuning_config.get('xgboost_params', {})
        elif model_type == 'lightgbm':
            model_params = self.tuning_config.get('lightgbm_params', {})
        elif model_type == 'random_forest':
            model_params = self.tuning_config.get('random_forest_params', {})
        elif model_type == 'catboost':
            model_params = self.tuning_config.get('catboost_params', {})
        else:
            logger.warning(f"지원하지 않는 모델 타입: {model_type}. XGBoost 파라미터를 사용합니다.")
            model_params = self.tuning_config.get('xgboost_params', {})
        for param_name, param_config in model_params.items():
            param_type = param_config['type']
            if param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            else:
                low = float(param_config['low']) if param_type == "float" else int(param_config['low'])
                high = float(param_config['high']) if param_type == "float" else int(param_config['high'])
                log = param_config.get('log', False)
                if param_type == "int":
                    params[param_name] = trial.suggest_int(param_name, low, high, log=log)
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(param_name, low, high, log=log)
        
        # Random Forest의 class_weight_pos를 class_weight 딕셔너리로 변환
        if model_type == 'random_forest' and 'class_weight_pos' in params:
            class_weight_pos = params.pop('class_weight_pos')
            params['class_weight'] = {0: 1.0, 1: class_weight_pos}
            logger.info(f"Random Forest class_weight 설정: {params['class_weight']}")
        
        # 튜닝/실험용 파라미터(n_jobs 등)는 모델 파라미터 dict에 포함하지 않음
        return params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective 함수
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            평가 점수
        """
        try:
            # 하이퍼파라미터 제안
            params = self._suggest_parameters(trial)
            
            # Trial 시작 로깅
            logger.info(f"Trial {trial.number} 시작")
            
            # 1. 하이퍼파라미터 로깅
            for param_name, param_value in params.items():
                try:
                    mlflow.log_param(f"trial_{trial.number}_{param_name}", param_value)
                except Exception as e:
                    logger.warning(f"Trial {trial.number} 하이퍼파라미터 로깅 실패 ({param_name}): {e}")
            
            # 설정 업데이트
            trial_config = self.config.copy()
            
            # 모델 타입 확인 및 로깅
            model_type = trial_config.get('model', {}).get('model_type', 'xgboost')
            logger.info(f"Trial {trial.number}: 모델 타입 = {model_type}")
            
            # Focal Loss 파라미터 처리
            focal_loss_params = {}
            params_copy = params.copy()
            
            if 'use_focal_loss' in params_copy:
                focal_loss_params['use_focal_loss'] = params_copy.pop('use_focal_loss')
            if 'focal_loss_alpha' in params_copy:
                focal_loss_params['focal_loss'] = {'alpha': params_copy.pop('focal_loss_alpha')}
            if 'focal_loss_gamma' in params_copy:
                if 'focal_loss' not in focal_loss_params:
                    focal_loss_params['focal_loss'] = {}
                focal_loss_params['focal_loss']['gamma'] = params_copy.pop('focal_loss_gamma')
            
            # 모델 타입별 파라미터 업데이트
            if model_type == 'xgboost':
                trial_config['model']['xgboost'].update(params_copy)
                if focal_loss_params:
                    trial_config['model']['xgboost'].update(focal_loss_params)
            elif model_type == 'lightgbm':
                trial_config['model']['lightgbm'].update(params_copy)
                if focal_loss_params:
                    trial_config['model']['lightgbm'].update(focal_loss_params)
            elif model_type == 'random_forest':
                trial_config['model']['random_forest'].update(params_copy)
                if focal_loss_params:
                    trial_config['model']['random_forest'].update(focal_loss_params)
            elif model_type == 'catboost':
                trial_config['model']['catboost'].update(params_copy)
                if focal_loss_params:
                    trial_config['model']['catboost'].update(focal_loss_params)
            else:
                logger.warning(f"지원하지 않는 모델 타입: {model_type}. XGBoost를 사용합니다.")
                trial_config['model']['xgboost'].update(params_copy)
                if focal_loss_params:
                    trial_config['model']['xgboost'].update(focal_loss_params)
            
            # 데이터 로드 (suicide_c, suicide_y, check 컬럼 제외)
            if self.nrows:
                df = pd.read_csv(self.data_path, nrows=self.nrows)
                logger.info(f"데이터 로드: {self.nrows}개 행 사용")
            else:
                df = pd.read_csv(self.data_path)
            
            # 불필요한 컬럼 제거 (NaN이 대부분인 컬럼들)
            columns_to_drop = ['suicide_c', 'suicide_y', 'check']
            existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
            if existing_columns_to_drop:
                df = df.drop(columns=existing_columns_to_drop)
                logger.info(f"불필요한 컬럼 제거: {existing_columns_to_drop}")
            
            # 테스트 세트 분리
            train_val_df, test_df, _ = split_test_set(df, self.config)
            
            # 교차 검증 수행
            from src.training import run_cross_validation
            cv_results = run_cross_validation(train_val_df, trial_config)
            
            if not cv_results or not cv_results.get('fold_results'):
                logger.warning(f"Trial {trial.number}: 교차 검증 실패")
                # 실패한 trial 로깅
                try:
                    mlflow.log_param(f"trial_{trial.number}_status", "failed")
                    mlflow.log_param(f"trial_{trial.number}_error", "cross_validation_failed")
                except Exception as e:
                    logger.warning(f"Trial {trial.number} 실패 로깅 실패: {e}")
                return float('-inf') if self.tuning_config['tuning']['direction'] == 'maximize' else float('inf')
            
            # 고급 평가 분석
            from src.evaluation import (
                analyze_fold_performance_distribution, analyze_fold_variability,
                calculate_confidence_intervals, create_comprehensive_evaluation_report
            )
            
            fold_results = cv_results['fold_results']
            
            # 폴드별 성능 분포 분석
            performance_distribution = analyze_fold_performance_distribution(fold_results)
            
            # 폴드 간 변동성 분석
            variability = analyze_fold_variability(fold_results)
            
            # 주요 지표 추출 및 고급 메트릭 로깅
            primary_metric = self.config.get('evaluation', {}).get('primary_metric', 'f1')
            cv_scores = []
            
            # MLflow 로깅 전에 실제 메트릭 값이 있는지 확인
            valid_metrics_found = False
            
            # 2. 각 폴드별 상세 메트릭 로깅
            for fold_idx, fold_result in enumerate(fold_results):
                # calculate_all_metrics의 반환 형식에 맞게 처리
                fold_metrics = fold_result.get('training_results', {}).get('val_metrics', {})
                
                # 타겟별 메트릭에서 primary_metric 찾기
                found_metric = False
                for target_name, target_metrics in fold_metrics.items():
                    if primary_metric in target_metrics:
                        score = target_metrics[primary_metric]
                        # 안전한 float 변환
                        score = safe_float_conversion(score)
                        
                        # 유효한 숫자인지 확인
                        if not is_valid_number(score):
                            logger.warning(f"Trial {trial.number} - 유효하지 않은 score 타입: {type(score)}, 값: {score}")
                            score = 0.0
                        
                        # NaN/Inf 체크 (이미 float이므로 안전)
                        if np.isnan(score) or np.isinf(score):
                            logger.warning(f"Trial {trial.number} - NaN/Inf score: {score}")
                            score = 0.0
                        
                        cv_scores.append(score)
                        valid_metrics_found = True
                        found_metric = True
                        
                        # 3. 각 폴드의 모든 메트릭 로깅
                        for metric_name, metric_value in target_metrics.items():
                            try:
                                safe_target_name = target_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                                safe_metric_name = metric_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                                mlflow.log_metric(f"trial_{trial.number}_fold_{fold_idx}_{safe_target_name}_{safe_metric_name}", safe_float(metric_value))
                            except Exception as e:
                                logger.warning(f"Trial {trial.number} 폴드 {fold_idx} 메트릭 로깅 실패 ({metric_name}): {e}")
                        
                        # 4. 폴드별 primary_metric 로깅 (기존 방식 유지)
                        safe_target_name = target_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                        safe_metric_name = primary_metric.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                        try:
                            mlflow.log_metric(f"fold_{fold_idx}_{safe_target_name}_{safe_metric_name}", safe_float(score))
                        except Exception as e:
                            logger.warning(f"MLflow 로깅 실패: {e}")
                        break  # 첫 번째 타겟에서 찾으면 중단
                
                if not found_metric:
                    logger.warning(f"Trial {trial.number} - primary_metric '{primary_metric}' not found in any target metrics")
                    cv_scores.append(0.0)
                    # 실패한 폴드 로깅
                    try:
                        mlflow.log_param(f"trial_{trial.number}_fold_{fold_idx}_status", "failed")
                        mlflow.log_param(f"trial_{trial.number}_fold_{fold_idx}_error", "primary_metric_not_found")
                    except Exception as e:
                        logger.warning(f"Trial {trial.number} 폴드 {fold_idx} 실패 로깅 실패: {e}")
            
            # CV 스코어 계산
            if cv_scores and valid_metrics_found:
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                logger.info(f"Trial {trial.number}: CV 스코어 = {mean_score:.6f} ± {std_score:.6f} (개별: {[f'{s:.6f}' for s in cv_scores]})")
                
                # 유효한 값인지 확인
                if np.isnan(mean_score) or np.isinf(mean_score):
                    logger.warning(f"Trial {trial.number}: 유효하지 않은 평균 스코어: {mean_score}")
                    # 실패한 trial 로깅
                    try:
                        mlflow.log_param(f"trial_{trial.number}_status", "failed")
                        mlflow.log_param(f"trial_{trial.number}_error", "invalid_mean_score")
                    except Exception as e:
                        logger.warning(f"Trial {trial.number} 실패 로깅 실패: {e}")
                    return float('-inf') if self.config.get('tuning', {}).get('direction', 'maximize') == 'maximize' else float('inf')
                
                # 5. Trial 성공 로깅 및 성능 지표
                try:
                    mlflow.log_param(f"trial_{trial.number}_status", "success")
                    mlflow.log_metric(f"trial_{trial.number}_mean_score", safe_float(mean_score))
                    mlflow.log_metric(f"trial_{trial.number}_std_score", safe_float(std_score))
                    mlflow.log_metric(f"trial_{trial.number}_min_score", safe_float(min(cv_scores)))
                    mlflow.log_metric(f"trial_{trial.number}_max_score", safe_float(max(cv_scores)))
                    
                    # 성능 변화 추이를 위한 trial 번호 로깅
                    mlflow.log_metric("trial_number", trial.number)
                    mlflow.log_metric("trial_score", safe_float(mean_score))
                    
                except Exception as e:
                    logger.warning(f"Trial {trial.number} 성공 로깅 실패: {e}")
                
                return mean_score
            else:
                logger.warning(f"Trial {trial.number}: CV 스코어가 비어있습니다.")
                # 실패한 trial 로깅
                try:
                    mlflow.log_param(f"trial_{trial.number}_status", "failed")
                    mlflow.log_param(f"trial_{trial.number}_error", "empty_cv_scores")
                except Exception as e:
                    logger.warning(f"Trial {trial.number} 실패 로깅 실패: {e}")
                return float('-inf') if self.tuning_config['tuning']['direction'] == 'maximize' else float('inf')
        
        except Exception as e:
            # 기존 에러 로깅
            logger.error(f"[DEBUG] Trial {trial.number} 예외 발생: {e}")
            # 추가: feature_columns와 관련 DataFrame 컬럼 로그
            try:
                if 'feature_columns' in locals():
                    logger.error(f"[DEBUG][EXCEPTION] Trial {trial.number} - feature_columns: {feature_columns}")
                if 'train_processed' in locals():
                    logger.error(f"[DEBUG][EXCEPTION] Trial {trial.number} - train_processed.columns: {list(train_processed.columns)}")
                if 'train_engineered' in locals():
                    logger.error(f"[DEBUG][EXCEPTION] Trial {trial.number} - train_engineered.columns: {list(train_engineered.columns)}")
                if 'X_train' in locals():
                    logger.error(f"[DEBUG][EXCEPTION] Trial {trial.number} - X_train.columns: {list(X_train.columns)}")
            except Exception as log_e:
                logger.error(f"[DEBUG][EXCEPTION] Trial {trial.number} - 추가 로그 중 에러: {log_e}")
            # 기존 로직 유지
            missing = [col for col in feature_columns if (('train_processed' in locals() and col not in train_processed.columns) or ('train_engineered' in locals() and col not in train_engineered.columns))] if 'feature_columns' in locals() else []
            logger.error(f"[DEBUG] Trial {trial.number} - 누락된 컬럼: {missing}")
            return float('-inf') if self.tuning_config['tuning']['direction'] == 'maximize' else float('inf')
        
        finally:
            # Trial 완료 후 메모리 정리 (안전한 방식)
            if trial.number % 10 == 0:  # 10개 trial마다 메모리 정리 (덜 자주)
                logger.info(f"Trial {trial.number} 완료 후 메모리 정리")
                # 안전한 메모리 정리 - 중요한 객체는 보호
                try:
                    # 현재 trial의 결과가 이미 MLflow에 저장되었는지 확인
                    if hasattr(trial, 'state') and trial.state == optuna.trial.TrialState.COMPLETE:
                        # trial이 성공적으로 완료된 경우에만 메모리 정리
                        collected = gc.collect()
                        logger.info(f"Trial {trial.number} 메모리 정리 완료: {collected}개 객체")
                except Exception as e:
                    logger.warning(f"Trial {trial.number} 메모리 정리 실패: {e}")
            # === 로그 핸들러 및 표준출력 flush 추가 ===
            import sys
            for handler in logger.handlers:
                try:
                    handler.flush()
                except Exception:
                    pass
            try:
                sys.stdout.flush()
            except Exception:
                pass 

    def optimize(self, start_mlflow_run: bool = True) -> Tuple[Dict[str, Any], float]:
        """
        하이퍼파라미터 최적화를 실행합니다.
        
        Args:
            start_mlflow_run: MLflow run을 시작할지 여부 (중첩 실행 시 False)
        Returns:
            최적 하이퍼파라미터와 최고 성능 점수
        """
        logger.info("하이퍼파라미터 최적화 시작")
        # MLflow 실험 설정
        experiment_name = self.tuning_config['mlflow']['experiment_name']
        mlflow.set_experiment(experiment_name)
        # Optuna study 생성
        sampler = self._create_sampler()
        direction = self.tuning_config['tuning']['direction']
        model_type = self.config.get('model', {}).get('model_type', 'xgboost')
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name=f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        # 최적화 실행
        n_trials = self.tuning_config['tuning']['n_trials']
        n_jobs = self.tuning_config['tuning'].get('n_jobs', 1)
        
        if start_mlflow_run:
            with mlflow.start_run(nested=True):
                return self._run_optimization(n_trials, n_jobs)
        else:
            return self._run_optimization(n_trials, n_jobs)

    def _run_optimization(self, n_trials: int, n_jobs: int) -> Tuple[Dict[str, Any], float]:
        """실제 최적화 실행 로직"""
        logger.info("최적화 시작 전 메모리 상태:")
        monitor_memory_usage()
        from src.utils.mlflow_manager import safe_log_param, safe_log_metric
        safe_log_param("optimization_algorithm", self.tuning_config['sampler']['type'], logger)
        safe_log_param("n_trials", n_trials, logger)
        safe_log_param("direction", self.tuning_config['tuning']['direction'], logger)
        primary_metric = self.config.get('evaluation', {}).get('primary_metric', 'f1')
        safe_log_param("primary_metric", primary_metric, logger)
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        logger.info("최적화 완료 후 메모리 상태:")
        monitor_memory_usage()
        logger.info("최적화 완료 후 안전한 메모리 정리:")
        safe_cleanup_memory()
        # 전체 trial 통계
        all_trials = self.study.trials
        successful_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
        safe_log_param("total_trials", len(all_trials), logger)
        safe_log_param("successful_trials", len(successful_trials), logger)
        safe_log_param("failed_trials", len(failed_trials), logger)
        safe_log_param("success_rate", len(successful_trials) / len(all_trials) if all_trials else 0, logger)
        # 성능 통계
        if successful_trials:
            scores = [t.value for t in successful_trials if t.value is not None]
            if scores:
                safe_log_metric("all_trials_mean_score", safe_float(np.mean(scores)), logger_instance=logger)
                safe_log_metric("all_trials_std_score", safe_float(np.std(scores)), logger_instance=logger)
                safe_log_metric("all_trials_min_score", safe_float(np.min(scores)), logger_instance=logger)
                safe_log_metric("all_trials_max_score", safe_float(np.max(scores)), logger_instance=logger)
                safe_log_metric("all_trials_median_score", safe_float(np.median(scores)), logger_instance=logger)
        safe_log_param("best_trial_number", self.study.best_trial.number, logger)
        safe_log_param("optimization_duration_seconds", 
                       (self.study.best_trial.datetime_complete - self.study.best_trial.datetime_start).total_seconds() 
                       if self.study.best_trial.datetime_complete else 0, logger)
        for param_name, param_value in self.best_params.items():
            safe_log_param(f"best_{param_name}", param_value, logger)
        if isinstance(self.best_score, (int, float)) and not np.isnan(self.best_score) and not np.isinf(self.best_score):
            safe_log_metric("best_score", safe_float(self.best_score), logger_instance=logger)
        else:
            logger.warning(f"유효하지 않은 best_score: {self.best_score}")
            try:
                self.best_score = float(self.best_score) if self.best_score != float('-inf') and self.best_score != float('inf') else 0.0
                safe_log_metric("best_score", safe_float(self.best_score), logger_instance=logger)
            except (ValueError, TypeError) as e:
                logger.warning(f"best_score 타입 변환 실패: {self.best_score} ({type(self.best_score)}): {e}")
                self.best_score = 0.0
                safe_log_metric("best_score", safe_float(self.best_score), logger_instance=logger)
        logger.info(f"최적화 완료!")
        logger.info(f"  - 최고 성능: {self.best_score:.4f}")
        logger.info(f"  - 최적 파라미터: {self.best_params}")
        return self.best_params, self.best_score 