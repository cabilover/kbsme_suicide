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
from src.utils import safe_float_conversion, is_valid_number

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        # 로그 파일 설정
        self.log_file_path = setup_tuning_logger()
        
        # MLflow 설정
        self.setup_mlflow()
        
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
            
            # 데이터 로드
            if self.nrows:
                df = pd.read_csv(self.data_path, nrows=self.nrows)
                logger.info(f"데이터 로드: {self.nrows}개 행 사용")
            else:
                df = pd.read_csv(self.data_path)
            
            # 테스트 세트 분리
            train_val_df, test_df, _ = split_test_set(df, self.config)
            
            # 교차 검증 수행
            from src.training import run_cross_validation
            cv_results = run_cross_validation(train_val_df, trial_config)
            
            if not cv_results or not cv_results.get('fold_results'):
                logger.warning(f"Trial {trial.number}: 교차 검증 실패")
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
            primary_metric = self.tuning_config['evaluation']['primary_metric']
            cv_scores = []
            
            # MLflow 로깅 전에 실제 메트릭 값이 있는지 확인
            valid_metrics_found = False
            
            for fold_idx, fold_result in enumerate(fold_results):
                metrics = fold_result.get('metrics', {})
                
                # 기본 지표 추출
                if primary_metric in metrics:
                    score = metrics[primary_metric]
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
                    
                    # MLflow 로깅
                    try:
                        mlflow.log_metric(f"fold_{fold_idx}_{primary_metric}", score)
                    except Exception as e:
                        logger.warning(f"MLflow 로깅 실패: {e}")
                else:
                    logger.warning(f"Trial {trial.number} - primary_metric '{primary_metric}' not found in metrics")
                    cv_scores.append(0.0)
            
            # CV 스코어 계산
            if cv_scores and valid_metrics_found:
                mean_score = np.mean(cv_scores)
                logger.info(f"Trial {trial.number}: CV 스코어 = {mean_score:.6f} (개별: {[f'{s:.6f}' for s in cv_scores]})")
                
                # 유효한 값인지 확인
                if np.isnan(mean_score) or np.isinf(mean_score):
                    logger.warning(f"Trial {trial.number}: 유효하지 않은 평균 스코어: {mean_score}")
                    return float('-inf') if self.tuning_config['tuning']['direction'] == 'maximize' else float('inf')
                
                return mean_score
            else:
                logger.warning(f"Trial {trial.number}: CV 스코어가 비어있습니다.")
                return float('-inf') if self.tuning_config['tuning']['direction'] == 'maximize' else float('inf')
            
        except Exception as e:
            logger.error(f"Trial {trial.number} 실패: {str(e)}")
            # 에러 로깅 제거 (중복 방지)
            # mlflow.log_param(f"trial_{trial.number}_error", str(e))
            return float('-inf') if self.tuning_config['tuning']['direction'] == 'maximize' else float('inf')
    
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
        try:
            mlflow.log_param("optimization_algorithm", self.tuning_config['sampler']['type'])
        except Exception as e:
            logger.warning(f"MLflow 최적화 알고리즘 로깅 실패: {e}")
        
        try:
            mlflow.log_param("n_trials", n_trials)
        except Exception as e:
            logger.warning(f"MLflow n_trials 로깅 실패: {e}")
        
        try:
            mlflow.log_param("direction", self.tuning_config['tuning']['direction'])
        except Exception as e:
            logger.warning(f"MLflow direction 로깅 실패: {e}")
        
        try:
            mlflow.log_param("primary_metric", self.tuning_config['evaluation']['primary_metric'])
        except Exception as e:
            logger.warning(f"MLflow primary_metric 로깅 실패: {e}")
        
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # 최적 파라미터를 best_ 접두사로 로깅하여 중복 방지
        for param_name, param_value in self.best_params.items():
            try:
                mlflow.log_param(f"best_{param_name}", param_value)
            except Exception as e:
                logger.warning(f"MLflow 최적 파라미터 로깅 실패 (best_{param_name}): {e}")
        
        if isinstance(self.best_score, (int, float)) and not np.isnan(self.best_score) and not np.isinf(self.best_score):
            try:
                mlflow.log_metric("best_score", self.best_score)
            except Exception as e:
                logger.warning(f"MLflow best_score 로깅 실패: {e}")
        else:
            logger.warning(f"유효하지 않은 best_score: {self.best_score}")
            # best_score가 유효하지 않은 경우 0.0으로 설정
            try:
                self.best_score = float(self.best_score) if self.best_score != float('-inf') and self.best_score != float('inf') else 0.0
                mlflow.log_metric("best_score", self.best_score)
            except (ValueError, TypeError) as e:
                logger.warning(f"best_score 타입 변환 실패: {self.best_score} ({type(self.best_score)}): {e}")
                self.best_score = 0.0
                mlflow.log_metric("best_score", self.best_score)
        
        logger.info(f"최적화 완료!")
        logger.info(f"  - 최고 성능: {self.best_score:.4f}")
        logger.info(f"  - 최적 파라미터: {self.best_params}")
        return self.best_params, self.best_score
    
    def save_results(self):
        """튜닝 결과를 저장합니다."""
        if self.study is None:
            logger.warning("저장할 튜닝 결과가 없습니다.")
            return
        
        results_config = self.tuning_config['results']
        
        # 최고 모델 저장
        if results_config.get('save_best_model', False):
            self._save_best_model()
        
        # Study 저장
        if results_config.get('save_study', False):
            self._save_study()
        
        # 시각화 생성
        if results_config.get('create_plots', False):
            self._create_plots()
    
    def _save_best_model(self):
        """최적 파라미터로 학습된 모델을 저장합니다."""
        model_save_path = self.tuning_config['results']['model_save_path']
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 최적 파라미터로 모델 재학습
        model_config = self.config.copy()
        
        # Focal Loss 파라미터 처리
        focal_loss_params = {}
        best_params_copy = self.best_params.copy()
        
        if 'use_focal_loss' in best_params_copy:
            focal_loss_params['use_focal_loss'] = best_params_copy.pop('use_focal_loss')
        if 'focal_loss_alpha' in best_params_copy:
            focal_loss_params['focal_loss'] = {'alpha': best_params_copy.pop('focal_loss_alpha')}
        if 'focal_loss_gamma' in best_params_copy:
            if 'focal_loss' not in focal_loss_params:
                focal_loss_params['focal_loss'] = {}
            focal_loss_params['focal_loss']['gamma'] = best_params_copy.pop('focal_loss_gamma')
        
        # 모델 타입별 파라미터 업데이트
        if model_config['model']['model_type'] == 'xgboost':
            model_config['model']['xgboost'].update(best_params_copy)
            if focal_loss_params:
                model_config['model']['xgboost'].update(focal_loss_params)
        elif model_config['model']['model_type'] == 'lightgbm':
            model_config['model']['lightgbm'].update(best_params_copy)
            if focal_loss_params:
                model_config['model']['lightgbm'].update(focal_loss_params)
        elif model_config['model']['model_type'] == 'random_forest':
            model_config['model']['random_forest'].update(best_params_copy)
            if focal_loss_params:
                model_config['model']['random_forest'].update(focal_loss_params)
        elif model_config['model']['model_type'] == 'catboost':
            model_config['model']['catboost'].update(best_params_copy)
            if focal_loss_params:
                model_config['model']['catboost'].update(focal_loss_params)
        else:
            logger.warning(f"지원하지 않는 모델 타입: {model_config['model']['model_type']}. XGBoost를 사용합니다.")
            model_config['model']['xgboost'].update(best_params_copy)
            if focal_loss_params:
                model_config['model']['xgboost'].update(focal_loss_params)
        
        # 전체 데이터로 최종 모델 학습
        df = pd.read_csv(self.data_path)
        
        # 테스트 세트 분리
        train_val_df, test_df, _ = split_test_set(df, self.config)
        
        # 전처리 및 피처 엔지니어링
        from src.preprocessing import fit_preprocessing_pipeline
        from src.feature_engineering import fit_feature_engineering
        
        preprocessor, _ = fit_preprocessing_pipeline(train_val_df, self.config)
        train_processed = transform_data(train_val_df, preprocessor, self.config)
        
        feature_info = fit_feature_engineering(train_processed, self.config)
        train_engineered = transform_features(train_processed, feature_info, self.config)
        
        # 모델 학습
        feature_columns = get_feature_columns(train_engineered, self.config)
        target_columns = get_target_columns_from_data(train_engineered, self.config)
        
        # 타겟 컬럼 매칭 문제 해결
        logger.info(f"[DEBUG] _save_best_model - train_engineered 컬럼: {list(train_engineered.columns)}")
        logger.info(f"[DEBUG] _save_best_model - feature_columns: {feature_columns}")
        logger.info(f"[DEBUG] _save_best_model - target_columns: {target_columns}")
        
        # 타겟 컬럼이 비어있는 경우 수동으로 찾기
        if not target_columns:
            logger.warning("get_target_columns_from_data에서 타겟 컬럼을 찾지 못했습니다. 수동으로 찾습니다.")
            available_targets = []
            for target in self.config['features']['target_columns']:
                candidates = [
                    target,
                    f"pass__{target}",
                    f"remainder__{target}",
                    f"num__{target}",
                    f"cat__{target}"
                ]
                for cand in candidates:
                    if cand in train_engineered.columns:
                        available_targets.append(cand)
                        logger.info(f"타겟 {target} 매칭됨: {cand}")
                        break
                else:
                    logger.warning(f"타겟 {target} 매칭 실패. 후보: {candidates}")
            
            if available_targets:
                target_columns = available_targets
                logger.info(f"수동으로 찾은 타겟 컬럼: {target_columns}")
            else:
                logger.error("타겟 컬럼을 찾을 수 없습니다!")
                return
        
        X_train = train_engineered[feature_columns]
        y_train = train_engineered[target_columns]
        
        logger.info(f"[DEBUG] _save_best_model - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"[DEBUG] _save_best_model - y_train 컬럼: {list(y_train.columns)}")
        
        # ModelFactory를 사용하여 모델 타입에 따라 동적으로 모델 생성
        model = ModelFactory.create_model(model_config)
        
        # 최종 모델 학습 시 타겟 컬럼 문제 해결
        # y_train이 비어있지 않은지 확인
        if y_train.empty or y_train.shape[1] == 0:
            logger.error("y_train이 비어있습니다. 타겟 컬럼을 다시 확인합니다.")
            # 수동으로 타겟 컬럼 찾기
            for target in self.config['features']['target_columns']:
                candidates = [
                    target,
                    f"pass__{target}",
                    f"remainder__{target}",
                    f"num__{target}",
                    f"cat__{target}"
                ]
                for cand in candidates:
                    if cand in train_engineered.columns:
                        y_train = train_engineered[[cand]]
                        logger.info(f"수동으로 타겟 컬럼 찾음: {cand}")
                        break
                if not y_train.empty:
                    break
            
            if y_train.empty:
                logger.error("타겟 컬럼을 찾을 수 없습니다!")
                return
        
        # _validate_input_data를 호출하지 않고 직접 모델 학습
        logger.info(f"최종 모델 학습 - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"최종 모델 학습 - y_train 컬럼: {list(y_train.columns)}")
        
        # 모델 학습
        model.fit(X_train, y_train)
        
        # 모델 저장
        model.save_model(model_save_path)
        logger.info(f"최고 모델 저장 완료: {model_save_path}")
    
    def _save_study(self):
        """Optuna study를 저장합니다."""
        study_save_path = self.tuning_config['results']['study_save_path']
        Path(study_save_path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.study, study_save_path)
        logger.info(f"Study 저장 완료: {study_save_path}")
    
    def _create_plots(self):
        """튜닝 결과 시각화를 생성합니다."""
        plots_save_path = self.tuning_config['results']['plots_save_path']
        Path(plots_save_path).mkdir(parents=True, exist_ok=True)
        
        # 1. 최적화 과정 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 최적화 히스토리
        try:
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=axes[0, 0])
        except TypeError:
            # 최신 버전에서는 ax 파라미터를 지원하지 않을 수 있음
            optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.savefig(f"{plots_save_path}/optimization_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 다른 플롯들도 개별적으로 생성
            try:
                optuna.visualization.matplotlib.plot_param_importances(self.study)
                plt.savefig(f"{plots_save_path}/param_importances.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"파라미터 중요도 플롯 생성 실패: {e}")
            
            try:
                optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
                plt.savefig(f"{plots_save_path}/parallel_coordinate.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"병렬 좌표 플롯 생성 실패: {e}")
            
            return
        
        axes[0, 0].set_title("Optimization History")
        
        # 파라미터 중요도
        try:
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[0, 1])
            axes[0, 1].set_title("Parameter Importances")
        except Exception as e:
            logger.warning(f"파라미터 중요도 플롯 생성 실패: {e}")
            axes[0, 1].text(0.5, 0.5, "Parameter Importances\n(Generation Failed)", 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 파라미터 관계
        try:
            optuna.visualization.matplotlib.plot_parallel_coordinate(self.study, ax=axes[1, 0])
            axes[1, 0].set_title("Parallel Coordinate Plot")
        except Exception as e:
            logger.warning(f"병렬 좌표 플롯 생성 실패: {e}")
            axes[1, 0].text(0.5, 0.5, "Parallel Coordinate Plot\n(Generation Failed)", 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 파라미터 분포
        try:
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[1, 1])
            axes[1, 1].set_title("Parameter Importances (Detailed)")
        except Exception as e:
            logger.warning(f"상세 파라미터 중요도 플롯 생성 실패: {e}")
            axes[1, 1].text(0.5, 0.5, "Parameter Importances (Detailed)\n(Generation Failed)", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{plots_save_path}/optimization_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 최적화 요약 저장
        summary_path = f"{plots_save_path}/optimization_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== 하이퍼파라미터 최적화 결과 ===\n\n")
            f.write(f"최적화 알고리즘: {self.tuning_config['sampler']['type']}\n")
            f.write(f"최적화 방향: {self.tuning_config['tuning']['direction']}\n")
            f.write(f"주요 지표: {self.tuning_config['evaluation']['primary_metric']}\n")
            f.write(f"총 시도 횟수: {len(self.study.trials)}\n")
            f.write(f"최고 성능: {self.best_score:.4f}\n\n")
            f.write("최적 파라미터:\n")
            for param, value in self.best_params.items():
                f.write(f"  {param}: {value}\n")
        
        logger.info(f"시각화 생성 완료: {plots_save_path}")

    def run_tuning(self) -> Dict[str, Any]:
        """
        하이퍼파라미터 튜닝을 실행합니다.
        
        Returns:
            튜닝 결과 딕셔너리 (best_params, best_score, tuning_time 등 포함)
        """
        logger.info("=== 하이퍼파라미터 튜닝 시작 ===")
        start_time = time.time()
        
        try:
            # Optuna study 생성
            study = optuna.create_study(
                direction=self.tuning_config.get('direction', 'maximize'),
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # 튜닝 실행
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )
            
            # 결과 추출
            best_params = study.best_params
            best_score = study.best_value
            tuning_time = time.time() - start_time
            
            logger.info(f"튜닝 완료 - 최고 성능: {best_score:.4f}")
            logger.info(f"튜닝 시간: {tuning_time:.2f}초")
            
            # 결과 저장
            self.study = study
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'tuning_time': tuning_time,
                'n_trials': len(study.trials),
                'study': study
            }
            
        except Exception as e:
            logger.error(f"튜닝 실행 중 오류 발생: {str(e)}")
            raise

    def setup_mlflow(self):
        """
        MLflow 설정을 초기화합니다.
        """
        # MLflow 설정
        mlflow_config = self.config.get('mlflow', {})
        experiment_name = mlflow_config.get('experiment_name', 'hyperparameter_tuning')
        
        # MLflow 실험 설정
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        logger.info(f"MLflow 실험 설정: {experiment_name} (ID: {self.experiment_id})")
        
        # 현재 run ID는 나중에 설정됨
        self.run_id = None

    def load_data(self) -> pd.DataFrame:
        """
        데이터를 로드합니다.
        
        Returns:
            로드된 데이터프레임
        """
        logger.info(f"데이터 로드 중: {self.data_path}")
        
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
        
        if self.nrows:
            data = pd.read_csv(self.data_path, nrows=self.nrows)
            logger.info(f"테스트용 데이터 로드: {len(data):,} 행")
        else:
            data = pd.read_csv(self.data_path)
            logger.info(f"전체 데이터 로드: {len(data):,} 행")
        
        return data


def setup_tuning_logger(log_filename=None):
    """
    튜닝 로그를 파일로 저장하기 위한 로거를 설정합니다.
    """
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"tuning_log_{timestamp}.txt"
    
    # results 디렉토리 생성
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / log_filename
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 루트 로거에 핸들러 추가
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    return log_path

def save_tuning_log(result, model_type, experiment_type, nrows=None, experiment_id=None, run_id=None, error_msg=None, log_file_path=None):
    """
    튜닝 결과 및 로그를 results 폴더에 txt로 저장합니다.
    """
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 이미 로그 파일이 있다면 그대로 사용, 없다면 새로 생성
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"tuning_log_{timestamp}.txt"
        log_file_path = results_dir / log_filename
    
    # result가 tuple이면 dict로 변환
    result_dict = None
    if isinstance(result, tuple) and len(result) == 2:
        best_params, best_score = result
        result_dict = {'best_params': best_params, 'best_score': best_score}
    elif isinstance(result, dict):
        result_dict = result
    else:
        result_dict = {}
    
    # 결과 요약을 로그 파일에 추가
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("튜닝 결과 요약\n")
        f.write("="*50 + "\n")
        
        f.write(f"모델 타입: {model_type}\n")
        f.write(f"실험 타입: {experiment_type}\n")
        if nrows:
            f.write(f"사용 데이터 행 수: {nrows:,}\n")
        
        if experiment_id and run_id:
            mlflow_link = f"http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}"
            f.write(f"MLflow 링크: {mlflow_link}\n")
        
        if error_msg:
            f.write(f"에러 메시지: {error_msg}\n")
        elif result_dict:
            f.write(f"최고 성능: {result_dict.get('best_score', 'N/A')}\n")
            f.write(f"최적 파라미터: {result_dict.get('best_params', 'N/A')}\n")
            f.write(f"튜닝 시간: {result_dict.get('tuning_time', 'N/A')}\n")
        
        f.write("="*50 + "\n")
    
    print(f"튜닝 로그가 저장되었습니다: {log_file_path}")


def main():
    """메인 함수 - 튜닝 실행 예시"""
    import argparse
    
    parser = argparse.ArgumentParser(description="하이퍼파라미터 튜닝 실행")
    parser.add_argument("--tuning_config", type=str, default="configs/hyperparameter_tuning.yaml",
                       help="튜닝 설정 파일 경로")
    parser.add_argument("--base_config", type=str, default="configs/default_config.yaml",
                       help="기본 설정 파일 경로")
    
    args = parser.parse_args()
    
    # 튜너 생성 및 최적화 실행
    tuner = HyperparameterTuner(args.tuning_config, args.base_config)
    best_params, best_score = tuner.optimize()
    
    # 결과 저장
    tuner.save_results()
    
    # 튜닝 로그 저장
    save_tuning_log(best_params, tuner.config['model']['model_type'], "hyperparameter_tuning",
                    tuner.nrows, tuner.study.number, tuner.study.id)
    
    print(f"\n=== 최적화 완료 ===")
    print(f"최고 성능: {best_score:.4f}")
    print(f"최적 파라미터: {best_params}")


if __name__ == "__main__":
    main() 