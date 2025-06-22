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

# 프로젝트 루트를 Python 경로에 추가
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.splits import load_config, split_test_set, get_cv_splits
from src.preprocessing import transform_data, fit_preprocessing_pipeline
from src.feature_engineering import transform_features, get_feature_columns, get_target_columns_from_data
from src.models.xgboost_model import XGBoostModel
from src.evaluation import calculate_all_metrics

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Optuna를 사용한 하이퍼파라미터 튜닝 클래스
    
    이 클래스는 MLflow와 연동되어 실험 추적 및 결과 저장을 지원합니다.
    """
    
    def __init__(self, tuning_config_path: str, base_config_path: str = None):
        """
        하이퍼파라미터 튜너를 초기화합니다.
        
        Args:
            tuning_config_path: 하이퍼파라미터 튜닝 설정 파일 경로
            base_config_path: 기본 설정 파일 경로 (선택사항)
        """
        self.tuning_config_path = tuning_config_path
        self.base_config_path = base_config_path or self._get_base_config_path()
        
        # 설정 로드
        self.tuning_config = self._load_tuning_config()
        self.config = self._merge_configs()
        
        # 기본 데이터 파일 경로 (디버깅용으로 오버라이드 가능)
        self.data_path = "data/processed/processed_data_with_features.csv"
        
        # Optuna study 및 결과 저장용
        self.study = None
        self.best_params = None
        self.best_score = None
        
        logger.info("하이퍼파라미터 튜너 초기화 완료")
        logger.info(f"  - 튜닝 설정: {tuning_config_path}")
        logger.info(f"  - 기본 설정: {self.base_config_path}")
        logger.info(f"  - 최적화 방향: {self.tuning_config['tuning']['direction']}")
        logger.info(f"  - 튜닝 횟수: {self.tuning_config['tuning']['n_trials']}")
    
    def _get_base_config_path(self) -> str:
        """기본 설정 파일 경로를 반환합니다."""
        return "configs/default_config.yaml"
    
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
            제안된 하이퍼파라미터 딕셔너리
        """
        params = {}
        xgboost_params = self.tuning_config['xgboost_params']
        
        for param_name, param_config in xgboost_params.items():
            param_type = param_config['type']
            # YAML에서 파싱된 값들을 명시적으로 숫자 타입으로 변환
            low = float(param_config['low']) if param_type == "float" else int(param_config['low'])
            high = float(param_config['high']) if param_type == "float" else int(param_config['high'])
            log = param_config.get('log', False)
            
            if param_type == "int":
                params[param_name] = trial.suggest_int(param_name, low, high, log=log)
            elif param_type == "float":
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
        
        return params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna 목적 함수
        
        주어진 하이퍼파라미터로 모델을 학습하고 성능을 평가합니다.
        """
        # 하이퍼파라미터 제안
        params = self._suggest_parameters(trial)
        
        # MLflow 중첩 실행 시작
        with mlflow.start_run(nested=True):
            # 파라미터 로깅
            mlflow.log_params(params)
            mlflow.log_param("trial_number", trial.number)
            
            try:
                # 데이터 로드 및 분할
                df = pd.read_csv(self.data_path)
                
                # 테스트 세트 분리
                train_val_df, test_df, _ = split_test_set(df, self.config)
                
                # 교차 검증 분할
                cv_splits = get_cv_splits(train_val_df, self.config)
                
                # 교차 검증으로 성능 평가
                cv_scores = []
                
                for fold_idx, (train_fold_df, val_fold_df, fold_info) in enumerate(cv_splits):
                    train_df = train_fold_df
                    val_df = val_fold_df
                    # 전처리 (훈련 데이터로만 fit)
                    from src.preprocessing import fit_preprocessing_pipeline
                    preprocessor, _ = fit_preprocessing_pipeline(train_df, self.config)
                    
                    train_processed = transform_data(train_df, preprocessor, self.config)
                    val_processed = transform_data(val_df, preprocessor, self.config)
                    
                    # 피처 엔지니어링
                    from src.feature_engineering import fit_feature_engineering
                    feature_info = fit_feature_engineering(train_processed, self.config)
                    
                    train_engineered = transform_features(train_processed, feature_info, self.config)
                    val_engineered = transform_features(val_processed, feature_info, self.config)
                    
                    # 피처와 타겟 분리
                    feature_columns = get_feature_columns(train_engineered, self.config)
                    target_columns = get_target_columns_from_data(train_engineered, self.config)
                    
                    X_train = train_engineered[feature_columns]
                    y_train = train_engineered[target_columns]
                    X_val = val_engineered[feature_columns]
                    y_val = val_engineered[target_columns]
                    
                    # 모델 학습
                    model_config = self.config.copy()
                    model_config['model']['xgboost'].update(params)
                    
                    model = XGBoostModel(model_config)
                    model.fit(X_train, y_train, X_val, y_val)
                    
                    # 예측 및 평가
                    val_predictions = model.predict(X_val)
                    val_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                    
                    metrics = calculate_all_metrics(y_val, val_predictions, val_proba, self.config)
                    
                    # 주요 지표 추출
                    primary_metric = self.tuning_config['evaluation']['primary_metric']
                    if primary_metric in metrics:
                        cv_scores.append(metrics[primary_metric])
                    else:
                        logger.warning(f"주요 지표 {primary_metric}을 찾을 수 없습니다. f1_score를 사용합니다.")
                        cv_scores.append(metrics.get('f1_score', 0.0))
                
                # 평균 성능 계산
                mean_score = np.mean(cv_scores)
                
                # MLflow에 결과 로깅
                mlflow.log_metric(f"cv_{primary_metric}_mean", mean_score)
                mlflow.log_metric(f"cv_{primary_metric}_std", np.std(cv_scores))
                
                for fold_idx, score in enumerate(cv_scores):
                    mlflow.log_metric(f"fold_{fold_idx+1}_{primary_metric}", score)
                
                logger.info(f"Trial {trial.number}: {primary_metric} = {mean_score:.4f} ± {np.std(cv_scores):.4f}")
                
                return mean_score
                
            except Exception as e:
                logger.error(f"Trial {trial.number} 실패: {str(e)}")
                mlflow.log_param("error", str(e))
                return float('-inf') if self.tuning_config['tuning']['direction'] == 'maximize' else float('inf')
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        하이퍼파라미터 최적화를 실행합니다.
        
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
        
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name=f"xgboost_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # 최적화 실행
        n_trials = self.tuning_config['tuning']['n_trials']
        n_jobs = self.tuning_config['tuning']['n_jobs']
        
        with mlflow.start_run():
            # 실험 정보 로깅
            mlflow.log_param("optimization_algorithm", self.tuning_config['sampler']['type'])
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("direction", direction)
            mlflow.log_param("primary_metric", self.tuning_config['evaluation']['primary_metric'])
            
            # 최적화 실행
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
            
            # 최적 결과 저장
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            
            # MLflow에 최적 결과 로깅
            mlflow.log_params(self.best_params)
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
        model_config['model']['xgboost'].update(self.best_params)
        
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
        
        X_train = train_engineered[feature_columns]
        y_train = train_engineered[target_columns]
        
        model = XGBoostModel(model_config)
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
        optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=axes[0, 0])
        axes[0, 0].set_title("Optimization History")
        
        # 파라미터 중요도
        optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[0, 1])
        axes[0, 1].set_title("Parameter Importances")
        
        # 파라미터 관계
        optuna.visualization.matplotlib.plot_parallel_coordinate(self.study, ax=axes[1, 0])
        axes[1, 0].set_title("Parallel Coordinate Plot")
        
        # 파라미터 분포
        optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[1, 1])
        axes[1, 1].set_title("Parameter Importances (Detailed)")
        
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
    
    print(f"\n=== 최적화 완료 ===")
    print(f"최고 성능: {best_score:.4f}")
    print(f"최적 파라미터: {best_params}")


if __name__ == "__main__":
    main() 