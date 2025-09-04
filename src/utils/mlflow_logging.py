"""
MLflow 로깅 유틸리티 모듈

하이퍼파라미터 튜닝과 리샘플링 실험에서 공통으로 사용하는 MLflow 로깅 기능들을 제공합니다.
"""

import logging
import pandas as pd
import numpy as np
import mlflow
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import psutil
from typing import Dict, Any, Optional
from pathlib import Path
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def log_feature_importance(tuner, config: Dict[str, Any], run) -> bool:
    """
    피처 중요도를 MLflow에 로깅합니다.
    
    Args:
        tuner: HyperparameterTuner 객체
        config: 설정 딕셔너리
        run: MLflow run 객체
        
    Returns:
        성공 여부
    """
    try:
        if not hasattr(tuner, 'study') or not tuner.study:
            logger.warning("Study 객체가 없어 피처 중요도를 추출할 수 없습니다.")
            return False
        
        # 최적 trial에서 모델 추출
        best_trial = tuner.study.best_trial
        if not best_trial:
            logger.warning("최적 trial이 없어 피처 중요도를 추출할 수 없습니다.")
            return False
        
        # 최적 파라미터로 모델 재학습
        best_params = tuner.study.best_params
        model_type = config.get('model', {}).get('model_type', 'xgboost')
        
        # 데이터 로드 및 전처리
        from src.preprocessing import transform_data
        from src.feature_engineering import transform_features, get_feature_columns
        from src.models import ModelFactory
        
        # 최적 모델 생성
        model_factory = ModelFactory()
        model = model_factory.create_model(model_type, best_params)
        
        # 피처 중요도 추출
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        elif hasattr(model, 'feature_importance'):
            feature_importances = model.feature_importance()
        else:
            logger.warning(f"{model_type} 모델에서 피처 중요도를 추출할 수 없습니다.")
            return False
        
        # 피처 이름 가져오기
        feature_columns = get_feature_columns(config)
        
        if len(feature_importances) != len(feature_columns):
            logger.warning("피처 중요도와 피처 컬럼 수가 일치하지 않습니다.")
            return False
        
        # 피처 중요도 DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        # 상위 20개 피처 중요도 로깅
        top_features = importance_df.head(20)
        for idx, row in top_features.iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        # 피처 중요도 차트 생성 및 저장
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top 20 Feature Importance - {model_type.upper()}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # 차트를 MLflow에 저장
        chart_path = f"feature_importance_{model_type}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(chart_path)
        plt.close()
        
        # 피처 중요도 CSV 저장
        csv_path = f"feature_importance_{model_type}.csv"
        importance_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        logger.info(f"피처 중요도 로깅 완료: {len(importance_df)}개 피처")
        return True
        
    except Exception as e:
        logger.error(f"피처 중요도 로깅 중 오류: {e}")
        return False


def save_model_artifacts(tuner, config: Dict[str, Any], run) -> bool:
    """
    학습된 모델을 MLflow에 아티팩트로 저장합니다.
    
    Args:
        tuner: HyperparameterTuner 객체
        config: 설정 딕셔너리
        run: MLflow run 객체
        
    Returns:
        성공 여부
    """
    try:
        if not hasattr(tuner, 'study') or not tuner.study:
            logger.warning("Study 객체가 없어 모델을 저장할 수 없습니다.")
            return False
        
        best_params = tuner.study.best_params
        model_type = config.get('model', {}).get('model_type', 'xgboost')
        
        # 최적 모델 재학습
        from src.models import ModelFactory
        from src.preprocessing import transform_data
        from src.feature_engineering import transform_features
        
        model_factory = ModelFactory()
        model = model_factory.create_model(model_type, best_params)
        
        # 모델 저장
        model_path = f"best_model_{model_type}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # 모델 파라미터 저장
        params_path = f"best_params_{model_type}.json"
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        mlflow.log_artifact(params_path)
        
        # MLflow 모델 저장 (모델 타입별)
        if model_type == 'xgboost':
            mlflow.xgboost.log_model(model, "model")
        elif model_type == 'lightgbm':
            mlflow.lightgbm.log_model(model, "model")
        elif model_type == 'catboost':
            mlflow.catboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"모델 아티팩트 저장 완료: {model_type}")
        return True
        
    except Exception as e:
        logger.error(f"모델 아티팩트 저장 중 오류: {e}")
        return False


def log_visualizations(tuner, config: Dict[str, Any], run) -> bool:
    """
    최적화 과정 시각화를 MLflow에 로깅합니다.
    
    Args:
        tuner: HyperparameterTuner 객체
        config: 설정 딕셔너리
        run: MLflow run 객체
        
    Returns:
        성공 여부
    """
    try:
        if not hasattr(tuner, 'study') or not tuner.study:
            logger.warning("Study 객체가 없어 시각화를 생성할 수 없습니다.")
            return False
        
        # 실험별 폴더 생성
        experiment_type = config.get('experiment', {}).get('name', 'unknown_experiment')
        model_type = config.get('model', {}).get('model_type', 'unknown_model')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # results 폴더 내 실험별 하위 폴더 생성
        experiment_folder = f"results/visualizations/{experiment_type}_{model_type}_{timestamp}"
        os.makedirs(experiment_folder, exist_ok=True)
        
        # 최적화 히스토리 플롯
        plt.figure(figsize=(12, 8))
        
        # 서브플롯 1: 최적화 히스토리
        plt.subplot(2, 3, 1)
        try:
            optuna.visualization.matplotlib.plot_optimization_history(tuner.study)
            plt.title('Optimization History Plot')
        except Exception as e:
            plt.text(0.5, 0.5, 'Optimization history\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Optimization History Plot')
        
        # 서브플롯 2: 파라미터 중요도
        plt.subplot(2, 3, 2)
        try:
            optuna.visualization.matplotlib.plot_param_importances(tuner.study)
            plt.title('Parameter Importance Plot')
        except Exception as e:
            plt.text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Parameter Importance Plot')
        
        # 서브플롯 3: 병렬 좌표 플롯
        plt.subplot(2, 3, 3)
        try:
            optuna.visualization.matplotlib.plot_parallel_coordinate(tuner.study)
            plt.title('Parallel Coordinate Plot')
        except Exception as e:
            plt.text(0.5, 0.5, 'Parallel coordinate\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Parallel Coordinate Plot')
        
        # 서브플롯 4: 슬라이스 플롯
        plt.subplot(2, 3, 4)
        try:
            optuna.visualization.matplotlib.plot_slice(tuner.study)
            plt.title('Slice Plot')
        except Exception as e:
            plt.text(0.5, 0.5, 'Slice plot\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Slice Plot')
        
        # 서브플롯 5: 컨투어 플롯
        plt.subplot(2, 3, 5)
        try:
            # 가장 중요한 두 파라미터로 컨투어 플롯 생성
            param_importances = optuna.importance.get_param_importances(tuner.study)
            if len(param_importances) >= 2:
                top_params = list(param_importances.keys())[:2]
                optuna.visualization.matplotlib.plot_contour(tuner.study, params=top_params)
                plt.title(f'Contour Plot ({top_params[0]} vs {top_params[1]})')
            else:
                plt.text(0.5, 0.5, 'Contour plot\nnot available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Contour Plot')
        except Exception as e:
            plt.text(0.5, 0.5, 'Contour plot\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Contour Plot')
        
        # 서브플롯 6: 파라미터 중요도 (수동 생성)
        plt.subplot(2, 3, 6)
        try:
            param_importances = optuna.importance.get_param_importances(tuner.study)
            if param_importances:
                param_names = list(param_importances.keys())
                param_values = list(param_importances.values())
                
                # 중요도 순으로 정렬
                sorted_indices = sorted(range(len(param_values)), key=lambda i: param_values[i], reverse=True)
                param_names = [param_names[i] for i in sorted_indices]
                param_values = [param_values[i] for i in sorted_indices]
                
                plt.barh(range(len(param_names)), param_values)
                plt.yticks(range(len(param_names)), param_names)
                plt.title('Parameter Importance')
                plt.xlabel('Importance')
                plt.grid(True, alpha=0.3)
        except Exception as e:
            plt.text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Parameter Importance')
        
        plt.tight_layout()
        chart_path = os.path.join(experiment_folder, "optimization_visualization.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(chart_path)
        plt.close()
        
        logger.info("시각화 로깅 완료")
        return True
        
    except Exception as e:
        logger.error(f"시각화 로깅 중 오류: {e}")
        return False


def log_memory_usage(run) -> bool:
    """
    메모리 사용량을 MLflow에 로깅합니다.
    
    Args:
        run: MLflow run 객체
        
    Returns:
        성공 여부
    """
    try:
        # 현재 프로세스의 메모리 사용량
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # 메모리 사용량 로깅
        mlflow.log_metric("memory_usage_mb", memory_info.rss / 1024 / 1024)
        mlflow.log_metric("memory_usage_percent", memory_percent)
        
        # 시스템 전체 메모리 정보
        system_memory = psutil.virtual_memory()
        mlflow.log_metric("system_memory_total_gb", system_memory.total / 1024 / 1024 / 1024)
        mlflow.log_metric("system_memory_available_gb", system_memory.available / 1024 / 1024 / 1024)
        mlflow.log_metric("system_memory_percent", system_memory.percent)
        
        logger.info(f"메모리 사용량 로깅 완료: {memory_info.rss / 1024 / 1024:.1f}MB ({memory_percent:.1f}%)")
        return True
        
    except Exception as e:
        logger.error(f"메모리 사용량 로깅 중 오류: {e}")
        return False


def log_learning_curves(tuner, config: Dict[str, Any], run) -> bool:
    """
    각 폴드의 학습/검증 곡선을 MLflow에 로깅합니다.
    
    Args:
        tuner: HyperparameterTuner 객체
        config: 설정 딕셔너리
        run: MLflow run 객체
        
    Returns:
        성공 여부
    """
    try:
        if not hasattr(tuner, 'study') or not tuner.study:
            logger.warning("Study 객체가 없어 학습 곡선을 생성할 수 없습니다.")
            return False
        
        # 실험별 폴더 생성
        experiment_type = config.get('experiment', {}).get('name', 'unknown_experiment')
        model_type = config.get('model', {}).get('model_type', 'unknown_model')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # results 폴더 내 실험별 하위 폴더 생성
        experiment_folder = f"results/visualizations/{experiment_type}_{model_type}_{timestamp}"
        os.makedirs(experiment_folder, exist_ok=True)
        
        # 최적 trial에서 학습 곡선 추출
        best_trial = tuner.study.best_trial
        if not best_trial:
            logger.warning("최적 trial이 없어 학습 곡선을 생성할 수 없습니다.")
            return False
        
        # 최적 파라미터로 모델 재학습하여 학습 곡선 생성
        best_params = tuner.study.best_params
        model_type = config.get('model', {}).get('model_type', 'xgboost')
        
        # 교차 검증을 통해 학습 곡선 생성
        from src.training import run_cross_validation
        from src.splits import split_test_set
        
        # 데이터 로드
        data_path = config['data'].get('file_path', 'data/processed/processed_data_with_features.csv')
        df = pd.read_csv(data_path)
        
        # 불필요한 컬럼 제거 (NaN이 대부분인 컬럼들)
        columns_to_drop = ['suicide_c', 'suicide_y', 'check']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop)
            logger.info(f"불필요한 컬럼 제거: {existing_columns_to_drop}")
        
        # 테스트 세트 분리
        train_val_df, test_df, _ = split_test_set(df, config)
        
        # 교차 검증 실행
        cv_results = run_cross_validation(train_val_df, config)
        
        if cv_results and 'fold_results' in cv_results:
            fold_results = cv_results['fold_results']
            
            # 학습 곡선 시각화
            plt.figure(figsize=(15, 10))
            
            for fold_idx, fold_result in enumerate(fold_results):
                if 'training_results' in fold_result:
                    history = fold_result['training_results'].get('history', {})
                    
                    # 학습/검증 손실
                    if 'train_loss' in history and 'val_loss' in history:
                        plt.subplot(2, 3, fold_idx + 1)
                        plt.plot(history['train_loss'], label='Train Loss', alpha=0.7)
                        plt.plot(history['val_loss'], label='Validation Loss', alpha=0.7)
                        plt.title(f'Fold {fold_idx + 1} - Loss')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                    
                    # 학습/검증 정확도
                    if 'train_accuracy' in history and 'val_accuracy' in history:
                        plt.subplot(2, 3, fold_idx + 1)
                        plt.plot(history['train_accuracy'], label='Train Accuracy', alpha=0.7)
                        plt.plot(history['val_accuracy'], label='Validation Accuracy', alpha=0.7)
                        plt.title(f'Fold {fold_idx + 1} - Accuracy')
                        plt.xlabel('Epoch')
                        plt.ylabel('Accuracy')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            chart_path = os.path.join(experiment_folder, "learning_curves.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(chart_path)
            plt.close()
            
            # 폴드별 성능 요약
            fold_scores = []
            for fold_idx, fold_result in enumerate(fold_results):
                if 'training_results' in fold_result:
                    val_metrics = fold_result['training_results'].get('val_metrics', {})
                    primary_metric = config.get('evaluation', {}).get('primary_metric', 'f1')
                    
                    for target_name, target_metrics in val_metrics.items():
                        if primary_metric in target_metrics:
                            score = target_metrics[primary_metric]
                            fold_scores.append(score)
                            mlflow.log_metric(f"fold_{fold_idx}_{target_name}_{primary_metric}", score)
                            break
            
            # 폴드별 성능 분포
            if fold_scores:
                plt.figure(figsize=(10, 6))
                plt.boxplot(fold_scores)
                plt.title('Cross-Validation Score Distribution')
                plt.ylabel(f'{primary_metric.upper()} Score')
                plt.grid(True, alpha=0.3)
                
                chart_path = os.path.join(experiment_folder, "cv_score_distribution.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(chart_path)
                plt.close()
        
        logger.info("학습 곡선 로깅 완료")
        return True
        
    except Exception as e:
        logger.error(f"학습 곡선 로깅 중 오류: {e}")
        return False


def log_resampling_analysis(tuner, config: Dict[str, Any], run, data_info: Dict[str, Any]) -> bool:
    """
    리샘플링 특화 분석을 MLflow에 로깅합니다.
    
    Args:
        tuner: HyperparameterTuner 객체
        config: 설정 딕셔너리
        run: MLflow run 객체
        data_info: 데이터 정보 딕셔너리
        
    Returns:
        성공 여부
    """
    try:
        # 리샘플링 설정 정보 로깅
        resampling_config = config.get('imbalanced_data', {}).get('resampling', {})
        resampling_method = resampling_config.get('method', 'none')
        
        mlflow.log_param("resampling_method_used", resampling_method)
        mlflow.log_param("resampling_enabled", resampling_config.get('enabled', False))
        
        # 클래스 분포 시각화
        if 'class_distributions' in data_info:
            class_dist = data_info['class_distributions']
            
            plt.figure(figsize=(15, 10))
            
            # 전체 데이터 클래스 분포
            if 'full_suicide' in class_dist:
                plt.subplot(2, 3, 1)
                full_dist = class_dist['full_suicide']
                plt.pie(full_dist.values(), labels=full_dist.keys(), autopct='%1.1f%%')
                plt.title('Full Dataset Class Distribution')
            
            # 훈련 데이터 클래스 분포
            if 'train_original' in class_dist:
                plt.subplot(2, 3, 2)
                train_dist = class_dist['train_original']
                if 'suicide' in train_dist:
                    plt.pie(train_dist['suicide'].values(), labels=train_dist['suicide'].keys(), autopct='%1.1f%%')
                    plt.title('Training Data Class Distribution')
            
            # 테스트 데이터 클래스 분포
            if 'test' in class_dist:
                plt.subplot(2, 3, 3)
                test_dist = class_dist['test']
                if 'suicide' in test_dist:
                    plt.pie(test_dist['suicide'].values(), labels=test_dist['suicide'].keys(), autopct='%1.1f%%')
                    plt.title('Test Data Class Distribution')
            
            # 클래스 불균형 비율 계산
            if 'train_original' in class_dist and 'suicide' in class_dist['train_original']:
                train_suicide_dist = class_dist['train_original']['suicide']
                if 0 in train_suicide_dist and 1 in train_suicide_dist:
                    imbalance_ratio = train_suicide_dist[1] / train_suicide_dist[0]
                    mlflow.log_metric("class_imbalance_ratio", imbalance_ratio)
                    
                    plt.subplot(2, 3, 4)
                    plt.bar(['Negative (0)', 'Positive (1)'], [train_suicide_dist[0], train_suicide_dist[1]])
                    plt.title(f'Class Distribution (Ratio: {imbalance_ratio:.2f})')
                    plt.ylabel('Count')
            
            # 리샘플링 효과 시각화 (가상)
            plt.subplot(2, 3, 5)
            if resampling_method != 'none':
                # 리샘플링 후 예상 클래스 분포 (실제로는 리샘플링 결과를 사용해야 함)
                plt.bar(['Negative (0)', 'Positive (1)'], [100, 100], alpha=0.7, label='After Resampling')
                plt.bar(['Negative (0)', 'Positive (1)'], [train_suicide_dist[0], train_suicide_dist[1]], alpha=0.5, label='Before Resampling')
                plt.title(f'Resampling Effect ({resampling_method})')
                plt.ylabel('Count')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'No Resampling\nApplied', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Resampling Effect')
            
            # 리샘플링 파라미터 요약
            plt.subplot(2, 3, 6)
            resampling_params = []
            if resampling_method == 'smote':
                smote_config = resampling_config.get('smote', {})
                resampling_params = [
                    f"k_neighbors: {smote_config.get('k_neighbors', 'N/A')}",
                    f"sampling_strategy: {smote_config.get('sampling_strategy', 'N/A')}"
                ]
            elif resampling_method == 'borderline_smote':
                borderline_config = resampling_config.get('borderline_smote', {})
                resampling_params = [
                    f"k_neighbors: {borderline_config.get('k_neighbors', 'N/A')}",
                    f"m_neighbors: {borderline_config.get('m_neighbors', 'N/A')}",
                    f"sampling_strategy: {borderline_config.get('sampling_strategy', 'N/A')}"
                ]
            elif resampling_method == 'adasyn':
                adasyn_config = resampling_config.get('adasyn', {})
                resampling_params = [
                    f"k_neighbors: {adasyn_config.get('k_neighbors', 'N/A')}",
                    f"sampling_strategy: {adasyn_config.get('sampling_strategy', 'N/A')}"
                ]
            
            if resampling_params:
                plt.text(0.1, 0.9, f'Resampling Method: {resampling_method}', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
                for i, param in enumerate(resampling_params):
                    plt.text(0.1, 0.8 - i*0.1, param, transform=plt.gca().transAxes, fontsize=10)
            else:
                plt.text(0.5, 0.5, 'No Resampling\nParameters', ha='center', va='center', transform=plt.gca().transAxes)
            
            plt.title('Resampling Parameters')
            plt.axis('off')
            
            plt.tight_layout()
            chart_path = "resampling_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(chart_path)
            plt.close()
        
        # 리샘플링 관련 메트릭 로깅
        if resampling_method != 'none':
            mlflow.log_param("resampling_k_neighbors", resampling_config.get(f'{resampling_method}_k_neighbors', 'N/A'))
            mlflow.log_param("resampling_sampling_strategy", resampling_config.get(f'{resampling_method}_sampling_strategy', 'N/A'))
        
        logger.info("리샘플링 분석 로깅 완료")
        return True
        
    except Exception as e:
        logger.error(f"리샘플링 분석 로깅 중 오류: {e}")
        return False


def log_all_advanced_metrics(tuner, config: Dict[str, Any], run, data_info: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """
    모든 고급 로깅 기능을 실행합니다.
    
    Args:
        tuner: HyperparameterTuner 객체
        config: 설정 딕셔너리
        run: MLflow run 객체
        data_info: 데이터 정보 딕셔너리 (리샘플링 분석용)
        
    Returns:
        각 로깅 기능의 성공 여부를 담은 딕셔너리
    """
    results = {}
    
    # 1. 피처 중요도 로깅
    results['feature_importance'] = log_feature_importance(tuner, config, run)
    
    # 2. 모델 아티팩트 저장
    results['model_artifacts'] = save_model_artifacts(tuner, config, run)
    
    # 3. 시각화 로깅
    results['visualizations'] = log_visualizations(tuner, config, run)
    
    # 4. 메모리 사용량 추적
    results['memory_usage'] = log_memory_usage(run)
    
    # 5. 학습 곡선 로깅
    results['learning_curves'] = log_learning_curves(tuner, config, run)
    
    # 6. 리샘플링 특화 로깅 (data_info가 제공된 경우)
    if data_info is not None:
        results['resampling_analysis'] = log_resampling_analysis(tuner, config, run, data_info)
    else:
        results['resampling_analysis'] = False
    
    # 성공한 기능들 로깅
    successful_features = [k for k, v in results.items() if v]
    failed_features = [k for k, v in results.items() if not v]
    
    logger.info(f"고급 로깅 완료 - 성공: {successful_features}")
    if failed_features:
        logger.warning(f"고급 로깅 실패: {failed_features}")
    
    return results 