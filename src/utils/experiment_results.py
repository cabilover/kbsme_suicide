"""
실험 결과 저장 유틸리티

실험 결과를 파일로 저장하는 공통 함수들을 제공합니다.
"""

import os
import json
import psutil
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def save_experiment_results(
    result: Tuple[Dict[str, Any], float], 
    model_type: str, 
    experiment_type: str, 
    nrows: Optional[int] = None, 
    experiment_id: Optional[str] = None, 
    run_id: Optional[str] = None, 
    config: Optional[Dict[str, Any]] = None, 
    data_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    실험 결과를 파일로 저장합니다.
    
    Args:
        result: 실험 결과 (best_params, best_score) 튜플
        model_type: 모델 타입
        experiment_type: 실험 타입
        nrows: 사용된 데이터 행 수
        experiment_id: MLflow 실험 ID
        run_id: MLflow run ID
        config: 설정 딕셔너리
        data_info: 데이터 정보 딕셔너리
        
    Returns:
        저장된 파일 경로
    """
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
    
    # MLflow에서 상세 메트릭 추출
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
                elif method == 'time_series_adapted':
                    time_series_config = resampling_config.get('time_series_adapted', {})
                    f.write(f"시계열 특화 리샘플링:\n")
                    f.write(f"  - 시간 가중치: {time_series_config.get('time_weight', 0.3)}\n")
                    f.write(f"  - 시간적 윈도우: {time_series_config.get('temporal_window', 3)}\n")
                    f.write(f"  - 계절성 가중치: {time_series_config.get('seasonality_weight', 0.2)}\n")
                    f.write(f"  - 패턴 보존: {time_series_config.get('pattern_preservation', True)}\n")
                    f.write(f"  - 추세 보존: {time_series_config.get('trend_preservation', True)}\n")
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
        
        # === 6. 상세 성능 지표 (최적 파라미터) ===
        f.write("=== 상세 성능 지표 (최적 파라미터) ===\n")
        if detailed_metrics:
            # 메트릭 카테고리화
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
                    # 각 카테고리별로 상위 10개만 표시
                    for metric_str in metrics[:10]:
                        f.write(f"{metric_str}\n")
                    if len(metrics) > 10:
                        f.write(f"  ... (총 {len(metrics)}개 중 상위 10개 표시)\n")
                    f.write("\n")
        else:
            f.write("상세 메트릭 정보: MLflow에서 추출할 수 없음\n")
        f.write("\n")
        
        # === 7. 하이퍼파라미터 튜닝 과정 ===
        f.write("=== 하이퍼파라미터 튜닝 과정 ===\n")
        if config:
            # 튜닝 설정 정보
            tuning_config = config.get('tuning', {})
            f.write("튜닝 설정:\n")
            f.write(f"  - 총 시도 횟수: {tuning_config.get('n_trials', 'N/A')}\n")
            f.write(f"  - 최적화 방향: {tuning_config.get('direction', 'N/A')}\n")
            f.write(f"  - 주요 메트릭: {tuning_config.get('metric', 'N/A')}\n")
            f.write(f"  - 타임아웃: {tuning_config.get('timeout', 'N/A')}초\n")
            
            # 튜닝 범위 정보
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
        
        # === 8. 교차 검증 상세 결과 ===
        f.write("=== 교차 검증 상세 결과 ===\n")
        if detailed_metrics:
            # 폴드별 성능 추출
            fold_metrics = {}
            cv_stats = {}
            
            # 폴드별 메트릭 수집
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
                    # 주요 통계만 표시
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
        
        # === 9. 모델 특성 분석 ===
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
        
        # === 12. 실험 결과 요약 ===
        f.write("=== 실험 결과 요약 ===\n")
        if isinstance(result, tuple) and len(result) == 2:
            # 일반 하이퍼파라미터 튜닝 결과
            best_params, best_score = result
            f.write(f"🎯 최고 성능: {best_score:.4f}\n")
            if config and 'tuning' in config:
                tuning_config = config['tuning']
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
    return str(filename) 