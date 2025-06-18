"""
모델 학습 모듈

이 모듈은 모델 학습 루프와 Early Stopping 로직을 구현합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import mlflow
from sklearn.model_selection import train_test_split
import warnings
from src.feature_engineering import get_target_columns_from_data
from src.utils import find_column_with_remainder, safe_feature_name

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                X_val: pd.DataFrame, y_val: pd.DataFrame,
                config: Dict[str, Any], fold_info: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    모델을 학습합니다.
    
    Args:
        X_train: 훈련 피처 데이터
        y_train: 훈련 타겟 데이터
        X_val: 검증 피처 데이터
        y_val: 검증 타겟 데이터
        config: 설정 딕셔너리
        fold_info: 폴드 정보
        
    Returns:
        학습된 모델과 학습 결과
    """
    from src.models.xgboost_model import XGBoostModel
    
    logger.info("모델 학습 시작")
    
    # === 타겟 결측치가 있는 샘플 제거 ===
    train_targets = get_target_columns_from_data(y_train, config)
    val_targets = get_target_columns_from_data(y_val, config)
    train_notnull = y_train[train_targets].notnull().all(axis=1)
    val_notnull = y_val[val_targets].notnull().all(axis=1)
    n_train_before = len(X_train)
    n_val_before = len(X_val)
    X_train, y_train = X_train.loc[train_notnull].reset_index(drop=True), y_train.loc[train_notnull].reset_index(drop=True)
    X_val, y_val = X_val.loc[val_notnull].reset_index(drop=True), y_val.loc[val_notnull].reset_index(drop=True)
    n_train_after = len(X_train)
    n_val_after = len(X_val)
    logger.info(f"결측 타겟 제거: train {n_train_before}->{n_train_after}, val {n_val_before}->{n_val_after}")
    # === END ===
    
    # 모델 초기화
    model = XGBoostModel(config)
    
    # Early Stopping 설정
    early_stopping = config['training']['early_stopping']
    patience = config['training']['patience']
    
    # 학습 수행 (Early Stopping 지원)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if early_stopping:
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
    
    # 검증 성능 평가
    val_predictions = model.predict(X_val)
    val_metrics = evaluate_predictions(y_val, val_predictions, config)
    
    # 피처 검증 결과 (첫 번째 폴드만, 원본 데이터에서 수행)
    feature_validation_results = None
    if fold_info.get('fold_idx') == 1 or fold_info.get('fold_year') == config['validation'].get('validation_start_year', 2021):
        # 원본 데이터로 피처 검증 수행 (전처리된 데이터가 아님)
        from src.feature_engineering import validate_existing_features
        # 원본 데이터는 fold_info에서 가져와야 하지만, 여기서는 간단히 건너뛰기
        logger.info("피처 검증은 원본 데이터에서 수행되어야 하므로 건너뜁니다.")
    
    # 피처 중요도 수집
    feature_importance = model.get_feature_importance()
    
    # 학습 결과 수집
    training_results = {
        'fold_info': fold_info,
        'val_metrics': val_metrics,
        'feature_importance': feature_importance,
        'model_params': config['model']['xgboost'],
        'early_stopping_used': early_stopping
    }
    
    if feature_validation_results is not None:
        training_results['feature_validation_results'] = feature_validation_results
    
    logger.info("모델 학습 완료")
    logger.info(f"검증 성능: {val_metrics}")
    
    return model, training_results


def evaluate_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame, 
                        config: Dict[str, Any]) -> Dict[str, float]:
    """
    예측 결과를 평가합니다.
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        config: 설정 딕셔너리
        
    Returns:
        평가 지표 딕셔너리
    """
    from src.evaluation import calculate_regression_metrics, calculate_classification_metrics
    
    metrics = {}
    all_pred_columns = list(y_pred.columns)
    
    # 타겟별로 평가
    for target in y_true.columns:
        # remainder__ 처리 일관성 적용
        pred_col = find_column_with_remainder(all_pred_columns, target.replace('remainder__', ''))
        if not pred_col:
            continue
        true_vals = y_true[target]
        pred_vals = y_pred[pred_col]
        
        # 전처리된 컬럼명에서 원본 타겟명 추출
        original_target = target.replace('remainder__', '') if target.startswith('remainder__') else target
        
        # 회귀 문제인지 분류 문제인지 판단
        if original_target.endswith('_next_year'):
            base_target = original_target.replace('_next_year', '')
            if base_target in ['anxiety_score', 'depress_score', 'sleep_score']:
                # 회귀 문제
                target_metrics = calculate_regression_metrics(true_vals, pred_vals)
                for metric_name, value in target_metrics.items():
                    metrics[f"{original_target}_{metric_name}"] = value
            elif base_target in ['suicide_t', 'suicide_a']:
                # 분류 문제
                target_metrics = calculate_classification_metrics(true_vals, pred_vals)
                for metric_name, value in target_metrics.items():
                    metrics[f"{original_target}_{metric_name}"] = value
        else:
            # 타겟 이름이 _next_year로 끝나지 않는 경우 기본적으로 분류로 처리
            target_metrics = calculate_classification_metrics(true_vals, pred_vals)
            for metric_name, value in target_metrics.items():
                metrics[f"{original_target}_{metric_name}"] = value
    
    return metrics


def log_training_results(training_results: Dict[str, Any], fold_count: int):
    """
    학습 결과를 MLflow에 로깅합니다.
    
    Args:
        training_results: 학습 결과
        fold_count: 폴드 번호
    """
    # 검증 메트릭 로깅
    val_metrics = training_results['val_metrics']
    for metric_name, value in val_metrics.items():
        mlflow.log_metric(f"fold_{fold_count}_{metric_name}", value)
    
    # Early Stopping 정보 로깅
    if training_results.get('early_stopping_used'):
        mlflow.log_metric(f"fold_{fold_count}_early_stopping_used", 1)
    else:
        mlflow.log_metric(f"fold_{fold_count}_early_stopping_used", 0)
    
    # 피처 중요도 로깅 (첫 번째 폴드만)
    if fold_count == 1 and training_results.get('feature_importance'):
        feature_importance = training_results['feature_importance']
        
        # 각 타겟별 피처 중요도 로깅
        for target, importance_df in feature_importance.items():
            # 상위 10개 피처만 로깅
            top_features = importance_df.head(10)
            for _, row in top_features.iterrows():
                # 피처 이름에서 특수문자 제거하여 MLflow 메트릭 이름으로 사용
                safe_feature_name = row['feature'].replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                mlflow.log_metric(
                    f"feature_importance_{target}_{safe_feature_name}", 
                    row['importance']
                )
        
        # 집계된 피처 중요도 로깅 (여러 타겟이 있는 경우)
        if len(feature_importance) > 1:
            try:
                from src.models.xgboost_model import XGBoostModel
                # 임시 모델 인스턴스 생성하여 집계 메서드 사용
                temp_model = XGBoostModel(training_results['model_params'])
                aggregated_importance = temp_model._aggregate_feature_importance(feature_importance)
                
                if 'aggregated' in aggregated_importance:
                    agg_df = aggregated_importance['aggregated']
                    top_agg_features = agg_df.head(10)
                    for _, row in top_agg_features.iterrows():
                        safe_feature_name = row['feature'].replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                        mlflow.log_metric(
                            f"aggregated_feature_importance_{safe_feature_name}", 
                            row['mean_importance']
                        )
                        mlflow.log_metric(
                            f"aggregated_feature_importance_std_{safe_feature_name}", 
                            row['std_importance']
                        )
            except Exception as e:
                logger.warning(f"집계된 피처 중요도 로깅 실패: {e}")
    
    # 피처 검증 결과 로깅 (첫 번째 폴드만)
    if fold_count == 1 and 'feature_validation_results' in training_results:
        validation_results = training_results['feature_validation_results']
        for feature, is_valid in validation_results.items():
            mlflow.log_metric(f"feature_validation_{feature}", int(is_valid))
    
    logger.info(f"폴드 {fold_count} 학습 결과 로깅 완료")


def run_cross_validation(train_val_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    교차 검증을 실행합니다.
    
    Args:
        train_val_df: 훈련/검증 데이터
        config: 설정 딕셔너리
        
    Returns:
        교차 검증 결과
    """
    from src.preprocessing import fit_preprocessing_pipeline, transform_data
    from src.feature_engineering import fit_feature_engineering, transform_features, get_feature_columns, get_target_columns_from_data
    
    logger.info("교차 검증 시작")
    
    # 교차 검증 결과 수집
    cv_results = {
        'fold_results': [],
        'aggregate_metrics': {},
        'best_model': None,
        'best_fold': None,
        'best_score': -np.inf,
        'early_stopping_used': False,
        'feature_validation_results': None
    }
    
    # 교차 검증 폴드 생성
    from src.splits import get_cv_splits
    
    fold_count = 0
    all_fold_metrics = []
    
    for train_fold_df, val_fold_df, fold_info in get_cv_splits(train_val_df, config):
        fold_count += 1
        logger.info(f"=== 폴드 {fold_count} 학습 ===")
        
        # 데이터 유출 방지를 위해 각 폴드의 훈련 데이터로만 전처리 파이프라인 학습
        preprocessor, _ = fit_preprocessing_pipeline(train_fold_df, config)
        
        # 피처 엔지니어링도 각 폴드의 훈련 데이터로만 학습
        _, feature_info = fit_feature_engineering(train_fold_df, config)
        
        # 전처리 적용
        train_processed = transform_data(train_fold_df, preprocessor, config)
        val_processed = transform_data(val_fold_df, preprocessor, config)
        
        # 피처 엔지니어링 적용
        train_engineered = transform_features(train_processed, feature_info, config)
        val_engineered = transform_features(val_processed, feature_info, config)
        
        # 피처와 타겟 분리
        feature_columns = get_feature_columns(train_engineered, config)
        target_columns = get_target_columns_from_data(train_engineered, config)
        
        X_train = train_engineered[feature_columns]
        y_train = train_engineered[target_columns]
        X_val = val_engineered[feature_columns]
        y_val = val_engineered[target_columns]
        
        # 모델 학습
        model, training_results = train_model(X_train, y_train, X_val, y_val, config, fold_info)
        
        # Early Stopping 사용 여부 추적
        if training_results.get('early_stopping_used'):
            cv_results['early_stopping_used'] = True
        
        # 피처 검증 결과 저장 (첫 번째 폴드만)
        if fold_count == 1 and 'feature_validation_results' in training_results:
            cv_results['feature_validation_results'] = training_results['feature_validation_results']
        
        # 결과 로깅
        log_training_results(training_results, fold_count)
        
        # 결과 수집
        cv_results['fold_results'].append({
            'fold_count': fold_count,
            'fold_info': fold_info,
            'training_results': training_results,
            'model': model,
            'preprocessor': preprocessor,
            'feature_info': feature_info
        })
        
        all_fold_metrics.append(training_results['val_metrics'])
        
        # 최고 성능 모델 추적 (평균 성능을 기준으로)
        if training_results['val_metrics']:
            # 모든 메트릭의 평균을 계산하여 성능 지표로 사용
            metric_values = list(training_results['val_metrics'].values())
            avg_score = np.mean([v for v in metric_values if not np.isnan(v)])
            
            if avg_score > cv_results['best_score']:
                cv_results['best_score'] = avg_score
                cv_results['best_fold'] = fold_count
                cv_results['best_model'] = model
                logger.info(f"새로운 최고 성능 모델 발견: 폴드 {fold_count} (점수: {avg_score:.4f})")
        
        logger.info(f"폴드 {fold_count} 완료")
    
    # 최고 성능 모델 저장 (교차 검증 완료 후)
    if cv_results['best_model'] and config['training']['save_best_model']:
        save_best_model(cv_results['best_model'], cv_results, config)
    
    # 집계 메트릭 계산
    if all_fold_metrics:
        aggregate_metrics = aggregate_cv_metrics(all_fold_metrics)
        cv_results['aggregate_metrics'] = aggregate_metrics
        
        # MLflow에 집계 메트릭 로깅
        for metric_name, value in aggregate_metrics.items():
            mlflow.log_metric(f"cv_{metric_name}", value)
        
        logger.info(f"교차 검증 완료: {fold_count}개 폴드")
        logger.info(f"집계 성능: {aggregate_metrics}")
        logger.info(f"최고 성능 폴드: {cv_results['best_fold']} (점수: {cv_results['best_score']:.4f})")
    
    return cv_results


def save_best_model(model: Any, cv_results: Dict[str, Any], config: Dict[str, Any]):
    """
    최고 성능 모델을 저장합니다.
    
    Args:
        model: 최고 성능 모델
        cv_results: 교차 검증 결과
        config: 설정 딕셔너리
    """
    model_save_path = config['training']['model_save_path']
    import os
    os.makedirs(model_save_path, exist_ok=True)
    
    best_fold = cv_results['best_fold']
    model_file = f"{model_save_path}/best_model_fold_{best_fold}.joblib"
    model.save_model(model_file)
    
    # MLflow에 모델 아티팩트로 저장
    mlflow.log_artifact(model_file, f"models/best_fold_{best_fold}")
    
    logger.info(f"최고 성능 모델 저장 완료: {model_file} (폴드 {best_fold})")


def aggregate_cv_metrics(all_metrics: list) -> Dict[str, float]:
    """
    교차 검증 메트릭을 집계합니다.
    
    Args:
        all_metrics: 모든 폴드의 메트릭 리스트
        
    Returns:
        집계된 메트릭
    """
    if not all_metrics:
        return {}
    
    # 모든 메트릭 이름 수집
    all_metric_names = set()
    for metrics in all_metrics:
        all_metric_names.update(metrics.keys())
    
    # 각 메트릭별로 평균과 표준편차 계산
    aggregate_metrics = {}
    
    for metric_name in all_metric_names:
        values = [metrics.get(metric_name, np.nan) for metrics in all_metrics]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            aggregate_metrics[f"{metric_name}_mean"] = np.mean(values)
            aggregate_metrics[f"{metric_name}_std"] = np.std(values)
    
    return aggregate_metrics


def main():
    """
    테스트용 메인 함수
    """
    import yaml
    
    # 설정 로드
    with open("configs/default_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 테스트 데이터 로드
    data_path = "data/processed/processed_data_with_features.csv"
    df = pd.read_csv(data_path, nrows=1000)
    
    logger.info(f"테스트 데이터 로드: {len(df):,} 행")
    
    # 교차 검증 실행
    cv_results = run_cross_validation(df, config)
    
    logger.info("학습 테스트 완료!")


if __name__ == "__main__":
    main() 