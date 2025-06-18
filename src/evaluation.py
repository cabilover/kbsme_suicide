"""
모델 평가 모듈

이 모듈은 회귀 및 분류 모델의 성능을 평가하는 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import warnings
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    회귀 모델의 성능 지표를 계산합니다.
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        
    Returns:
        성능 지표 딕셔너리
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 기본 지표 계산
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # 추가 지표
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # MAPE (Mean Absolute Percentage Error)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    return metrics


def calculate_classification_metrics(y_true: pd.Series, y_pred: pd.Series, 
                                   y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    분류 모델의 성능 지표를 계산합니다.
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        y_pred_proba: 예측 확률 (ROC-AUC 계산용)
        
    Returns:
        성능 지표 딕셔너리
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 기본 지표 계산
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC 계산 (확률이 제공된 경우)
        roc_auc = None
        if y_pred_proba is not None and len(y_pred_proba.shape) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                roc_auc = None
        
        # 추가 지표
        accuracy = (y_true == y_pred).mean()
        
        # 클래스별 샘플 수
        class_counts = y_true.value_counts()
        total_samples = len(y_true)
        positive_samples = class_counts.get(1, 0)
        negative_samples = class_counts.get(0, 0)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'total_samples': total_samples,
            'positive_ratio': positive_samples / total_samples if total_samples > 0 else 0
        }
        
        if roc_auc is not None:
            metrics['roc_auc'] = roc_auc
    
    return metrics


def calculate_all_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, 
                         y_pred_proba: Dict[str, np.ndarray] = None,
                         config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    모든 타겟에 대한 성능 지표를 계산합니다.
    
    Args:
        y_true: 실제 값 데이터프레임
        y_pred: 예측 값 데이터프레임
        y_pred_proba: 예측 확률 딕셔너리 (분류 문제용)
        config: 설정 딕셔너리
        
    Returns:
        모든 성능 지표
    """
    all_metrics = {}
    
    for target in y_true.columns:
        if target not in y_pred.columns:
            logger.warning(f"타겟 {target}에 대한 예측값이 없습니다.")
            continue
        
        true_vals = y_true[target]
        pred_vals = y_pred[target]
        
        # 회귀 문제인지 분류 문제인지 판단
        if target.endswith('_next_year'):
            base_target = target.replace('_next_year', '')
            if base_target in ['anxiety_score', 'depress_score', 'sleep_score']:
                # 회귀 문제
                target_metrics = calculate_regression_metrics(true_vals, pred_vals)
                all_metrics[target] = {
                    'type': 'regression',
                    'metrics': target_metrics
                }
            elif base_target in ['suicide_t', 'suicide_a']:
                # 분류 문제
                pred_proba = None
                if y_pred_proba and target in y_pred_proba:
                    pred_proba = y_pred_proba[target]
                
                target_metrics = calculate_classification_metrics(true_vals, pred_vals, pred_proba)
                all_metrics[target] = {
                    'type': 'classification',
                    'metrics': target_metrics
                }
        else:
            # 타겟 이름이 _next_year로 끝나지 않는 경우 기본적으로 회귀로 처리
            target_metrics = calculate_regression_metrics(true_vals, pred_vals)
            all_metrics[target] = {
                'type': 'regression',
                'metrics': target_metrics
            }
    
    return all_metrics


def print_evaluation_summary(all_metrics: Dict[str, Any], feature_validation_results: Dict[str, bool] = None, 
                           early_stopping_used: bool = False, cv_results: Dict[str, Any] = None):
    """
    평가 결과를 요약하여 출력합니다.
    
    Args:
        all_metrics: 모든 성능 지표
        feature_validation_results: 피처 검증 결과 (선택사항)
        early_stopping_used: Early Stopping 사용 여부 (선택사항)
        cv_results: 교차 검증 결과 (선택사항)
    """
    logger.info("=== 모델 성능 평가 결과 ===")
    
    # Early Stopping 정보 출력
    if early_stopping_used:
        logger.info("Early Stopping: 활성화 ✓")
    else:
        logger.info("Early Stopping: 비활성화")
    
    # 교차 검증 정보 출력
    if cv_results:
        logger.info(f"교차 검증 폴드 수: {len(cv_results.get('fold_results', []))}")
        if cv_results.get('best_fold'):
            logger.info(f"최고 성능 폴드: {cv_results['best_fold']} (점수: {cv_results.get('best_score', 0):.4f})")
    
    for target, target_info in all_metrics.items():
        target_type = target_info['type']
        metrics = target_info['metrics']
        
        logger.info(f"\n{target} ({target_type}):")
        
        if target_type == 'regression':
            logger.info(f"  MAE: {metrics['mae']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  R²: {metrics['r2']:.4f}")
            logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        else:
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1']:.4f}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            if 'roc_auc' in metrics:
                logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  Positive Ratio: {metrics['positive_ratio']:.4f}")
    
    # 피처 검증 결과 출력
    if feature_validation_results:
        logger.info("\n=== 피처 검증 결과 ===")
        valid_count = sum(feature_validation_results.values())
        total_count = len(feature_validation_results)
        logger.info(f"검증된 피처: {valid_count}/{total_count}개")
        
        for feature, is_valid in feature_validation_results.items():
            status = "✓" if is_valid else "✗"
            logger.info(f"  {feature}: {status}")
        
        if valid_count < total_count:
            logger.warning("일부 피처에서 데이터 유출이 발견되었습니다!")


def save_evaluation_results(all_metrics: Dict[str, Any], save_path: str = None):
    """
    평가 결과를 파일로 저장합니다.
    
    Args:
        all_metrics: 모든 성능 지표
        save_path: 저장할 파일 경로
    """
    if save_path is None:
        save_path = "evaluation_results.json"
    
    # Path 객체로 변환
    save_path = Path(save_path)
    
    # 디렉토리 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    
    # JSON 직렬화를 위해 numpy 타입을 Python 타입으로 변환
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # 딕셔너리를 재귀적으로 변환
    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        else:
            return convert_numpy_types(d)
    
    converted_metrics = convert_dict(all_metrics)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(converted_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"평가 결과 저장 완료: {save_path}")


def create_confusion_matrix_plot(y_true: pd.Series, y_pred: pd.Series, 
                                target_name: str, save_path: str = None):
    """
    혼동 행렬을 시각화합니다.
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        target_name: 타겟 이름
        save_path: 저장할 파일 경로
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 혼동 행렬 계산
        cm = confusion_matrix(y_true, y_pred)
        
        # 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {target_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"혼동 행렬 저장 완료: {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib 또는 seaborn이 설치되지 않아 혼동 행렬을 시각화할 수 없습니다.")


def create_regression_plot(y_true: pd.Series, y_pred: pd.Series, 
                          target_name: str, save_path: str = None):
    """
    회귀 결과를 시각화합니다.
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        target_name: 타겟 이름
        save_path: 저장할 파일 경로
    """
    try:
        import matplotlib.pyplot as plt
        
        # 산점도 생성
        plt.figure(figsize=(10, 8))
        
        # 산점도
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'Prediction vs True - {target_name}')
        
        # 잔차 플롯
        plt.subplot(2, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # 잔차 히스토그램
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        
        # Q-Q 플롯
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"회귀 플롯 저장 완료: {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib이 설치되지 않아 회귀 플롯을 생성할 수 없습니다.")


def main():
    """
    테스트용 메인 함수
    """
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    # 회귀 테스트
    y_true_reg = np.random.randn(n_samples)
    y_pred_reg = y_true_reg + np.random.normal(0, 0.5, n_samples)
    
    reg_metrics = calculate_regression_metrics(pd.Series(y_true_reg), pd.Series(y_pred_reg))
    logger.info("회귀 메트릭 테스트:")
    for metric, value in reg_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # 분류 테스트
    y_true_clf = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    y_pred_clf = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    clf_metrics = calculate_classification_metrics(pd.Series(y_true_clf), pd.Series(y_pred_clf))
    logger.info("분류 메트릭 테스트:")
    for metric, value in clf_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("평가 모듈 테스트 완료!")


if __name__ == "__main__":
    main() 