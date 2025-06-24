"""
모델 평가 모듈

이 모듈은 다양한 성능 지표를 계산하고 평가 결과를 시각화하는 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    balanced_accuracy_score, roc_curve
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
        
        # 기본 지표 계산 - 예외 발생 시 0.0 반환
        try:
            mae = mean_absolute_error(y_true, y_pred)
        except Exception as e:
            logger.warning(f"MAE 계산 중 예외 발생: {e}, 0.0 반환")
            mae = 0.0
            
        try:
            mse = mean_squared_error(y_true, y_pred)
        except Exception as e:
            logger.warning(f"MSE 계산 중 예외 발생: {e}, 0.0 반환")
            mse = 0.0
            
        try:
            rmse = np.sqrt(mse)
        except Exception as e:
            logger.warning(f"RMSE 계산 중 예외 발생: {e}, 0.0 반환")
            rmse = 0.0
            
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"R2 계산 중 예외 발생: {e}, 0.0 반환")
            r2 = 0.0
        
        # 추가 지표 - 예외 발생 시 0.0 반환
        try:
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # MAPE (Mean Absolute Percentage Error)
        except Exception as e:
            logger.warning(f"MAPE 계산 중 예외 발생: {e}, 0.0 반환")
            mape = 0.0
        
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
        
        # === 타입 변환 추가 ===
        y_true = pd.Series(y_true).astype(int)
        y_pred = pd.Series(y_pred).astype(int)
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba.astype(float)
            else:
                y_pred_proba = y_pred_proba.astype(float)
        # =====================
        
        # 클래스 분포 확인
        class_counts = y_true.value_counts()
        total_samples = len(y_true)
        positive_samples = class_counts.get(1, 0)
        negative_samples = class_counts.get(0, 0)
        positive_ratio = positive_samples / total_samples if total_samples > 0 else 0
        
        # 예측 분포 확인
        pred_counts = y_pred.value_counts()
        pred_positive = pred_counts.get(1, 0)
        
        logger.info(f"클래스 분포 - 실제: 0={negative_samples}, 1={positive_samples} (비율: {positive_ratio:.6f})")
        logger.info(f"예측 분포 - 예측: 0={pred_counts.get(0, 0)}, 1={pred_positive}")
        
        # 극도로 불균형한 데이터 처리
        if positive_samples == 0:
            logger.warning("양성 클래스가 없습니다. 모든 메트릭을 0.0으로 설정합니다.")
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 1.0,
                'balanced_accuracy': 0.5, 'roc_auc': 0.0, 'pr_auc': 0.0,
                'positive_samples': 0, 'negative_samples': total_samples,
                'total_samples': total_samples, 'positive_ratio': 0.0
            }
        
        # 모든 예측이 0인 경우 처리
        if pred_positive == 0:
            logger.warning("모든 예측이 0입니다. 극도로 불균형한 데이터에서 일반적인 현상입니다.")
            # 기본 지표 계산 - 예외 발생 시 0.0 반환
            try:
                precision = 0.0  # 모든 예측이 0이므로 precision은 0
                recall = 0.0     # 모든 예측이 0이므로 recall은 0
                f1 = 0.0         # precision과 recall이 모두 0이므로 f1도 0
            except Exception as e:
                logger.warning(f"기본 지표 계산 중 예외 발생: {e}, 0.0 반환")
                precision = recall = f1 = 0.0
            
            # 정확도는 높을 수 있음 (대부분이 0이므로)
            try:
                accuracy = (y_true == y_pred).mean()
            except Exception as e:
                logger.warning(f"Accuracy 계산 중 예외 발생: {e}, 0.0 반환")
                accuracy = 0.0
                
            try:
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Balanced accuracy 계산 중 예외 발생: {e}, 0.0 반환")
                balanced_acc = 0.5  # 균형 정확도는 0.5 (랜덤 예측과 동일)
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'positive_samples': positive_samples,
                'negative_samples': negative_samples,
                'total_samples': total_samples,
                'positive_ratio': positive_ratio,
                'roc_auc': 0.0,  # 모든 예측이 0이므로 ROC-AUC는 의미없음
                'pr_auc': 0.0    # 모든 예측이 0이므로 PR-AUC는 의미없음
            }
            
            return metrics
        
        # 일반적인 경우의 계산
        # 기본 지표 계산 - 예외 발생 시 0.0 반환
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            logger.warning(f"Precision 계산 중 예외 발생: {e}, 0.0 반환")
            precision = 0.0
            
        try:
            recall = recall_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            logger.warning(f"Recall 계산 중 예외 발생: {e}, 0.0 반환")
            recall = 0.0
            
        try:
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            logger.warning(f"F1-score 계산 중 예외 발생: {e}, 0.0 반환")
            f1 = 0.0
        
        # ROC-AUC 계산 (확률이 제공된 경우)
        roc_auc = None
        if y_pred_proba is not None and len(y_pred_proba.shape) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception as e:
                logger.warning(f"ROC-AUC 계산 중 예외 발생: {e}, 0.0 반환")
                roc_auc = 0.0
        
        # 추가 지표 - 예외 발생 시 0.0 반환
        try:
            accuracy = (y_true == y_pred).mean()
        except Exception as e:
            logger.warning(f"Accuracy 계산 중 예외 발생: {e}, 0.0 반환")
            accuracy = 0.0
            
        try:
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Balanced accuracy 계산 중 예외 발생: {e}, 0.0 반환")
            balanced_acc = 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'total_samples': total_samples,
            'positive_ratio': positive_ratio
        }
        
        if roc_auc is not None:
            metrics['roc_auc'] = roc_auc
        
        # PR-AUC 계산 (확률이 제공된 경우)
        if y_pred_proba is not None:
            try:
                # 예측 확률의 형태 확인 및 조정
                if len(y_pred_proba.shape) > 1:
                    # 2D 배열인 경우 양성 클래스 확률 사용
                    if y_pred_proba.shape[1] == 2:
                        y_pred_proba_positive = y_pred_proba[:, 1]
                    else:
                        y_pred_proba_positive = y_pred_proba[:, 0]
                else:
                    # 1D 배열인 경우 그대로 사용
                    y_pred_proba_positive = y_pred_proba
                
                pr_auc = average_precision_score(y_true, y_pred_proba_positive)
                metrics['pr_auc'] = pr_auc
                logger.info(f"PR-AUC 계산 완료: {pr_auc:.4f}")
            except Exception as e:
                logger.warning(f"PR-AUC 계산 중 예외 발생: {e}, 0.0 반환")
                metrics['pr_auc'] = 0.0
        else:
            metrics['pr_auc'] = 0.0
        
        return metrics


def calculate_precision_recall_curve(y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Precision-Recall Curve 계산
    
    Args:
        y_true: 실제 값
        y_pred_proba: 예측 확률 (양성 클래스)
        
    Returns:
        PR curve 데이터와 AUC
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        return {
            'precision': precision,
            'recall': recall, 
            'thresholds': thresholds,
            'pr_auc': pr_auc
        }
    except Exception as e:
        logger.warning(f"Precision-Recall Curve 계산 중 예외 발생: {e}, 기본값 반환")
        return {
            'precision': np.array([0.0]),
            'recall': np.array([0.0]), 
            'thresholds': np.array([0.5]),
            'pr_auc': 0.0
        }


def calculate_balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Balanced Accuracy 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        
    Returns:
        Balanced Accuracy 값
    """
    try:
        return balanced_accuracy_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Balanced Accuracy 계산 중 예외 발생: {e}, 0.0 반환")
        return 0.0


def compare_roc_vs_pr_auc(y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    ROC-AUC vs PR-AUC 비교
    
    Args:
        y_true: 실제 값
        y_pred_proba: 예측 확률
        
    Returns:
        ROC-AUC와 PR-AUC 값
    """
    try:
        # ROC-AUC 계산
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = np.trapz(tpr, fpr)
        
        # PR-AUC 계산
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    except Exception as e:
        logger.warning(f"ROC vs PR-AUC 비교 계산 중 예외 발생: {e}, 0.0 반환")
        return {
            'roc_auc': 0.0,
            'pr_auc': 0.0
        }


def optimize_f1_threshold(y_true: pd.Series, y_pred_proba: np.ndarray, 
                         thresholds: np.ndarray = None) -> Dict[str, Any]:
    """
    F1-Score 최적화를 위한 임계값 찾기
    
    Args:
        y_true: 실제 값
        y_pred_proba: 예측 확률
        thresholds: 테스트할 임계값들 (None이면 자동 생성)
        
    Returns:
        최적 임계값과 해당 F1-score
    """
    try:
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)
        
        best_f1 = 0
        best_threshold = 0.5
        threshold_results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            threshold_results.append({
                'threshold': threshold,
                'f1_score': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return {
            'best_threshold': best_threshold,
            'best_f1_score': best_f1,
            'all_results': threshold_results
        }
    except Exception as e:
        logger.warning(f"F1 threshold 최적화 중 예외 발생: {e}, 기본값 반환")
        return {
            'best_threshold': 0.5,
            'best_f1_score': 0.0,
            'all_results': []
        }


def create_precision_recall_plot(y_true: pd.Series, y_pred_proba: np.ndarray, 
                                target_name: str, save_path: str = None):
    """
    Precision-Recall Curve 시각화
    
    Args:
        y_true: 실제 값
        y_pred_proba: 예측 확률
        target_name: 타겟 이름
        save_path: 저장 경로
    """
    try:
        import matplotlib.pyplot as plt
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {target_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR Curve 저장 완료: {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib이 설치되지 않아 PR curve를 시각화할 수 없습니다.")


def create_roc_curve_plot(y_true: pd.Series, y_pred_proba: np.ndarray, 
                         target_name: str, save_path: str = None):
    """
    ROC Curve 시각화
    
    Args:
        y_true: 실제 값
        y_pred_proba: 예측 확률
        target_name: 타겟 이름
        save_path: 저장 경로
    """
    try:
        import matplotlib.pyplot as plt
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {target_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC Curve 저장 완료: {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib이 설치되지 않아 ROC curve를 시각화할 수 없습니다.")


def analyze_fold_performance_distribution(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    폴드별 성능 지표 분포 분석
    
    Args:
        fold_results: 폴드별 결과 리스트
        
    Returns:
        성능 지표 통계 정보
    """
    analysis = {}
    
    for target in fold_results[0].get('metrics', {}).keys():
        target_analysis = {}
        
        for metric in ['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy', 'roc_auc', 'pr_auc']:
            if metric in fold_results[0]['metrics'][target]:
                values = [fold['metrics'][target].get(metric, 0) for fold in fold_results]
                values = [v for v in values if v is not None]
                
                if values:
                    target_analysis[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
        
        analysis[target] = target_analysis
    
    return analysis


def calculate_confidence_intervals(performance_metrics: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    신뢰구간 계산
    
    Args:
        performance_metrics: 성능 지표 리스트
        confidence: 신뢰수준 (0.95 = 95%)
        
    Returns:
        신뢰구간 정보
    """
    if len(performance_metrics) < 2:
        return {'lower': performance_metrics[0], 'upper': performance_metrics[0]}
    
    mean_val = np.mean(performance_metrics)
    std_val = np.std(performance_metrics, ddof=1)
    n = len(performance_metrics)
    
    # t-분포 사용 (소표본)
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin_of_error = t_value * (std_val / np.sqrt(n))
    
    return {
        'mean': mean_val,
        'lower': mean_val - margin_of_error,
        'upper': mean_val + margin_of_error,
        'confidence': confidence
    }


def analyze_fold_variability(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    폴드 간 성능 변동성 분석
    
    Args:
        fold_results: 폴드별 결과 리스트
        
    Returns:
        변동성 분석 결과
    """
    variability = {}
    
    for target in fold_results[0].get('metrics', {}).keys():
        target_variability = {}
        
        for metric in ['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy']:
            if metric in fold_results[0]['metrics'][target]:
                values = [fold['metrics'][target].get(metric, 0) for fold in fold_results]
                values = [v for v in values if v is not None]
                
                if len(values) > 1:
                    cv = np.std(values) / np.mean(values)  # 변동계수
                    target_variability[metric] = {
                        'coefficient_of_variation': cv,
                        'stability': 'high' if cv < 0.1 else 'medium' if cv < 0.2 else 'low'
                    }
        
        variability[target] = target_variability
    
    return variability


def calculate_all_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, 
                         y_pred_proba: Dict[str, np.ndarray] = None,
                         config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    모든 타겟에 대한 종합적인 평가 지표를 계산합니다.
    
    Args:
        y_true: 실제 값 DataFrame
        y_pred: 예측 값 DataFrame
        y_pred_proba: 예측 확률 Dictionary (선택사항)
        config: 설정 딕셔너리 (선택사항)
        
    Returns:
        모든 평가 지표를 포함한 딕셔너리
    """
    logger.info("=== 종합 평가 지표 계산 시작 ===")
    logger.info(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    logger.info(f"y_true columns: {list(y_true.columns)}")
    logger.info(f"y_pred columns: {list(y_pred.columns)}")
    
    all_metrics = {}
    
    # 각 타겟별로 평가
    for target in y_true.columns:
        logger.info(f"타겟 '{target}' 평가 중...")
        
        # 예측 컬럼 찾기
        pred_col = None
        for col in y_pred.columns:
            if target in col or col.replace('remainder__', '') == target.replace('remainder__', ''):
                pred_col = col
                break
        
        if pred_col is None:
            logger.warning(f"타겟 '{target}'에 대한 예측 컬럼을 찾을 수 없습니다.")
            continue
        
        logger.info(f"타겟 '{target}' -> 예측 컬럼 '{pred_col}'")
        
        true_vals = y_true[target]
        pred_vals = y_pred[pred_col]
        
        # 결측치 제거
        valid_mask = true_vals.notna() & pred_vals.notna()
        true_vals_clean = true_vals[valid_mask]
        pred_vals_clean = pred_vals[valid_mask]
        
        logger.info(f"타겟 '{target}' - 유효한 샘플 수: {len(true_vals_clean)}")
        logger.info(f"타겟 '{target}' - true_vals unique: {true_vals_clean.unique()}")
        logger.info(f"타겟 '{target}' - pred_vals unique: {pred_vals_clean.unique()}")
        
        if len(true_vals_clean) == 0:
            logger.warning(f"타겟 '{target}'에 대한 유효한 샘플이 없습니다.")
            continue
        
        # 타겟 타입 판단
        target_original = target.replace('remainder__', '')
        if target_original.endswith('_next_year'):
            base_target = target_original.replace('_next_year', '')
            
            if base_target in ['anxiety_score', 'depress_score', 'sleep_score']:
                # 회귀 문제
                logger.info(f"타겟 '{target}' - 회귀 문제로 처리")
                target_metrics = calculate_regression_metrics(true_vals_clean, pred_vals_clean)
                all_metrics[target] = target_metrics
                for metric_name, value in target_metrics.items():
                    logger.info(f"  {metric_name}: {value}")
            elif base_target in ['suicide_t', 'suicide_a']:
                # 분류 문제
                logger.info(f"타겟 '{target}' - 분류 문제로 처리")
                pred_proba = None
                if y_pred_proba and target in y_pred_proba:
                    pred_proba = y_pred_proba[target]
                    logger.info(f"타겟 '{target}' - 예측 확률 사용 가능")
                target_metrics = calculate_classification_metrics(true_vals_clean, pred_vals_clean, pred_proba)
                # 빈 메트릭도 0.0으로 채움
                for m in ['precision','recall','f1','accuracy','balanced_accuracy','roc_auc','pr_auc']:
                    if m not in target_metrics:
                        target_metrics[m] = 0.0
                all_metrics[target] = target_metrics
                for metric_name, value in target_metrics.items():
                    logger.info(f"  {metric_name}: {value}")
        else:
            # 기본적으로 분류로 처리
            logger.info(f"타겟 '{target}' - 기본 분류 문제로 처리")
            target_metrics = calculate_classification_metrics(true_vals_clean, pred_vals_clean)
            for m in ['precision','recall','f1','accuracy','balanced_accuracy','roc_auc','pr_auc']:
                if m not in target_metrics:
                    target_metrics[m] = 0.0
            all_metrics[target] = target_metrics
            for metric_name, value in target_metrics.items():
                logger.info(f"  {metric_name}: {value}")
    
    logger.info(f"=== 종합 평가 지표 계산 완료 - {len(all_metrics)}개 타겟 ===")
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
            logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            if 'pr_auc' in metrics and metrics['pr_auc'] is not None:
                logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
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


def evaluate_with_advanced_metrics(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: np.ndarray,
                                 target_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    고급 평가 지표들을 포함한 종합 평가 수행
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        y_pred_proba: 예측 확률
        target_name: 타겟 이름
        config: 설정 딕셔너리
        
    Returns:
        모든 평가 결과
    """
    if config is None:
        config = {}
    
    # 기본 분류 지표 계산
    basic_metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    
    results = {
        'target_name': target_name,
        'basic_metrics': basic_metrics,
        'advanced_metrics': {}
    }
    
    # 고급 평가 지표들 계산
    if config.get('advanced_metrics', {}).get('enable_precision_recall_curve', True):
        try:
            pr_curve = calculate_precision_recall_curve(y_true, y_pred_proba)
            results['advanced_metrics']['precision_recall_curve'] = pr_curve
        except Exception as e:
            logger.warning(f"PR Curve 계산 실패: {e}")
    
    if config.get('advanced_metrics', {}).get('enable_roc_curve', True):
        try:
            roc_pr_comparison = compare_roc_vs_pr_auc(y_true, y_pred_proba)
            results['advanced_metrics']['roc_pr_comparison'] = roc_pr_comparison
        except Exception as e:
            logger.warning(f"ROC vs PR-AUC 비교 실패: {e}")
    
    if config.get('advanced_metrics', {}).get('enable_f1_threshold_optimization', True):
        try:
            f1_config = config.get('advanced_metrics', {}).get('f1_threshold_optimization', {})
            threshold_range = f1_config.get('threshold_range', [0.1, 0.9])
            step_size = f1_config.get('step_size', 0.05)
            
            thresholds = np.arange(threshold_range[0], threshold_range[1] + step_size, step_size)
            f1_optimization = optimize_f1_threshold(y_true, y_pred_proba, thresholds)
            results['advanced_metrics']['f1_threshold_optimization'] = f1_optimization
        except Exception as e:
            logger.warning(f"F1 임계값 최적화 실패: {e}")
    
    return results


def create_comprehensive_evaluation_report(all_metrics: Dict[str, Any], 
                                         fold_results: List[Dict[str, Any]] = None,
                                         config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    종합적인 평가 리포트 생성
    
    Args:
        all_metrics: 모든 성능 지표
        fold_results: 폴드별 결과 (선택사항)
        config: 설정 딕셔너리
        
    Returns:
        종합 평가 리포트
    """
    if config is None:
        config = {}
    
    report = {
        'summary': all_metrics,
        'fold_analysis': {},
        'recommendations': []
    }
    
    # 폴드 분석 수행
    if fold_results and config.get('advanced_metrics', {}).get('enable_fold_analysis', True):
        try:
            # 폴드별 성능 분포 분석
            fold_analysis = analyze_fold_performance_distribution(fold_results)
            report['fold_analysis']['performance_distribution'] = fold_analysis
            
            # 폴드 간 변동성 분석
            fold_variability = analyze_fold_variability(fold_results)
            report['fold_analysis']['variability'] = fold_variability
            
            # 신뢰구간 계산
            if config.get('advanced_metrics', {}).get('fold_analysis', {}).get('calculate_confidence_intervals', True):
                confidence_level = config.get('advanced_metrics', {}).get('fold_analysis', {}).get('confidence_level', 0.95)
                
                for target in fold_results[0].get('metrics', {}).keys():
                    for metric in ['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy']:
                        if metric in fold_results[0]['metrics'][target]:
                            values = [fold['metrics'][target].get(metric, 0) for fold in fold_results]
                            values = [v for v in values if v is not None]
                            
                            if len(values) > 1:
                                ci = calculate_confidence_intervals(values, confidence_level)
                                if 'confidence_intervals' not in report['fold_analysis']:
                                    report['fold_analysis']['confidence_intervals'] = {}
                                if target not in report['fold_analysis']['confidence_intervals']:
                                    report['fold_analysis']['confidence_intervals'][target] = {}
                                report['fold_analysis']['confidence_intervals'][target][metric] = ci
                                
        except Exception as e:
            logger.warning(f"폴드 분석 실패: {e}")
    
    # 권장사항 생성
    recommendations = []
    
    for target, target_info in all_metrics.items():
        if target_info['type'] == 'classification':
            metrics = target_info['metrics']
            
            # 불균형 데이터 관련 권장사항
            if metrics.get('positive_ratio', 0) < 0.1:
                recommendations.append(f"{target}: 극도 불균형 데이터 (양성 비율: {metrics['positive_ratio']:.3f}). "
                                    "Focal Loss나 SMOTE 사용을 고려하세요.")
            
            # 성능 관련 권장사항
            if metrics.get('f1', 0) < 0.3:
                recommendations.append(f"{target}: 낮은 F1-score ({metrics['f1']:.3f}). "
                                    "모델 튜닝이나 피처 엔지니어링이 필요합니다.")
            
            if metrics.get('balanced_accuracy', 0) < 0.6:
                recommendations.append(f"{target}: 낮은 Balanced Accuracy ({metrics['balanced_accuracy']:.3f}). "
                                    "클래스 불균형 처리가 필요합니다.")
    
    report['recommendations'] = recommendations
    
    return report


def save_advanced_evaluation_plots(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: np.ndarray,
                                 target_name: str, config: Dict[str, Any] = None):
    """
    고급 평가 플롯들을 생성하고 저장
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        y_pred_proba: 예측 확률
        target_name: 타겟 이름
        config: 설정 딕셔너리
    """
    if config is None:
        config = {}
    
    plots_config = config.get('evaluation', {})
    save_plots = plots_config.get('save_plots', True)
    plots_path = plots_config.get('plots_save_path', 'evaluation_plots/')
    
    if not save_plots:
        return
    
    # 디렉토리 생성
    plots_path = Path(plots_path)
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # 혼동 행렬 저장
    try:
        cm_path = plots_path / f"{target_name}_confusion_matrix.png"
        create_confusion_matrix_plot(y_true, y_pred, target_name, str(cm_path))
    except Exception as e:
        logger.warning(f"혼동 행렬 저장 실패: {e}")
    
    # PR Curve 저장
    if plots_config.get('advanced_metrics', {}).get('enable_precision_recall_curve', True):
        try:
            pr_path = plots_path / f"{target_name}_precision_recall_curve.png"
            create_precision_recall_plot(y_true, y_pred_proba, target_name, str(pr_path))
        except Exception as e:
            logger.warning(f"PR Curve 저장 실패: {e}")
    
    # ROC Curve 저장
    if plots_config.get('advanced_metrics', {}).get('enable_roc_curve', True):
        try:
            roc_path = plots_path / f"{target_name}_roc_curve.png"
            create_roc_curve_plot(y_true, y_pred_proba, target_name, str(roc_path))
        except Exception as e:
            logger.warning(f"ROC Curve 저장 실패: {e}")


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
    
    # 분류 테스트 (불균형 데이터 시뮬레이션)
    y_true_clf = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% 양성 클래스
    y_pred_proba = np.random.beta(2, 8, n_samples)  # 불균형 예측 확률
    y_pred_clf = (y_pred_proba > 0.5).astype(int)
    
    clf_metrics = calculate_classification_metrics(pd.Series(y_true_clf), pd.Series(y_pred_clf), y_pred_proba)
    logger.info("분류 메트릭 테스트:")
    for metric, value in clf_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # 고급 평가 지표 테스트
    logger.info("\n=== 고급 평가 지표 테스트 ===")
    
    # PR Curve 테스트
    pr_curve = calculate_precision_recall_curve(pd.Series(y_true_clf), y_pred_proba)
    logger.info(f"PR-AUC: {pr_curve['pr_auc']:.4f}")
    
    # ROC vs PR-AUC 비교
    roc_pr_comparison = compare_roc_vs_pr_auc(pd.Series(y_true_clf), y_pred_proba)
    logger.info(f"ROC-AUC: {roc_pr_comparison['roc_auc']:.4f}")
    logger.info(f"PR-AUC: {roc_pr_comparison['pr_auc']:.4f}")
    
    # F1 임계값 최적화
    f1_optimization = optimize_f1_threshold(pd.Series(y_true_clf), y_pred_proba)
    logger.info(f"최적 F1 임계값: {f1_optimization['best_threshold']:.3f}")
    logger.info(f"최적 F1-Score: {f1_optimization['best_f1_score']:.4f}")
    
    # 신뢰구간 테스트
    test_metrics = [0.75, 0.78, 0.72, 0.80, 0.76]
    ci = calculate_confidence_intervals(test_metrics, 0.95)
    logger.info(f"신뢰구간 (95%): {ci['lower']:.4f} - {ci['upper']:.4f}")
    
    # 종합 평가 테스트
    config = {
        'advanced_metrics': {
            'enable_precision_recall_curve': True,
            'enable_roc_curve': True,
            'enable_f1_threshold_optimization': True,
            'f1_threshold_optimization': {
                'threshold_range': [0.1, 0.9],
                'step_size': 0.05
            }
        }
    }
    
    advanced_results = evaluate_with_advanced_metrics(
        pd.Series(y_true_clf), 
        pd.Series(y_pred_clf), 
        y_pred_proba, 
        'test_target', 
        config
    )
    
    logger.info("고급 평가 결과:")
    logger.info(f"  기본 F1-Score: {advanced_results['basic_metrics']['f1']:.4f}")
    logger.info(f"  최적 F1-Score: {advanced_results['advanced_metrics']['f1_threshold_optimization']['best_f1_score']:.4f}")
    
    logger.info("평가 모듈 테스트 완료!")


if __name__ == "__main__":
    main() 