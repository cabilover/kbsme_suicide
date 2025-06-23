"""
손실 함수 모듈

이 모듈은 불균형 데이터 처리를 위한 다양한 손실 함수를 구현합니다.
주로 Focal Loss와 관련된 함수들을 포함합니다.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging
import warnings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss:
    """
    Focal Loss 구현
    
    불균형 데이터에서 소수 클래스 예측 성능을 향상시키기 위한 손실 함수입니다.
    쉬운 샘플의 기여도를 낮추고 어려운 샘플(오분류되기 쉬운 소수 클래스)에 집중합니다.
    
    참고: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). 
    Focal loss for dense object detection. In Proceedings of the IEEE international 
    conference on computer vision (pp. 2980-2988).
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 reduction: str = 'mean', eps: float = 1e-7):
        """
        Focal Loss 초기화
        
        Args:
            alpha: 클래스 가중치 (0 < alpha < 1). 소수 클래스에 더 큰 가중치를 부여
            gamma: 포커싱 파라미터 (gamma >= 0). 쉬운 샘플의 기여도를 낮춤
            reduction: 손실 계산 방식 ('mean', 'sum', 'none')
            eps: 수치 안정성을 위한 작은 값
        """
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        
        logger.info(f"Focal Loss 초기화: alpha={alpha}, gamma={gamma}, reduction={reduction}")
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Focal Loss 계산
        
        Args:
            y_true: 실제 값 (0 또는 1)
            y_pred: 예측 확률 (0~1)
            
        Returns:
            Focal Loss 값
        """
        return self.forward(y_true, y_pred)
    
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Focal Loss 순전파 계산
        
        Args:
            y_true: 실제 값 (0 또는 1)
            y_pred: 예측 확률 (0~1)
            
        Returns:
            Focal Loss 값
        """
        # 수치 안정성을 위한 클리핑
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        
        # Cross-entropy 계산
        ce_loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        
        # Focal Loss 계산
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # 예측 확률
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha 가중치 적용
        alpha_weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        
        # 최종 Focal Loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Reduction 적용
        if self.reduction == 'mean':
            return np.mean(focal_loss)
        elif self.reduction == 'sum':
            return np.sum(focal_loss)
        else:  # 'none'
            return focal_loss
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Focal Loss 역전파 계산 (그래디언트)
        
        Args:
            y_true: 실제 값 (0 또는 1)
            y_pred: 예측 확률 (0~1)
            
        Returns:
            그래디언트
        """
        # 수치 안정성을 위한 클리핑
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        
        # 예측 확률
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Focal Loss 그래디언트 계산
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        
        # 그래디언트 계산
        grad = alpha_weight * focal_weight * (y_pred - y_true)
        
        return grad


def calculate_focal_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                        alpha: float = 1.0, gamma: float = 2.0) -> float:
    """
    Focal Loss를 계산하는 편의 함수
    
    Args:
        y_true: 실제 값 (0 또는 1)
        y_pred: 예측 확률 (0~1)
        alpha: 클래스 가중치
        gamma: 포커싱 파라미터
        
    Returns:
        Focal Loss 값
    """
    focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
    return focal_loss(y_true, y_pred)


def calculate_class_weights(y: pd.Series, method: str = 'balanced') -> Dict[int, float]:
    """
    클래스 가중치를 계산합니다.
    
    Args:
        y: 타겟 시리즈
        method: 가중치 계산 방법 ('balanced', 'inverse', 'sqrt_inverse')
        
    Returns:
        클래스별 가중치 딕셔너리
    """
    class_counts = y.value_counts()
    total_samples = len(y)
    
    if method == 'balanced':
        # scikit-learn의 'balanced' 방식
        weights = {}
        for class_label in class_counts.index:
            weights[class_label] = total_samples / (len(class_counts) * class_counts[class_label])
    
    elif method == 'inverse':
        # 역수 기반 가중치
        weights = {}
        for class_label in class_counts.index:
            weights[class_label] = total_samples / class_counts[class_label]
    
    elif method == 'sqrt_inverse':
        # 제곱근 역수 기반 가중치 (덜 극단적)
        weights = {}
        for class_label in class_counts.index:
            weights[class_label] = np.sqrt(total_samples / class_counts[class_label])
    
    else:
        raise ValueError(f"지원하지 않는 가중치 계산 방법: {method}")
    
    logger.info(f"클래스 가중치 계산 완료 ({method}): {weights}")
    return weights


def calculate_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              metric: str = 'f1') -> float:
    """
    최적의 분류 임계값을 계산합니다.
    
    Args:
        y_true: 실제 값
        y_pred_proba: 예측 확률
        metric: 최적화할 지표 ('f1', 'precision', 'recall', 'balanced_accuracy')
        
    Returns:
        최적 임계값
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"지원하지 않는 지표: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"최적 임계값 계산 완료: {best_threshold:.3f} ({metric}={best_score:.4f})")
    return best_threshold


def validate_focal_loss_parameters(alpha: float, gamma: float) -> bool:
    """
    Focal Loss 파라미터의 유효성을 검증합니다.
    
    Args:
        alpha: 클래스 가중치
        gamma: 포커싱 파라미터
        
    Returns:
        유효성 여부
    """
    if not (0.0 <= alpha <= 1.0):
        logger.warning(f"alpha는 0과 1 사이여야 합니다. 현재 값: {alpha}")
        return False
    
    if gamma < 0.0:
        logger.warning(f"gamma는 0 이상이어야 합니다. 현재 값: {gamma}")
        return False
    
    return True


def main():
    """
    테스트용 메인 함수
    """
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    # 극도 불균형 데이터 생성 (849:1 비율 시뮬레이션)
    y_true = np.random.choice([0, 1], n_samples, p=[0.9988, 0.0012])
    y_pred_proba = np.random.random(n_samples)
    
    # Focal Loss 테스트
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_value = focal_loss(y_true, y_pred_proba)
    
    print(f"Focal Loss 테스트 결과: {loss_value:.6f}")
    
    # 클래스 가중치 계산 테스트
    y_series = pd.Series(y_true)
    weights = calculate_class_weights(y_series, method='balanced')
    print(f"클래스 가중치: {weights}")
    
    # 최적 임계값 계산 테스트
    optimal_threshold = calculate_optimal_threshold(y_true, y_pred_proba, metric='f1')
    print(f"최적 임계값: {optimal_threshold:.3f}")


if __name__ == "__main__":
    main() 