"""
모델 기본 클래스

이 모듈은 모든 머신러닝 모델이 따라야 할 기본 인터페이스를 정의합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod
import warnings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    모든 머신러닝 모델의 기본 클래스
    
    이 클래스는 모든 모델이 구현해야 할 공통 인터페이스를 정의합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        기본 모델 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.models = {}
        self.target_columns = config['features']['target_columns']
        self.regression_targets = []
        self.classification_targets = []
        self.is_fitted = False
        self.model_type = self.__class__.__name__.lower().replace('model', '')
        
        # 타겟 타입 분류
        self._classify_targets()
        
        logger.info(f"{self.model_type.upper()} 모델 초기화 완료")
        logger.info(f"  - 회귀 타겟: {self.regression_targets}")
        logger.info(f"  - 분류 타겟: {self.classification_targets}")
    
    def _classify_targets(self):
        """타겟을 회귀와 분류로 분류합니다."""
        for target in self.target_columns:
            if target.endswith('_next_year'):
                base_target = target.replace('_next_year', '')
                if base_target in ['anxiety_score', 'depress_score', 'sleep_score']:
                    self.regression_targets.append(target)
                elif base_target in ['suicide_t', 'suicide_a']:
                    self.classification_targets.append(target)
    
    @abstractmethod
    def _get_model_params(self, target: str) -> Dict[str, Any]:
        """
        특정 타겟에 대한 모델 파라미터를 반환합니다.
        
        Args:
            target: 타겟 컬럼명
            
        Returns:
            모델 파라미터 딕셔너리
        """
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, 
            X_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> 'BaseModel':
        """
        모델을 학습합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 데이터프레임
            X_val: 검증 피처 데이터프레임 (Early Stopping용, 선택사항)
            y_val: 검증 타겟 데이터프레임 (Early Stopping용, 선택사항)
            
        Returns:
            학습된 모델
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        예측을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            
        Returns:
            예측 결과 데이터프레임
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        분류 문제에서 예측 확률을 반환합니다.
        
        Args:
            X: 피처 데이터프레임
            
        Returns:
            타겟별 예측 확률 딕셔너리
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self, target: str = None, aggregate: bool = False) -> Dict[str, pd.DataFrame]:
        """
        피처 중요도를 반환합니다.
        
        Args:
            target: 특정 타겟 (None이면 모든 타겟)
            aggregate: 모든 타겟의 중요도를 집계할지 여부
            
        Returns:
            피처 중요도 딕셔너리
        """
        pass
    
    def save_model(self, filepath: str):
        """
        모델을 저장합니다.
        
        Args:
            filepath: 저장할 파일 경로
        """
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"모델이 {filepath}에 저장되었습니다.")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseModel':
        """
        저장된 모델을 로드합니다.
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드된 모델
        """
        import joblib
        model = joblib.load(filepath)
        logger.info(f"모델이 {filepath}에서 로드되었습니다.")
        return model
    
    def _validate_input_data(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """
        입력 데이터를 검증하고 전처리합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 데이터프레임 (선택사항)
            
        Returns:
            전처리된 피처 데이터프레임
        """
        # XGBoost 입력에 object 컬럼이 포함되지 않도록 처리
        X = X.select_dtypes(include=['number', 'bool', 'category'])
        
        # NaN, inf 값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        
        if y is not None:
            y = y.select_dtypes(include=['number', 'bool', 'category'])
            y = y.replace([np.inf, -np.inf], np.nan)
            return X, y
        
        return X
    
    def _calculate_scale_pos_weight(self, y: pd.Series) -> float:
        """
        분류 문제에서 양성 클래스 가중치를 계산합니다.
        
        Args:
            y: 타겟 시리즈
            
        Returns:
            양성 클래스 가중치
        """
        if len(y.unique()) != 2:
            return 1.0
        
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        
        if pos_count == 0:
            return 1.0
        
        return neg_count / pos_count
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보를 반환합니다.
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'regression_targets': self.regression_targets,
            'classification_targets': self.classification_targets,
            'num_models': len(self.models)
        } 