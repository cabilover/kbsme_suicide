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
            # 전처리 과정에서 추가된 접두사 제거
            clean_target = target
            for prefix in ['pass__', 'num__', 'cat__']:
                if target.startswith(prefix):
                    clean_target = target[len(prefix):]
                    break
            
            # _next_year 접미사가 있는 경우와 없는 경우 모두 처리
            if clean_target.endswith('_next_year'):
                base_target = clean_target.replace('_next_year', '')
                if base_target in ['anxiety_score', 'depress_score', 'sleep_score']:
                    self.regression_targets.append(target)
                elif base_target in ['suicide_t', 'suicide_a']:
                    self.classification_targets.append(target)
            else:
                # _next_year 접미사가 없는 경우도 처리
                if clean_target in ['anxiety_score', 'depress_score', 'sleep_score']:
                    self.regression_targets.append(target)
                elif clean_target in ['suicide_t', 'suicide_a']:
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
        
        주의: 이 메서드는 각 모델 클래스에서 오버라이드되어야 합니다.
        - XGBoost: 범주형 변수를 숫자로 변환
        - CatBoost: 범주형 변수 보존
        - LightGBM: 범주형 변수 보존  
        - Random Forest: 범주형 변수 보존
        
        기본 구현: 최소한의 데이터 검증만 수행
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 데이터프레임 (선택사항)
            
        Returns:
            전처리된 피처 데이터프레임 (또는 튜플)
        """
        logger.info(f"[DEBUG] BaseModel _validate_input_data 입력 X shape: {X.shape}")
        logger.info(f"[DEBUG] BaseModel _validate_input_data 입력 X dtypes: {X.dtypes.value_counts()}")
        
        # 기본적인 데이터 검증만 수행 (각 모델에서 오버라이드 권장)
        X_cleaned = X.copy()
        
        # inf 값 처리 (모든 모델에서 공통적으로 필요한 처리)
        X_cleaned = X_cleaned.replace([np.inf, -np.inf], np.nan)
        
        if y is not None:
            # y가 Series면 DataFrame으로 변환
            if isinstance(y, pd.Series):
                y = y.to_frame()
            
            # y 데이터 처리: 타겟 컬럼은 그대로 유지하되 inf 값만 처리
            y_cleaned = y.copy()
            y_cleaned = y_cleaned.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"[DEBUG] BaseModel _validate_input_data 출력 X shape: {X_cleaned.shape}, y shape: {y_cleaned.shape}")
            return X_cleaned, y_cleaned
        
        logger.info(f"[DEBUG] BaseModel _validate_input_data 출력 X shape: {X_cleaned.shape}")
        return X_cleaned
    
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

    def _find_available_targets(self, y: pd.DataFrame) -> list:
        """
        다양한 접두사와 원본명을 모두 고려하여 y에서 사용 가능한 타겟 컬럼을 찾습니다.
        """
        all_columns = list(y.columns)
        logger.info(f"[DEBUG] _find_available_targets - y 컬럼: {all_columns}")
        logger.info(f"[DEBUG] _find_available_targets - target_columns: {self.target_columns}")
        
        available_targets = []
        for target in self.target_columns:
            candidates = [
                target,
                f"pass__{target}",
                f"remainder__{target}",
                f"num__{target}",
                f"cat__{target}"
            ]
            logger.info(f"[DEBUG] 타겟 {target} 후보: {candidates}")
            
            for cand in candidates:
                if cand in all_columns:
                    available_targets.append(cand)
                    logger.info(f"[DEBUG] 타겟 {target} 매칭됨: {cand}")
                    break
            else:
                logger.warning(f"[DEBUG] 타겟 {target} 매칭 실패. 후보: {candidates}")
        
        logger.info(f"[DEBUG] 최종 사용 가능한 타겟: {available_targets}")
        return available_targets 