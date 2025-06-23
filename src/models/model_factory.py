"""
모델 팩토리

이 모듈은 설정 파일의 model_type에 따라 적절한 모델 인스턴스를 생성합니다.
"""

import logging
from typing import Dict, Any, Type
from .base_model import BaseModel

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFactory:
    """
    모델 팩토리 클래스
    
    설정 파일의 model_type에 따라 적절한 모델 인스턴스를 생성합니다.
    """
    
    _models = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]):
        """
        새로운 모델 타입을 등록합니다.
        
        Args:
            model_type: 모델 타입명
            model_class: 모델 클래스
        """
        cls._models[model_type.lower()] = model_class
        logger.info(f"모델 타입 '{model_type}'이 등록되었습니다.")
    
    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> BaseModel:
        """
        설정에 따라 모델 인스턴스를 생성합니다.
        
        Args:
            config: 설정 딕셔너리
            
        Returns:
            생성된 모델 인스턴스
            
        Raises:
            ValueError: 지원하지 않는 모델 타입인 경우
        """
        model_type = config.get('model', {}).get('model_type', 'xgboost').lower()
        
        if model_type not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(
                f"지원하지 않는 모델 타입: {model_type}. "
                f"사용 가능한 모델: {available_models}"
            )
        
        model_class = cls._models[model_type]
        logger.info(f"모델 타입 '{model_type}'으로 모델을 생성합니다.")
        
        return model_class(config)
    
    @classmethod
    def get_available_models(cls) -> list:
        """
        사용 가능한 모델 타입 목록을 반환합니다.
        
        Returns:
            사용 가능한 모델 타입 목록
        """
        return list(cls._models.keys())
    
    @classmethod
    def validate_model_type(cls, model_type: str) -> bool:
        """
        모델 타입이 지원되는지 확인합니다.
        
        Args:
            model_type: 확인할 모델 타입
            
        Returns:
            지원 여부
        """
        return model_type.lower() in cls._models


# 모델 등록을 위한 데코레이터
def register_model(model_type: str):
    """
    모델 클래스를 자동으로 등록하는 데코레이터
    
    Args:
        model_type: 모델 타입명
    """
    def decorator(model_class: Type[BaseModel]):
        ModelFactory.register_model(model_type, model_class)
        return model_class
    return decorator 