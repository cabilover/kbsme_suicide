"""
공통 유틸리티 함수 모듈

이 모듈은 프로젝트 전반에서 사용되는 공통 유틸리티 함수들을 포함합니다.
"""

import logging
import numbers
import numpy as np
from typing import List, Optional

# 로깅 설정
logger = logging.getLogger(__name__)


def find_column_with_remainder(df_columns: List[str], colname: str) -> Optional[str]:
    """
    df_columns에서 colname 또는 remainder__colname이 존재하면 반환
    
    Args:
        df_columns: 데이터프레임 컬럼명 리스트
        colname: 찾을 컬럼명
        
    Returns:
        실제 존재하는 컬럼명(str) 또는 None
    """
    if colname in df_columns:
        return colname
    elif f"remainder__{colname}" in df_columns:
        return f"remainder__{colname}"
    return None


def safe_feature_name(feature_name: str) -> str:
    """
    피처명을 MLflow 로깅에 안전한 형태로 변환합니다.
    
    Args:
        feature_name: 원본 피처명
        
    Returns:
        안전한 피처명
    """
    # 특수문자 제거 및 언더스코어로 대체
    safe_name = feature_name.replace(' ', '_').replace('-', '_').replace('.', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    return safe_name


def set_random_seed(seed: int = 42):
    """
    재현성을 위한 랜덤 시드를 설정합니다.
    
    Args:
        seed: 랜덤 시드
    """
    import numpy as np
    import random
    
    np.random.seed(seed)
    random.seed(seed)
    
    # XGBoost 시드 설정 (가능한 경우)
    try:
        import xgboost as xgb
        xgb.set_config(verbosity=0)
    except ImportError:
        pass
    
    logger.info(f"랜덤 시드 설정 완료: {seed}")


def safe_float_conversion(value):
    """Convert value to float if possible; otherwise return np.nan."""
    if isinstance(value, numbers.Number):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def is_valid_number(value):
    """Return True if value is numeric and not NaN/Inf."""
    return (
        isinstance(value, numbers.Number)
        and not np.isnan(value)
        and not np.isinf(value)
    )
