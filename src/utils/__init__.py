# utils 패키지
from .config_manager import ConfigManager

# utils.py의 함수들을 직접 정의
def find_column_with_remainder(df_columns, colname):
    """df_columns에서 colname 또는 remainder__colname이 존재하면 반환"""
    if colname in df_columns:
        return colname
    elif f"remainder__{colname}" in df_columns:
        return f"remainder__{colname}"
    return None

def safe_feature_name(feature_name):
    """피처명을 MLflow 로깅에 안전한 형태로 변환"""
    safe_name = feature_name.replace(' ', '_').replace('-', '_').replace('.', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
    return safe_name

def set_random_seed(seed=42):
    """재현성을 위한 랜덤 시드를 설정"""
    import numpy as np
    import random
    import logging
    
    logger = logging.getLogger(__name__)
    
    np.random.seed(seed)
    random.seed(seed)
    
    try:
        import xgboost as xgb
        xgb.set_config(verbosity=0)
    except ImportError:
        pass
    
    logger.info(f"랜덤 시드 설정 완료: {seed}")

def safe_float_conversion(value):
    """Convert value to float if possible; otherwise return np.nan."""
    import numbers
    import numpy as np
    
    if isinstance(value, numbers.Number):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def is_valid_number(value):
    """Return True if value is numeric and not NaN/Inf."""
    import numbers
    import numpy as np
    
    return (
        isinstance(value, numbers.Number)
        and not np.isnan(value)
        and not np.isinf(value)
    )

__all__ = ['ConfigManager', 'find_column_with_remainder', 'safe_feature_name', 'set_random_seed', 'safe_float_conversion', 'is_valid_number'] 