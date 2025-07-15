"""
공통 유틸리티 함수 모듈

이 모듈은 프로젝트 전반에서 사용되는 공통 유틸리티 함수들을 포함합니다.
"""

import logging
import numbers
import numpy as np
from typing import List, Optional
import os
from datetime import datetime

# 로깅 설정
logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    프로젝트 전반에서 사용할 공통 로깅 설정을 구성합니다.
    
    Args:
        level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로 (None이면 콘솔만 출력)
        log_format: 로그 메시지 형식
    """
    # 로깅 레벨 설정
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # 로거 설정
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[]
    )
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # 루트 로거에 핸들러 추가
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # 기존 핸들러 제거
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 추가 (지정된 경우)
    if log_file:
        # 로그 디렉토리 생성
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"로깅 설정 완료 - 레벨: {level}, 파일: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    지정된 이름의 로거를 반환합니다.
    
    Args:
        name: 로거 이름
        
    Returns:
        설정된 로거 객체
    """
    return logging.getLogger(name)


def log_experiment_info(
    experiment_name: str,
    parameters: dict,
    metrics: dict,
    log_file: Optional[str] = None
) -> None:
    """
    실험 정보를 로그 파일에 기록합니다.
    
    Args:
        experiment_name: 실험 이름
        parameters: 실험 파라미터
        metrics: 실험 결과 메트릭
        log_file: 로그 파일 경로 (None이면 기본 경로 사용)
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/experiment_{experiment_name}_{timestamp}.log"
    
    # 로그 디렉토리 생성
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== 실험 정보 ===\n")
        f.write(f"실험명: {experiment_name}\n")
        f.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"=== 파라미터 ===\n")
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\n=== 메트릭 ===\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"실험 정보가 {log_file}에 저장되었습니다.")


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
    """Convert value to float if possible; otherwise return np.nan.
    - 변환 불가 시 np.nan 반환 (nan/inf를 0.0으로 바꾸지 않음)
    - MLflow metric 기록에는 직접 사용하지 않는 것이 안전함
    """
    if isinstance(value, numbers.Number):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def safe_float(value):
    """float 변환 + nan/inf/-inf를 0.0으로 변환 (MLflow metric 기록에 안전)
    - safe_float_conversion과 달리 nan/inf/-inf를 0.0으로 바꿔줌
    """
    v = safe_float_conversion(value)
    if v is None or np.isnan(v) or np.isinf(v):
        return 0.0
    return float(v)


def is_valid_number(value):
    """Return True if value is numeric and not NaN/Inf."""
    return (
        isinstance(value, numbers.Number)
        and not np.isnan(value)
        and not np.isinf(value)
    )
