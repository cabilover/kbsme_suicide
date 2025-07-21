# utils 패키지
from .config_manager import ConfigManager
import numbers
import numpy as np
import logging
import os
from datetime import datetime
from typing import List, Optional
import sys
import io
from contextlib import contextmanager

def find_column_with_remainder(df_columns, colname):
    """df_columns에서 colname 또는 remainder__colname이 존재하면 반환"""
    if colname in df_columns:
        return colname
    elif f"remainder__{colname}" in df_columns:
        return f"remainder__{colname}"
    return None


def safe_feature_name(feature_name):
    """피처 이름을 안전하게 변환"""
    return feature_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')


def set_random_seed(seed=42):
    """재현성을 위한 랜덤 시드를 설정"""
    import random
    logger = __import__('logging').getLogger(__name__)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import xgboost as xgb
        xgb.set_config(verbosity=0)
    except ImportError:
        pass
    logger.info(f"랜덤 시드 설정 완료: {seed}")


def safe_float_conversion(value):
    """안전한 float 변환"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_float(value):
    """안전한 float 변환 (기본값 0.0)"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def is_valid_number(value):
    """값이 유효한 숫자인지 확인"""
    if isinstance(value, numbers.Number):
        return True
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


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
    
    logger = logging.getLogger(__name__)
    logger.info(f"로깅 설정 완료 - 레벨: {level}, 파일: {log_file}")


def setup_experiment_logging(
    experiment_type: str,
    model_type: str,
    log_level: str = "INFO",
    capture_console: bool = True,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> str:
    """
    실험별 로깅 설정을 구성합니다.
    
    Args:
        experiment_type: 실험 타입 (hyperparameter_tuning, resampling_experiment 등)
        model_type: 모델 타입 (catboost, xgboost, lightgbm, random_forest 등)
        log_level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        capture_console: 터미널 출력 캡처 여부
        log_format: 로그 메시지 형식
        
    Returns:
        로그 파일 경로
    """
    # 통일된 로그 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_type}_{model_type}_{timestamp}.log"
    log_file_path = f"logs/{log_filename}"
    
    # 로그 디렉토리 생성
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 로깅 설정
    setup_logging(level=log_level, log_file=log_file_path, log_format=log_format)
    
    # 실험 시작 정보 로깅
    logger = logging.getLogger(__name__)
    logger.info(f"=== 실험 시작 ===")
    logger.info(f"실험 타입: {experiment_type}")
    logger.info(f"모델 타입: {model_type}")
    logger.info(f"로그 파일: {log_file_path}")
    logger.info(f"터미널 출력 캡처: {capture_console}")
    logger.info(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"=" * 50)
    
    return log_file_path


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
    
    logger = logging.getLogger(__name__)
    logger.info(f"실험 정보가 {log_file}에 저장되었습니다.")


def log_experiment_summary(
    experiment_type: str,
    model_type: str,
    best_score: float,
    best_params: dict,
    execution_time: float,
    n_trials: int,
    data_info: dict = None,
    log_file_path: str = None
) -> None:
    """
    실험 요약 정보를 로그 파일에 기록합니다.
    
    Args:
        experiment_type: 실험 타입
        model_type: 모델 타입
        best_score: 최고 성능 점수
        best_params: 최적 파라미터
        execution_time: 실행 시간 (초)
        n_trials: 시도 횟수
        data_info: 데이터 정보
        log_file_path: 로그 파일 경로
    """
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"logs/{experiment_type}_{model_type}_{timestamp}.log"
    
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n=== 실험 요약 ===\n")
        f.write(f"실험 타입: {experiment_type}\n")
        f.write(f"모델 타입: {model_type}\n")
        f.write(f"최고 성능: {best_score:.6f}\n")
        f.write(f"실행 시간: {execution_time:.2f}초\n")
        f.write(f"시도 횟수: {n_trials}\n")
        f.write(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"=== 최적 파라미터 ===\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        
        if data_info:
            f.write(f"\n=== 데이터 정보 ===\n")
            for key, value in data_info.items():
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\n" + "=" * 50 + "\n")
    
    logger = logging.getLogger(__name__)
    logger.info(f"실험 요약이 {log_file_path}에 저장되었습니다.")


class ConsoleCapture:
    """
    터미널 출력을 파일로 캡처하는 클래스
    """
    
    def __init__(self, log_file_path: str):
        """
        ConsoleCapture 초기화
        
        Args:
            log_file_path: 로그 파일 경로
        """
        self.log_file_path = log_file_path
        self.original_stdout = None
        self.original_stderr = None
        self.stdout_capture = None
        self.stderr_capture = None
        
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        # 원래 출력 저장
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 파일 핸들러 생성
        log_file = open(self.log_file_path, 'a', encoding='utf-8')
        
        # 출력 캡처 객체 생성
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        
        # 출력 리다이렉트
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        try:
            # 캡처된 출력을 파일에 저장
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                # stdout 캡처 내용 저장
                stdout_content = self.stdout_capture.getvalue()
                if stdout_content:
                    f.write("\n=== STDOUT 캡처 ===\n")
                    f.write(stdout_content)
                    f.write("\n")
                
                # stderr 캡처 내용 저장
                stderr_content = self.stderr_capture.getvalue()
                if stderr_content:
                    f.write("\n=== STDERR 캡처 ===\n")
                    f.write(stderr_content)
                    f.write("\n")
                
                # 예외 정보 저장
                if exc_type is not None:
                    f.write(f"\n=== 예외 발생 ===\n")
                    f.write(f"예외 타입: {exc_type.__name__}\n")
                    f.write(f"예외 메시지: {exc_val}\n")
                    f.write(f"발생 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("\n")
                
                f.write(f"\n=== 실험 종료 ===\n")
                f.write(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")
                
        finally:
            # 원래 출력으로 복원
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            # 캡처 객체 정리
            if self.stdout_capture:
                self.stdout_capture.close()
            if self.stderr_capture:
                self.stderr_capture.close()


@contextmanager
def experiment_logging_context(
    experiment_type: str,
    model_type: str,
    log_level: str = "INFO",
    capture_console: bool = True
):
    """
    실험 로깅을 위한 컨텍스트 매니저
    
    Args:
        experiment_type: 실험 타입
        model_type: 모델 타입
        log_level: 로깅 레벨
        capture_console: 터미널 출력 캡처 여부
        
    Yields:
        로그 파일 경로
    """
    # 실험별 로깅 설정
    log_file_path = setup_experiment_logging(
        experiment_type=experiment_type,
        model_type=model_type,
        log_level=log_level,
        capture_console=capture_console
    )
    
    # 터미널 출력 캡처 설정
    if capture_console:
        with ConsoleCapture(log_file_path):
            yield log_file_path
    else:
        yield log_file_path


__all__ = [
    'ConfigManager', 
    'find_column_with_remainder', 
    'safe_feature_name', 
    'set_random_seed', 
    'safe_float_conversion', 
    'safe_float', 
    'is_valid_number',
    'setup_logging',
    'setup_experiment_logging',
    'get_logger',
    'log_experiment_info',
    'log_experiment_summary',
    'ConsoleCapture',
    'experiment_logging_context'
] 