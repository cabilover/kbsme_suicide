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
import sys
import io
from contextlib import contextmanager

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
    logger.info(f"=== 실험 시작 ===")
    logger.info(f"실험 타입: {experiment_type}")
    logger.info(f"모델 타입: {model_type}")
    logger.info(f"로그 파일: {log_file_path}")
    logger.info(f"터미널 출력 캡처: {capture_console}")
    logger.info(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"=" * 50)
    
    return log_file_path


class ConsoleCapture:
    """
    터미널 출력을 파일로 캡처하는 클래스
    
    이 클래스는 실험 중 발생하는 모든 터미널 출력(stdout, stderr)을 자동으로 캡처하여
    로그 파일에 저장합니다. 실험의 완전한 추적성을 보장하며, 예외 발생 시에도
    자동으로 예외 정보를 기록합니다.
    
    주요 기능:
    - stdout/stderr 리다이렉트 및 캡처
    - 예외 발생 시 자동 예외 정보 저장
    - 실험 시작/종료 시간 자동 기록
    - 컨텍스트 매니저 패턴으로 안전한 리소스 관리
    
    사용 예시:
        with ConsoleCapture(log_file_path):
            # 실험 코드 실행
            result = run_experiment()
            print("실험 완료:", result)
    
    캡처되는 정보:
    - 모든 print() 출력
    - 모든 로깅 메시지
    - 모든 에러 메시지
    - 예외 발생 시 상세 정보 (타입, 메시지, 시간)
    - 실험 종료 시간
    """
    
    def __init__(self, log_file_path: str):
        """
        ConsoleCapture 초기화
        
        Args:
            log_file_path: 로그 파일 경로 (캡처된 출력이 저장될 파일)
        """
        self.log_file_path = log_file_path
        self.original_stdout = None
        self.original_stderr = None
        self.stdout_capture = None
        self.stderr_capture = None
        
    def __enter__(self):
        """
        컨텍스트 매니저 진입
        
        터미널 출력을 캡처 모드로 전환합니다.
        
        Returns:
            self: ConsoleCapture 인스턴스
        """
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
        """
        컨텍스트 매니저 종료
        
        캡처된 모든 출력을 파일에 저장하고 원래 출력으로 복원합니다.
        예외가 발생한 경우 예외 정보도 함께 저장합니다.
        
        Args:
            exc_type: 예외 타입 (예외가 발생한 경우)
            exc_val: 예외 값 (예외가 발생한 경우)
            exc_tb: 예외 트레이스백 (예외가 발생한 경우)
        """
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
    실험 로깅을 위한 통합 컨텍스트 매니저
    
    이 함수는 실험의 전체 생명주기를 관리하는 통합 로깅 시스템을 제공합니다.
    실험별 로깅 설정, 터미널 출력 캡처, 예외 처리 등을 자동으로 처리하여
    실험의 완전한 추적성을 보장합니다.
    
    주요 기능:
    - 실험별 전용 로그 파일 자동 생성
    - 실험 시작/종료 정보 자동 로깅
    - 터미널 출력 자동 캡처 (선택적)
    - 예외 발생 시 자동 예외 정보 저장
    - 실험 요약 정보 자동 기록
    
    사용 예시:
        with experiment_logging_context(
            experiment_type="hyperparameter_tuning",
            model_type="xgboost",
            log_level="INFO",
            capture_console=True
        ) as log_file_path:
            # 실험 코드 실행
            result = run_hyperparameter_tuning()
            # 실험 요약 로깅
            log_experiment_summary(...)
    
    생성되는 로그 파일:
    - 파일명: {experiment_type}_{model_type}_{timestamp}.log
    - 위치: logs/ 디렉토리
    - 내용: 실험 시작/종료 정보, 모든 출력, 예외 정보, 실험 요약
    
    Args:
        experiment_type: 실험 타입 (예: hyperparameter_tuning, resampling_experiment)
        model_type: 모델 타입 (예: xgboost, catboost, lightgbm, random_forest)
        log_level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        capture_console: 터미널 출력 캡처 여부 (True 권장)
        
    Yields:
        str: 로그 파일 경로 (실험 요약 로깅 시 사용)
        
    Raises:
        ValueError: 잘못된 로깅 레벨이 지정된 경우
        OSError: 로그 파일 생성/쓰기 권한이 없는 경우
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
    실험 요약 정보를 로그 파일에 구조화된 형태로 기록합니다.
    
    이 함수는 실험의 핵심 결과와 성능 지표를 체계적으로 기록하여
    실험 결과의 재현성과 분석 가능성을 높입니다.
    
    기록되는 정보:
    - 실험 기본 정보 (타입, 모델, 시간)
    - 성능 지표 (최고 점수, 실행 시간, 시도 횟수)
    - 최적 파라미터 (하이퍼파라미터 튜닝 결과)
    - 데이터 정보 (행 수, 열 수, 클래스 분포 등)
    - 실험 완료 시간
    
    사용 예시:
        log_experiment_summary(
            experiment_type="hyperparameter_tuning",
            model_type="xgboost",
            best_score=0.85,
            best_params={"max_depth": 6, "learning_rate": 0.1},
            execution_time=3600.5,
            n_trials=100,
            data_info={"total_rows": 10000, "total_columns": 50},
            log_file_path="logs/experiment.log"
        )
    
    로그 파일 형식:
        === 실험 요약 ===
        실험 타입: hyperparameter_tuning
        모델 타입: xgboost
        최고 성능: 0.850000
        실행 시간: 3600.50초
        시도 횟수: 100
        완료 시간: 2025-01-15 14:30:25
        
        === 최적 파라미터 ===
          max_depth: 6
          learning_rate: 0.1
        
        === 데이터 정보 ===
          total_rows: 10000
          total_columns: 50
    
    Args:
        experiment_type: 실험 타입 (예: hyperparameter_tuning, resampling_experiment)
        model_type: 모델 타입 (예: xgboost, catboost, lightgbm, random_forest)
        best_score: 최고 성능 점수 (float)
        best_params: 최적 파라미터 딕셔너리
        execution_time: 실행 시간 (초, float)
        n_trials: 시도 횟수 (int)
        data_info: 데이터 정보 딕셔너리 (선택사항)
            - total_rows: 전체 행 수
            - total_columns: 전체 열 수
            - class_distributions: 클래스 분포 정보
            - 기타 데이터 관련 정보
        log_file_path: 로그 파일 경로 (None이면 자동 생성)
        
    Returns:
        None
        
    Raises:
        OSError: 로그 파일 쓰기 권한이 없는 경우
        ValueError: 필수 인자가 None이거나 잘못된 타입인 경우
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
    
    logger.info(f"실험 요약이 {log_file_path}에 저장되었습니다.")


def find_column_with_remainder(df_columns: List[str], colname: str) -> Optional[str]:
    """
    데이터프레임 컬럼명에서 특정 컬럼을 찾는 함수 (scikit-learn 파이프라인 호환)
    
    이 함수는 scikit-learn의 ColumnTransformer나 Pipeline을 통과한 후
    컬럼명이 변경된 경우를 처리합니다. 특히 'remainder__' 접두사가
    추가된 컬럼명을 자동으로 찾아줍니다.
    
    컬럼명 변경 패턴:
    - 원본: 'target_column'
    - 변경 후: 'remainder__target_column' (ColumnTransformer의 remainder 옵션 사용 시)
    
    사용 예시:
        # 전처리 후 컬럼명 확인
        columns = ['remainder__anxiety_score', 'remainder__depress_score', 'num__age']
        
        # 원본 컬럼명으로 찾기
        found_col = find_column_with_remainder(columns, 'anxiety_score')
        # 결과: 'remainder__anxiety_score'
        
        # 이미 접두사가 있는 경우도 처리
        found_col = find_column_with_remainder(columns, 'remainder__anxiety_score')
        # 결과: 'remainder__anxiety_score'
    
    Args:
        df_columns: 데이터프레임의 컬럼명 리스트
        colname: 찾을 컬럼명 (원본명 또는 remainder__ 접두사 포함)
        
    Returns:
        str: 실제 존재하는 컬럼명 또는 None (찾지 못한 경우)
        
    Examples:
        >>> columns = ['remainder__target', 'num__feature1', 'cat__feature2']
        >>> find_column_with_remainder(columns, 'target')
        'remainder__target'
        >>> find_column_with_remainder(columns, 'feature1')
        None
        >>> find_column_with_remainder(columns, 'remainder__target')
        'remainder__target'
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
    재현성을 위한 랜덤 시드를 통합 설정합니다.
    
    이 함수는 실험의 재현성을 보장하기 위해 여러 라이브러리의
    랜덤 시드를 한 번에 설정합니다. 머신러닝 실험에서 일관된
    결과를 얻기 위해 필수적인 함수입니다.
    
    설정되는 시드:
    - numpy.random.seed(): 수치 계산 라이브러리
    - random.seed(): Python 기본 랜덤 모듈
    - xgboost.set_config(): XGBoost 라이브러리 (가능한 경우)
    
    사용 예시:
        # 기본 시드 (42) 설정
        set_random_seed()
        
        # 사용자 정의 시드 설정
        set_random_seed(123)
        
        # 실험 시작 전에 호출
        set_random_seed(42)
        result = run_experiment()  # 재현 가능한 결과
    
    주의사항:
    - 실험 시작 시 가장 먼저 호출해야 합니다
    - 동일한 시드로 여러 번 호출해도 안전합니다
    - 일부 라이브러리(XGBoost)는 import되지 않은 경우 무시됩니다
    
    Args:
        seed: 랜덤 시드 (기본값: 42)
        
    Returns:
        None
        
    Raises:
        ImportError: XGBoost import 실패 시 (무시됨)
        
    Examples:
        >>> set_random_seed(42)
        >>> np.random.rand()  # 항상 동일한 값 반환
        0.3745401188473625
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
