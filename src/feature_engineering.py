"""
피처 엔지니어링 모듈

이 모듈은 기존 피처의 유효성을 검증하고 새로운 피처를 생성합니다.
- 기존 피처의 데이터 유출 검증
- 시간 기반 피처 생성
- 데이터 유출 방지 원칙 준수
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
import warnings
from src.utils import find_column_with_remainder
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_existing_features(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, bool]:
    """
    기존 피처의 데이터 유출 여부를 검증합니다.
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        피처별 검증 결과 딕셔너리
    """
    validation_results = {}
    
    # 검증 설정 가져오기
    validation_config = config['features']['validation']
    validation_scope = validation_config['scope']
    sample_size = validation_config['sample_size']
    
    # 시계열 설정 가져오기
    id_column = config['time_series']['id_column']
    year_column = config['time_series']['year_column']
    
    # 설정에서 점수 타겟 가져오기
    target_vars = config.get('features', {}).get('target_variables', {})
    original_targets = target_vars.get('original_targets', {})
    score_targets = original_targets.get('score_targets', ['anxiety_score', 'depress_score', 'sleep_score', 'comp'])
    
    # 검증할 피처 목록 동적 생성
    features_to_validate = []
    for score_target in score_targets:
        features_to_validate.extend([
            f'{score_target}_rolling_mean_2y',
            f'{score_target}_rolling_std_2y',
            f'{score_target}_yoy_change'
        ])
    
    logger.info(f"기존 피처 데이터 유출 검증 시작 (범위: {validation_scope})")
    logger.info(f"검증할 피처: {features_to_validate}")
    
    # 검증할 ID 목록 결정
    if id_column not in df.columns:
        logger.warning(f"ID 컬럼 '{id_column}'을 찾을 수 없습니다. 검증을 건너뜁니다.")
        return {feature: True for feature in features_to_validate}
    
    all_ids = df[id_column].unique()
    if validation_scope == 'none':
        logger.info("검증을 건너뜁니다.")
        return {feature: True for feature in features_to_validate}
    elif validation_scope == 'full':
        ids_to_validate = all_ids
        logger.info(f"전체 ID 검증: {len(ids_to_validate):,}개 ID")
    else:  # 'sample'
        ids_to_validate = all_ids[:sample_size]
        logger.info(f"샘플 ID 검증: {len(ids_to_validate):,}개 ID (전체 {len(all_ids):,}개 중)")
    
    for feature in features_to_validate:
        # 전처리 후 피처 이름 찾기 (여러 가능성 시도)
        feature_column = find_column_with_remainder(df.columns, feature)
        
        if feature_column is None:
            # 원본 이름도 시도
            feature_column = feature
        
        if feature_column not in df.columns:
            validation_results[feature] = False
            logger.warning(f"  {feature}: 컬럼이 존재하지 않음")
            continue
        
        # 데이터 유출 검증: 각 ID별로 연도 순서를 확인
        is_valid = True
        validation_count = 0
        
        for id_val in ids_to_validate:
            id_data = df[df[id_column] == id_val].sort_values(year_column)
            
            if len(id_data) < 2:
                continue
                
            validation_count += 1
            
            # rolling 통계의 경우 첫 번째 값이 NaN이어야 함 (미래 정보 없음)
            if feature.endswith('_rolling_mean_2y') or feature.endswith('_rolling_std_2y'):
                if not pd.isna(id_data[feature_column].iloc[0]):
                    is_valid = False
                    logger.warning(f"  {feature}: ID {id_val}에서 첫 번째 값이 NaN이 아님")
                    if validation_scope == 'sample':
                        logger.warning(f"    샘플 검증에서 발견됨. 전체 검증을 권장합니다.")
                    break
            
            # yoy_change의 경우 첫 번째 값이 NaN이어야 함 (이전 연도 정보 없음)
            elif feature.endswith('_yoy_change'):
                if not pd.isna(id_data[feature_column].iloc[0]):
                    is_valid = False
                    logger.warning(f"  {feature}: ID {id_val}에서 첫 번째 값이 NaN이 아님")
                    if validation_scope == 'sample':
                        logger.warning(f"    샘플 검증에서 발견됨. 전체 검증을 권장합니다.")
                    break
        
        validation_results[feature] = is_valid
        status = "✓" if is_valid else "✗"
        logger.info(f"  {feature}: {status} (검증된 ID: {validation_count}개)")
    
    # 검증 요약
    valid_count = sum(validation_results.values())
    total_count = len(validation_results)
    logger.info(f"검증 완료: {valid_count}/{total_count}개 피처 유효")
    
    if valid_count < total_count and validation_scope == 'sample':
        logger.warning("일부 피처에서 문제가 발견되었습니다. 전체 검증을 권장합니다.")
    
    return validation_results


def create_time_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    시간 기반 피처를 생성합니다.
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        시간 피처가 추가된 데이터프레임
    """
    date_column = config['time_series']['date_column']
    
    logger.info("시간 기반 피처 생성")
    
    df_with_time = df.copy()
    
    # 날짜 컬럼을 datetime으로 변환
    if date_column in df.columns:
        df_with_time[date_column] = pd.to_datetime(df_with_time[date_column])
        
        # 월, 요일, 연도 내 경과 일수 생성
        df_with_time['month'] = df_with_time[date_column].dt.month
        df_with_time['day_of_week'] = df_with_time[date_column].dt.dayofweek
        df_with_time['day_of_year'] = df_with_time[date_column].dt.dayofyear
        
        logger.info("  - month, day_of_week, day_of_year 생성 완료")
    
    return df_with_time


def create_lagged_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    지연 피처를 생성합니다 (설정에서 활성화된 경우).
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        지연 피처가 추가된 데이터프레임
    """
    if not config['features']['enable_lagged_features']:
        return df
    
    id_column = config['time_series']['id_column']
    year_column = config['time_series']['year_column']
    lag_periods = config['features']['lag_periods']
    
    logger.info(f"지연 피처 생성 (lag_periods: {lag_periods})")
    
    df_with_lags = df.copy()
    
    # 지연 피처 대상 컬럼
    lag_columns = config['features']['lagged_features']['columns']
    
    # ID별로 정렬 후 지연 피처 생성
    df_with_lags = df_with_lags.sort_values([id_column, year_column])
    
    for col in lag_columns:
        if col in df.columns:
            for lag in lag_periods:
                lag_col_name = f"{col}_lag_{lag}"
                df_with_lags[lag_col_name] = df_with_lags.groupby(id_column)[col].shift(lag)
                logger.info(f"  - {lag_col_name} 생성 완료")
    
    return df_with_lags


def create_rolling_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    이동 평균/표준편차 피처를 생성합니다 (설정에서 활성화된 경우).
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        이동 통계 피처가 추가된 데이터프레임
    """
    if not config['features']['enable_rolling_stats']:
        return df
    
    id_column = config['time_series']['id_column']
    year_column = config['time_series']['year_column']
    window_sizes = config['features']['rolling_stats']['window_sizes']
    
    logger.info(f"이동 통계 피처 생성 (window_sizes: {window_sizes})")
    
    df_with_rolling = df.copy()
    
    # 이동 통계 대상 컬럼
    rolling_columns = config['features']['rolling_stats']['columns']
    
    # ID별로 정렬
    df_with_rolling = df_with_rolling.sort_values([id_column, year_column])
    
    for col in rolling_columns:
        if col in df.columns:
            for window in window_sizes:
                # 이동 평균
                mean_col_name = f"{col}_rolling_mean_{window}y"
                df_with_rolling[mean_col_name] = df_with_rolling.groupby(id_column)[col].rolling(
                    window=window, min_periods=1
                ).mean().reset_index(0, drop=True)
                
                # 이동 표준편차
                std_col_name = f"{col}_rolling_std_{window}y"
                df_with_rolling[std_col_name] = df_with_rolling.groupby(id_column)[col].rolling(
                    window=window, min_periods=1
                ).std().reset_index(0, drop=True)
                
                logger.info(f"  - {mean_col_name}, {std_col_name} 생성 완료")
    
    return df_with_rolling


def fit_feature_engineering(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    피처 엔지니어링을 수행합니다.
    
    Args:
        df: 훈련 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        피처 엔지니어링이 적용된 데이터프레임과 피처 정보
    """
    # Feature engineering 활성화 여부 확인
    enable_feature_engineering = config['features'].get('enable_feature_engineering', True)
    
    if not enable_feature_engineering:
        logger.info("피처 엔지니어링 비활성화 - 기본 피처만 사용")
        df_engineered = df.copy()
        
        # 피처 정보 수집
        feature_info = {
            'validation_results': {},
            'total_features': len(df_engineered.columns),
            'original_features': len(df.columns),
            'new_features': 0,
            'feature_engineering_disabled': True
        }
        
        logger.info(f"피처 엔지니어링 완료: {feature_info['total_features']}개 피처")
        logger.info(f"  - 원본 피처: {feature_info['original_features']}개")
        logger.info(f"  - 새로 생성된 피처: {feature_info['new_features']}개 (비활성화)")
        logger.info(f"[DEBUG] 피처 엔지니어링 직후 컬럼 목록: {list(df_engineered.columns)}")
        
        return df_engineered, feature_info
    
    logger.info("피처 엔지니어링 시작")
    
    # 1단계: 기존 피처 유효성 검증
    validation_results = validate_existing_features(df, config)
    
    # 2단계: 시간 기반 피처 생성
    df_with_time = create_time_features(df, config)
    
    # 3단계: 지연 피처 생성 (설정에 따라)
    df_with_lags = create_lagged_features(df_with_time, config)
    
    # 4단계: 이동 통계 피처 생성 (설정에 따라)
    df_with_rolling = create_rolling_features(df_with_lags, config)
    
    # 피처 정보 수집
    feature_info = {
        'validation_results': validation_results,
        'total_features': len(df_with_rolling.columns),
        'original_features': len(df.columns),
        'new_features': len(df_with_rolling.columns) - len(df.columns)
    }
    
    logger.info(f"피처 엔지니어링 완료: {feature_info['total_features']}개 피처")
    logger.info(f"  - 원본 피처: {feature_info['original_features']}개")
    logger.info(f"  - 새로 생성된 피처: {feature_info['new_features']}개")
    logger.info(f"[DEBUG] 피처 엔지니어링 직후 컬럼 목록: {list(df_with_rolling.columns)}")
    
    return df_with_rolling, feature_info


def transform_features(df: pd.DataFrame, feature_info: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """
    학습된 피처 엔지니어링 정보를 사용하여 데이터를 변환합니다.
    
    Args:
        df: 변환할 데이터프레임
        feature_info: 학습된 피처 정보
        config: 설정 딕셔너리
        
    Returns:
        피처 엔지니어링이 적용된 데이터프레임
    """
    # Feature engineering 활성화 여부 확인
    enable_feature_engineering = config['features'].get('enable_feature_engineering', True)
    
    if not enable_feature_engineering:
        logger.info("피처 엔지니어링 적용 (기본 피처만 사용)")
        df_transformed = df.copy()
        logger.info(f"피처 엔지니어링 적용 완료: {len(df_transformed.columns)}개 피처")
        return df_transformed
    
    logger.info("피처 엔지니어링 적용")
    
    # 1단계: 시간 기반 피처 생성
    df_with_time = create_time_features(df, config)
    
    # 2단계: 지연 피처 생성 (설정에 따라)
    df_with_lags = create_lagged_features(df_with_time, config)
    
    # 3단계: 이동 통계 피처 생성 (설정에 따라)
    df_with_rolling = create_rolling_features(df_with_lags, config)
    
    logger.info(f"피처 엔지니어링 적용 완료: {len(df_with_rolling.columns)}개 피처")
    
    return df_with_rolling


def get_target_columns(config: Dict[str, Any]) -> List[str]:
    """
    타겟 컬럼 목록을 반환합니다.
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        타겟 컬럼 목록
    """
    return config['features']['target_columns']


def get_target_columns_from_data(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    데이터프레임에서 실제로 존재하는 타겟 컬럼 목록을 반환합니다.
    """
    target_columns = get_target_columns(config)
    all_columns = list(df.columns)
    available_targets = []
    
    logger.info(f"[DEBUG] get_target_columns_from_data - all_columns: {all_columns}")
    logger.info(f"[DEBUG] get_target_columns_from_data - target_columns: {target_columns}")
    
    for target in target_columns:
        # 모든 가능한 접두사 확인
        candidates = [
            target,  # 원본 이름
            f"pass__{target}",  # pass__ 접두사
            f"remainder__{target}",  # remainder__ 접두사
            f"num__{target}",  # num__ 접두사
            f"cat__{target}"  # cat__ 접두사
        ]
        
        found = False
        for candidate in candidates:
            if candidate in all_columns:
                available_targets.append(candidate)
                logger.info(f"타겟 {target} 매칭됨: {candidate}")
                found = True
                break
        
        if not found:
            logger.warning(f"타겟 컬럼을 찾을 수 없습니다: {target} (후보: {candidates})")
    
    logger.info(f"[DEBUG] get_target_columns_from_data - available_targets: {available_targets}")
    return available_targets


def get_feature_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    피처 컬럼 목록을 반환합니다 (타겟은 항상 제외).
    """
    selected_features = config['features'].get('selected_features', [])
    all_columns = list(df.columns)
    target_columns = get_target_columns(config)
    
    if not selected_features:
        # 기존 로직 (타겟, ID, 날짜, 범주형/수치형 제외)
        exclude_columns = target_columns + [
            config['time_series']['id_column'],
            config['time_series']['date_column'],
            config['time_series']['year_column']
        ]
        categorical_columns = config['preprocessing']['categorical_imputation']['columns']
        exclude_columns.extend(categorical_columns)
        numerical_columns = config['preprocessing']['numerical_imputation']['columns']
        exclude_columns.extend(numerical_columns)
        feature_columns = [col for col in all_columns if col not in exclude_columns]
        logger.info(f"기존 로직으로 피처 컬럼 선택: {len(feature_columns)}개")
        return feature_columns
    
    # selected_features 기반으로 피처 필터링
    available_features = []
    for feature in selected_features:
        # 1. 원본 이름으로 찾기
        if feature in all_columns:
            available_features.append(feature)
            continue
            
        # 2. find_column_with_remainder 사용
        col = find_column_with_remainder(all_columns, feature)
        if col:
            available_features.append(col)
            continue
            
        # 3. 전처리 후 접두사가 붙은 컬럼들 찾기
        prefixes = ['num__', 'cat__', 'pass__']
        found = False
        for prefix in prefixes:
            prefixed_col = f"{prefix}{feature}"
            if prefixed_col in all_columns:
                available_features.append(prefixed_col)
                found = True
                break
        
        if not found:
            # 4. OneHotEncoder로 생성된 컬럼들 중 해당 피처에서 나온 것들 찾기
            encoded_cols = [col for col in all_columns if col.startswith(f"{feature}_")]
            available_features.extend(encoded_cols)
            
        if not found and not encoded_cols:
            logger.warning(f"피처를 찾을 수 없습니다: {feature}")
    
    available_features = sorted(list(set(available_features)))
    
    # --- 반드시 타겟 컬럼은 feature set에서 제거 ---
    # 타겟 컬럼의 전처리 후 이름도 고려
    target_columns_processed = []
    for target in target_columns:
        target_columns_processed.append(target)
        # 전처리 후 접두사가 붙은 타겟 컬럼들도 제거
        for prefix in ['num__', 'cat__', 'pass__']:
            prefixed_target = f"{prefix}{target}"
            if prefixed_target in all_columns:
                target_columns_processed.append(prefixed_target)
    
    available_features = [f for f in available_features if f not in target_columns_processed]
    
    # 타겟 컬럼이 selected_features/available_features에 포함되어 있으면 경고
    for target in target_columns:
        if target in selected_features:
            logger.warning(f"[WARNING] selected_features에 타겟 컬럼이 포함되어 있습니다: {target}")
        if target in available_features:
            logger.warning(f"[WARNING] available_features에 타겟 컬럼이 포함되어 있습니다: {target}")
    
    logger.info(f"selected_features 기반 피처 컬럼 선택(타겟 제외): {len(available_features)}개")
    logger.debug(f"선택된 피처: {available_features}")
    return available_features


def main(config=None):
    """피처 엔지니어링 메인 함수"""
    if config is None:
        # 설정 파일 로드
        config_path = Path("configs/base/common.yaml")
        
        if not config_path.exists():
            logger.error("설정 파일을 찾을 수 없습니다!")
            logger.error("해결 방법:")
            logger.error("1. configs/base/common.yaml 파일이 존재하는지 확인하세요")
            logger.error("2. 파일이 없다면 다음 구조로 생성하세요:")
            logger.error("""
features:
  target_variables:
    original_targets:
      score_targets: ["anxiety_score", "depress_score", "sleep_score", "comp"]
      binary_targets: ["suicide_t", "suicide_a"]
    next_year_targets:
      score_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
      binary_targets: ["suicide_t_next_year", "suicide_a_next_year"]
  
  target_types:
    regression_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
    classification_targets: ["suicide_t_next_year", "suicide_a_next_year"]
  
  target_columns: ["suicide_a_next_year"]
  
  validation:
    scope: "sample"
    sample_size: 1000
  
  enable_lagged_features: true
  lag_periods: [1, 2]
  lagged_features:
    columns: ["anxiety_score", "depress_score", "sleep_score"]
  
  enable_rolling_features: true
  rolling_features:
    windows: [2, 3]
    columns: ["anxiety_score", "depress_score", "sleep_score"]
    functions: ["mean", "std"]

time_series:
  id_column: "id"
  year_column: "yov"
  date_column: "date"
            """)
            raise SystemExit(1)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                import yaml
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            logger.error("해결 방법:")
            logger.error("1. configs/base/common.yaml 파일의 YAML 문법을 확인하세요")
            logger.error("2. 들여쓰기가 올바른지 확인하세요 (탭 대신 스페이스 사용)")
            logger.error("3. 콜론(:) 뒤에 공백이 있는지 확인하세요")
            raise SystemExit(1)
        
        # 필수 설정 검증
        required_sections = ['features']
        for section in required_sections:
            if section not in config:
                logger.error(f"설정 파일에 필수 섹션 '{section}'이 없습니다!")
                logger.error("해결 방법:")
                logger.error(f"configs/base/common.yaml 파일에 '{section}' 섹션을 추가하세요")
                raise SystemExit(1)
        
        features_config = config.get('features', {})
        required_feature_sections = ['target_variables', 'target_types']
        missing_sections = [section for section in required_feature_sections if section not in features_config]
        
        if missing_sections:
            logger.error(f"설정 파일에 필수 피처 섹션이 없습니다: {missing_sections}")
            logger.error("해결 방법:")
            logger.error("configs/base/common.yaml 파일의 features 섹션에 다음을 추가하세요:")
            logger.error("""
  target_variables:
    original_targets:
      score_targets: ["anxiety_score", "depress_score", "sleep_score", "comp"]
      binary_targets: ["suicide_t", "suicide_a"]
    next_year_targets:
      score_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
      binary_targets: ["suicide_t_next_year", "suicide_a_next_year"]
  
  target_types:
    regression_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
    classification_targets: ["suicide_t_next_year", "suicide_a_next_year"]
            """)
            raise SystemExit(1)
    
    # 데이터 로드
    data_path = Path("data/processed/processed_data_with_features.csv")
    if not data_path.exists():
        logger.error("전처리된 데이터 파일을 찾을 수 없습니다!")
        logger.error("해결 방법:")
        logger.error("1. data/processed/processed_data_with_features.csv 파일이 존재하는지 확인하세요")
        logger.error("2. 파일이 없다면 먼저 src/data_analysis.py를 실행하여 데이터를 전처리하세요")
        raise SystemExit(1)
    
    df = pd.read_csv(data_path)
    logger.info(f"데이터 로드 완료: {df.shape}")
    
    # 피처 엔지니어링 실행
    df_engineered, feature_info = fit_feature_engineering(df, config)
    
    # 결과 저장
    output_path = Path("data/processed/processed_data_with_features.csv")
    df_engineered.to_csv(output_path, index=False)
    logger.info(f"피처 엔지니어링 완료: {df_engineered.shape}")
    logger.info(f"결과 저장: {output_path}")
    
    # 타겟 컬럼 정보 출력
    target_columns = get_target_columns(config)
    logger.info(f"타겟 컬럼: {target_columns}")
    
    print("✅ 설정 파일 검증 완료 - 피처 엔지니어링이 정상적으로 완료되었습니다.")


if __name__ == "__main__":
    main() 