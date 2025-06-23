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
        피처별 유효성 검증 결과
    """
    # 전처리 후 컬럼 이름 찾기 (여러 가능성 시도)
    id_column = find_column_with_remainder(df.columns, config['time_series']['id_column'])
    year_column = find_column_with_remainder(df.columns, config['time_series']['year_column'])
    
    # 컬럼을 찾지 못한 경우 원본 이름도 시도
    if id_column is None:
        id_column = config['time_series']['id_column']
    if year_column is None:
        year_column = config['time_series']['year_column']
    
    # 검증 설정 가져오기
    validation_config = config['features'].get('validation', {})
    validation_scope = validation_config.get('scope', 'sample')  # 'sample', 'full', 'none'
    sample_size = validation_config.get('sample_size', 10)
    
    validation_results = {}
    
    # 검증할 피처 목록
    features_to_validate = [
        'anxiety_score_rolling_mean_2y',
        'anxiety_score_rolling_std_2y', 
        'anxiety_score_yoy_change',
        'depress_score_rolling_mean_2y',
        'depress_score_rolling_std_2y',
        'depress_score_yoy_change',
        'sleep_score_rolling_mean_2y',
        'sleep_score_rolling_std_2y',
        'sleep_score_yoy_change'
    ]
    
    logger.info(f"기존 피처 데이터 유출 검증 시작 (범위: {validation_scope})")
    
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
    for target in target_columns:
        # utils.py의 find_column_with_remainder 사용 + 'pass__' 접두사도 체크
        found_col = find_column_with_remainder(all_columns, target)
        if found_col:
            available_targets.append(found_col)
        else:
            # 'pass__' 접두사도 체크
            pass_col = f"pass__{target}"
            if pass_col in all_columns:
                available_targets.append(pass_col)
            else:
                logger.warning(f"타겟 컬럼을 찾을 수 없습니다: {target}")
    return available_targets


def get_feature_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    피처 컬럼 목록을 반환합니다 (타겟 제외).
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        피처 컬럼 목록
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
        col = find_column_with_remainder(all_columns, feature)
        if col:
            available_features.append(col)
        else:
            # OneHotEncoder로 생성된 컬럼들 중 해당 피처에서 나온 것들 찾기
            encoded_cols = [col for col in all_columns if col.startswith(f"{feature}_")]
            available_features.extend(encoded_cols)
    available_features = sorted(list(set(available_features)))
    # 타겟 컬럼이 selected_features에 포함되어 있으면 경고
    for target in target_columns:
        if target in selected_features:
            logger.warning(f"[WARNING] selected_features에 타겟 컬럼이 포함되어 있습니다: {target}")
        if target in available_features:
            logger.warning(f"[WARNING] available_features에 타겟 컬럼이 포함되어 있습니다: {target}")
    logger.info(f"selected_features 기반 피처 컬럼 선택: {len(available_features)}개")
    logger.debug(f"선택된 피처: {available_features}")
    return available_features


def main():
    """
    테스트용 메인 함수
    """
    import yaml
    
    # 설정 로드
    with open("configs/default_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 테스트 데이터 로드
    data_path = "data/processed/processed_data_with_features.csv"
    df = pd.read_csv(data_path, nrows=1000)
    
    logger.info(f"테스트 데이터 로드: {len(df):,} 행")
    
    # 피처 엔지니어링 수행
    df_engineered, feature_info = fit_feature_engineering(df, config)
    
    # 피처 컬럼 확인
    feature_columns = get_feature_columns(df_engineered, config)
    
    logger.info("피처 엔지니어링 테스트 완료!")


if __name__ == "__main__":
    main() 