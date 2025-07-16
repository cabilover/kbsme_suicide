"""
데이터 분할 전략 모듈

이 모듈은 시계열 데이터의 특성을 고려한 데이터 분할 전략을 구현합니다.
- ID 기반 최종 테스트 세트 분리
- 시간 기반 Walk-Forward 검증
- 개인 기반 그룹 분할
- 데이터 유출 방지 원칙 준수
"""

import pandas as pd
import numpy as np
from typing import Tuple, Generator, Dict, Any
import logging
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import yaml
from pathlib import Path
from src.utils import setup_logging

# 로깅 설정
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    """
    설정 파일을 로드합니다.
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def split_test_set(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    ID 기반으로 최종 테스트 세트를 분리합니다.
    
    Args:
        df: 전체 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        train_val_df: 훈련/검증용 데이터프레임
        test_df: 최종 테스트용 데이터프레임
        test_ids: 테스트 세트에 포함된 ID 배열
    """
    test_ratio = config['data_split']['test_ids_ratio']
    random_state = config['data_split']['random_state']
    id_column = config['time_series']['id_column']
    
    # 고유한 ID 목록 추출
    unique_ids = df[id_column].unique()
    
    # ID 기반으로 train/val와 test 분할
    splitter = GroupShuffleSplit(
        n_splits=1, 
        test_size=test_ratio, 
        random_state=random_state
    )
    
    # 더미 그룹 생성 (ID별로 그룹화)
    groups = df[id_column].values
    
    # 분할 수행
    train_val_indices, test_indices = next(splitter.split(df, groups=groups))
    
    # 테스트 세트에 포함된 ID 추출
    test_ids = df.iloc[test_indices][id_column].unique()
    
    # 데이터프레임 분할
    train_val_df = df.iloc[train_val_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    logger.info(f"데이터 분할 완료:")
    logger.info(f"  - 전체 데이터: {len(df):,} 행")
    logger.info(f"  - 훈련/검증 데이터: {len(train_val_df):,} 행")
    logger.info(f"  - 테스트 데이터: {len(test_df):,} 행")
    logger.info(f"  - 테스트 ID 수: {len(test_ids):,}개")
    
    return train_val_df, test_df, test_ids


def get_time_series_folds(df: pd.DataFrame, config: Dict[str, Any]) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], None, None]:
    """
    시간 기반 Walk-Forward 교차 검증 폴드를 생성합니다.
    
    Args:
        df: 훈련/검증용 데이터프레임
        config: 설정 딕셔너리
        
    Yields:
        train_fold_df: 현재 폴드의 훈련 데이터
        val_fold_df: 현재 폴드의 검증 데이터
        fold_info: 폴드 정보 딕셔너리
    """
    year_column = config['time_series']['year_column']
    id_column = config['time_series']['id_column']
    val_ratio = config['validation']['val_ids_ratio']
    random_state = config['data_split']['random_state']
    validation_start_year = config['validation']['validation_start_year']
    min_train_years = config['validation']['min_train_years']
    max_train_years = config['validation']['max_train_years']
    
    # 연도 범위 확인 (float를 int로 변환)
    available_years = sorted([int(year) for year in df[year_column].unique()])
    logger.info(f"사용 가능한 연도: {available_years}")
    
    # 검증 시작 연도부터 끝까지 순회
    for val_year in range(validation_start_year, max(available_years) + 1):
        if val_year not in available_years:
            continue
            
        # 훈련 기간 계산
        train_start_year = max(min(available_years), val_year - max_train_years)
        train_end_year = val_year - 1
        
        # 최소 훈련 연도 수 확인
        if train_end_year - train_start_year + 1 < min_train_years:
            continue
            
        # 훈련 데이터 추출 (float 비교를 위해 변환)
        train_mask = (df[year_column].astype(int) >= train_start_year) & (df[year_column].astype(int) <= train_end_year)
        train_data = df[train_mask].copy()
        
        # 검증 데이터 추출 (float 비교를 위해 변환)
        val_mask = df[year_column].astype(int) == val_year
        val_data = df[val_mask].copy()
        
        if len(train_data) == 0 or len(val_data) == 0:
            continue
            
        # 훈련 기간에 해당하는 ID들을 train/val로 분할
        train_ids = train_data[id_column].unique()
        
        # ID 기반 분할
        splitter = GroupShuffleSplit(
            n_splits=1, 
            test_size=val_ratio, 
            random_state=random_state
        )
        
        # 더미 그룹 생성
        groups = train_data[id_column].values
        
        # 분할 수행
        train_indices, val_indices = next(splitter.split(train_data, groups=groups))
        
        # 최종 훈련/검증 데이터 추출
        train_fold_df = train_data.iloc[train_indices].copy()
        val_fold_df = val_data[val_data[id_column].isin(train_data.iloc[val_indices][id_column].unique())].copy()
        
        # 폴드 정보 생성
        fold_info = {
            'fold_year': val_year,
            'train_start_year': train_start_year,
            'train_end_year': train_end_year,
            'train_ids_count': len(train_fold_df[id_column].unique()),
            'val_ids_count': len(val_fold_df[id_column].unique()),
            'train_samples': len(train_fold_df),
            'val_samples': len(val_fold_df),
            'train_years': train_end_year - train_start_year + 1,
            'strategy': 'time_series_walk_forward'
        }
        
        logger.info(f"폴드 생성: {val_year}년 검증")
        logger.info(f"  - 훈련 기간: {train_start_year}-{train_end_year} ({fold_info['train_years']}년)")
        logger.info(f"  - 훈련 데이터: {fold_info['train_samples']:,} 행 ({fold_info['train_ids_count']:,}개 ID)")
        logger.info(f"  - 검증 데이터: {fold_info['val_samples']:,} 행 ({fold_info['val_ids_count']:,}개 ID)")
        
        yield train_fold_df, val_fold_df, fold_info


def get_group_kfold_splits(df: pd.DataFrame, config: Dict[str, Any]) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], None, None]:
    """
    순수 그룹 기반 K-Fold 교차 검증 폴드를 생성합니다.
    (시간 순서는 고려하지 않고 ID만을 기준으로 분할)
    
    Args:
        df: 훈련/검증용 데이터프레임
        config: 설정 딕셔너리
        
    Yields:
        train_fold_df: 현재 폴드의 훈련 데이터
        val_fold_df: 현재 폴드의 검증 데이터
        fold_info: 폴드 정보 딕셔너리
    """
    id_column = config['time_series']['id_column']
    year_column = config['time_series']['year_column']
    num_folds = config['validation']['num_cv_folds']
    
    # 고유한 ID 목록 추출
    unique_ids = df[id_column].unique()
    logger.info(f"그룹 K-Fold 분할: {len(unique_ids):,}개 ID를 {num_folds}개 폴드로 분할")
    
    # GroupKFold 생성
    group_kfold = GroupKFold(n_splits=num_folds)
    
    # ID별로 그룹화하여 폴드 생성
    groups = df[id_column].values
    
    for fold_idx, (train_indices, val_indices) in enumerate(group_kfold.split(df, groups=groups), 1):
        # 훈련/검증 데이터 추출
        train_fold_df = df.iloc[train_indices].copy()
        val_fold_df = df.iloc[val_indices].copy()
        
        # 폴드 정보 생성 (연도 범위는 int로 변환)
        fold_info = {
            'fold_idx': fold_idx,
            'train_ids_count': len(train_fold_df[id_column].unique()),
            'val_ids_count': len(val_fold_df[id_column].unique()),
            'train_samples': len(train_fold_df),
            'val_samples': len(val_fold_df),
            'train_years_range': (int(train_fold_df[year_column].min()), int(train_fold_df[year_column].max())),
            'val_years_range': (int(val_fold_df[year_column].min()), int(val_fold_df[year_column].max())),
            'strategy': 'group_kfold'
        }
        
        logger.info(f"그룹 K-Fold 폴드 {fold_idx} 생성 완료")
        logger.info(f"  - 훈련 데이터: {fold_info['train_samples']:,} 행 ({fold_info['train_ids_count']:,}개 ID)")
        logger.info(f"  - 검증 데이터: {fold_info['val_samples']:,} 행 ({fold_info['val_ids_count']:,}개 ID)")
        logger.info(f"  - 훈련 연도 범위: {fold_info['train_years_range']}")
        logger.info(f"  - 검증 연도 범위: {fold_info['val_years_range']}")
        
        yield train_fold_df, val_fold_df, fold_info


def get_time_series_group_kfold_splits(df: pd.DataFrame, config: Dict[str, Any]) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], None, None]:
    """
    시간과 그룹을 모두 고려한 K-Fold 교차 검증 폴드를 생성합니다.
    (가장 복잡하지만 이상적인 전략: ID 겹침 방지 + 시간 순서 보장)
    
    Args:
        df: 훈련/검증용 데이터프레임
        config: 설정 딕셔너리
        
    Yields:
        train_fold_df: 현재 폴드의 훈련 데이터
        val_fold_df: 현재 폴드의 검증 데이터
        fold_info: 폴드 정보 딕셔너리
    """
    id_column = config['time_series']['id_column']
    year_column = config['time_series']['year_column']
    num_folds = config['validation']['num_cv_folds']
    val_ratio = config['validation']['val_ids_ratio']
    random_state = config['data_split']['random_state']
    
    # 고유한 ID 목록 추출
    unique_ids = df[id_column].unique()
    available_years = sorted([int(year) for year in df[year_column].unique()])
    
    logger.info(f"시간-그룹 K-Fold 분할: {len(unique_ids):,}개 ID를 {num_folds}개 폴드로 분할")
    logger.info(f"사용 가능한 연도: {available_years}")
    
    # 1단계: ID를 num_folds로 분할
    group_kfold = GroupKFold(n_splits=num_folds)
    groups = df[id_column].values
    
    for fold_idx, (train_indices, val_indices) in enumerate(group_kfold.split(df, groups=groups), 1):
        # 2단계: 각 폴드 내에서 시간 기반 분할
        train_val_data = df.iloc[train_indices].copy()
        val_ids = df.iloc[val_indices][id_column].unique()
        
        # 검증 ID의 데이터만 추출
        val_data = df[df[id_column].isin(val_ids)].copy()
        
        # 검증 ID의 데이터를 시간 순서로 분할
        # 각 ID별로 중간 연도를 기준으로 훈련/검증 분할
        train_fold_df = pd.DataFrame()
        val_fold_df = pd.DataFrame()
        
        for val_id in val_ids:
            id_data = val_data[val_data[id_column] == val_id].copy()
            id_years = sorted([int(year) for year in id_data[year_column].unique()])
            
            if len(id_years) < 2:
                # 데이터가 1년 이하인 경우 훈련에 포함
                train_fold_df = pd.concat([train_fold_df, id_data], ignore_index=True)
                continue
            
            # 중간 연도를 기준으로 분할
            mid_year_idx = len(id_years) // 2
            train_years = id_years[:mid_year_idx]
            val_years = id_years[mid_year_idx:]
            
            # 훈련/검증 데이터 분할 (float 비교를 위해 변환)
            train_mask = id_data[year_column].astype(int).isin(train_years)
            val_mask = id_data[year_column].astype(int).isin(val_years)
            
            train_fold_df = pd.concat([train_fold_df, id_data[train_mask]], ignore_index=True)
            val_fold_df = pd.concat([val_fold_df, id_data[val_mask]], ignore_index=True)
        
        # 3단계: 훈련 ID에서 추가 훈련 데이터 생성
        train_ids = train_val_data[id_column].unique()
        train_ids = [id for id in train_ids if id not in val_ids]
        
        if len(train_ids) > 0:
            additional_train_data = train_val_data[train_val_data[id_column].isin(train_ids)].copy()
            train_fold_df = pd.concat([train_fold_df, additional_train_data], ignore_index=True)
        
        # 폴드 정보 생성 (연도 범위는 int로 변환)
        train_years_range = None
        val_years_range = None
        
        if len(train_fold_df) > 0:
            train_years_range = (int(train_fold_df[year_column].min()), int(train_fold_df[year_column].max()))
        if len(val_fold_df) > 0:
            val_years_range = (int(val_fold_df[year_column].min()), int(val_fold_df[year_column].max()))
        
        fold_info = {
            'fold_idx': fold_idx,
            'train_ids_count': len(train_fold_df[id_column].unique()),
            'val_ids_count': len(val_fold_df[id_column].unique()),
            'train_samples': len(train_fold_df),
            'val_samples': len(val_fold_df),
            'train_years_range': train_years_range,
            'val_years_range': val_years_range,
            'strategy': 'time_series_group_kfold'
        }
        
        logger.info(f"시간-그룹 K-Fold 폴드 {fold_idx} 생성 완료")
        logger.info(f"  - 훈련 데이터: {fold_info['train_samples']:,} 행 ({fold_info['train_ids_count']:,}개 ID)")
        logger.info(f"  - 검증 데이터: {fold_info['val_samples']:,} 행 ({fold_info['val_ids_count']:,}개 ID)")
        if fold_info['train_years_range'] is not None:
            logger.info(f"  - 훈련 연도 범위: {fold_info['train_years_range']}")
        if fold_info['val_years_range'] is not None:
            logger.info(f"  - 검증 연도 범위: {fold_info['val_years_range']}")
        
        yield train_fold_df, val_fold_df, fold_info


def get_cv_splits(df: pd.DataFrame, config: Dict[str, Any]) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], None, None]:
    """
    교차 검증 분할을 생성하는 메인 함수입니다.
    
    Args:
        df: 훈련/검증용 데이터프레임
        config: 설정 딕셔너리
        
    Yields:
        train_fold_df: 현재 폴드의 훈련 데이터
        val_fold_df: 현재 폴드의 검증 데이터
        fold_info: 폴드 정보 딕셔너리
    """
    strategy = config['validation']['strategy']
    
    if strategy == "time_series_walk_forward":
        logger.info("시간 기반 Walk-Forward 검증 전략 사용")
        yield from get_time_series_folds(df, config)
    elif strategy == "time_series_group_kfold":
        logger.info("시간-그룹 기반 K-Fold 검증 전략 사용")
        yield from get_time_series_group_kfold_splits(df, config)
    elif strategy == "group_kfold":
        logger.info("순수 그룹 기반 K-Fold 검증 전략 사용")
        yield from get_group_kfold_splits(df, config)
    elif strategy == "stratified_group_kfold":
        logger.info("ID 기반 Stratified Group K-Fold 검증 전략 사용")
        yield from get_stratified_group_kfold_splits(df, config)
    else:
        raise ValueError(f"지원하지 않는 검증 전략: {strategy}")


def validate_splits(train_val_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """
    분할 결과의 유효성을 검증합니다.
    
    Args:
        train_val_df: 훈련/검증 데이터
        test_df: 테스트 데이터
        config: 설정 딕셔너리
        
    Returns:
        검증 통과 여부
    """
    id_column = config['time_series']['id_column']
    
    # 1. ID 겹침 확인
    train_val_ids = set(train_val_df[id_column].unique())
    test_ids = set(test_df[id_column].unique())
    
    if train_val_ids & test_ids:
        logger.error("훈련/검증 세트와 테스트 세트에 겹치는 ID가 있습니다!")
        return False
    
    # 2. 데이터 손실 확인
    total_ids = len(train_val_ids | test_ids)
    original_ids = len(train_val_df[id_column].unique()) + len(test_df[id_column].unique())
    
    if total_ids != original_ids:
        logger.error("분할 과정에서 데이터 손실이 발생했습니다!")
        return False
    
    logger.info("분할 검증 통과 ✓")
    return True


def log_splits_info(train_val_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    분할 정보를 로깅하고 MLflow용 정보를 반환합니다.
    
    Args:
        train_val_df: 훈련/검증 데이터
        test_df: 테스트 데이터
        config: 설정 딕셔너리
        
    Returns:
        MLflow 로깅용 정보 딕셔너리
    """
    id_column = config['time_series']['id_column']
    year_column = config['time_series']['year_column']
    
    # 기본 통계
    splits_info = {
        'total_samples': len(train_val_df) + len(test_df),
        'train_val_samples': len(train_val_df),
        'test_samples': len(test_df),
        'total_ids': len(train_val_df[id_column].unique()) + len(test_df[id_column].unique()),
        'train_val_ids': len(train_val_df[id_column].unique()),
        'test_ids': len(test_df[id_column].unique()),
        'test_ratio': config['data_split']['test_ids_ratio'],
        'validation_strategy': config['validation']['strategy'],
        'year_range': {
            'train_val': (train_val_df[year_column].min(), train_val_df[year_column].max()),
            'test': (test_df[year_column].min(), test_df[year_column].max())
        }
    }
    
    # 로깅
    if config['logging']['log_splits_info']:
        logger.info("=== 데이터 분할 정보 ===")
        logger.info(f"전체 샘플 수: {splits_info['total_samples']:,}")
        logger.info(f"훈련/검증 샘플 수: {splits_info['train_val_samples']:,}")
        logger.info(f"테스트 샘플 수: {splits_info['test_samples']:,}")
        logger.info(f"전체 ID 수: {splits_info['total_ids']:,}")
        logger.info(f"훈련/검증 ID 수: {splits_info['train_val_ids']:,}")
        logger.info(f"테스트 ID 수: {splits_info['test_ids']:,}")
        logger.info(f"테스트 비율: {splits_info['test_ratio']:.1%}")
        logger.info(f"검증 전략: {splits_info['validation_strategy']}")
        logger.info(f"훈련/검증 연도 범위: {splits_info['year_range']['train_val']}")
        logger.info(f"테스트 연도 범위: {splits_info['year_range']['test']}")
    
    return splits_info


def calculate_id_class_ratios(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """
    각 ID별 클래스 비율을 계산합니다.
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        ID별 양성 클래스 비율 시리즈
    """
    id_column = config['time_series']['id_column']
    target_columns = config['features']['target_columns']
    
    # 첫 번째 분류 타겟 사용 (보통 suicide_a_next_year)
    target_col = target_columns[0] if target_columns else 'suicide_a_next_year'
    
    # ID별 양성 클래스 비율 계산
    id_class_ratios = df.groupby(id_column)[target_col].agg(lambda x: (x == 1).mean())
    
    logger.info(f"ID별 클래스 비율 계산 완료:")
    logger.info(f"  - 총 ID 수: {len(id_class_ratios):,}")
    logger.info(f"  - 양성 클래스가 있는 ID 수: {(id_class_ratios > 0).sum():,}")
    logger.info(f"  - 양성 클래스 비율 범위: {id_class_ratios.min():.4f} ~ {id_class_ratios.max():.4f}")
    
    return id_class_ratios


def split_fold_data(fold_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    폴드 데이터를 훈련/검증으로 분할합니다.
    
    Args:
        fold_data: 폴드 데이터
        config: 설정 딕셔너리
        
    Returns:
        train_fold_df: 훈련 데이터
        val_fold_df: 검증 데이터
    """
    id_column = config['time_series']['id_column']
    val_ratio = config['validation']['val_ids_ratio']
    random_state = config['data_split']['random_state']
    
    # 고유한 ID 목록 추출
    unique_ids = fold_data[id_column].unique()
    
    # ID 기반 분할
    splitter = GroupShuffleSplit(
        n_splits=1, 
        test_size=val_ratio, 
        random_state=random_state
    )
    
    # 더미 그룹 생성
    groups = fold_data[id_column].values
    
    # 분할 수행
    train_indices, val_indices = next(splitter.split(fold_data, groups=groups))
    
    # 데이터프레임 분할
    train_fold_df = fold_data.iloc[train_indices].copy()
    val_fold_df = fold_data.iloc[val_indices].copy()
    
    return train_fold_df, val_fold_df


def get_stratified_group_kfold_splits(df: pd.DataFrame, config: Dict[str, Any]) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]], None, None]:
    """
    ID 기반 Stratified Group K-Fold 교차 검증 폴드를 생성합니다.
    각 폴드의 클래스 비율을 균형있게 유지하면서 ID 겹침을 방지합니다.
    
    Args:
        df: 훈련/검증용 데이터프레임
        config: 설정 딕셔너리
        
    Yields:
        train_fold_df: 현재 폴드의 훈련 데이터
        val_fold_df: 현재 폴드의 검증 데이터
        fold_info: 폴드 정보 딕셔너리
    """
    id_column = config['time_series']['id_column']
    year_column = config['time_series']['year_column']
    num_folds = config['validation']['num_cv_folds']
    stratification_config = config['validation'].get('stratification', {})
    min_positive_samples = stratification_config.get('min_positive_samples_per_fold', 1)
    
    logger.info(f"ID 기반 Stratified Group K-Fold 분할 시작")
    logger.info(f"  - 총 데이터: {len(df):,} 행")
    logger.info(f"  - 폴드 수: {num_folds}")
    logger.info(f"  - 최소 양성 샘플/폴드: {min_positive_samples}")
    
    # 1. ID별 클래스 비율 계산
    id_class_ratios = calculate_id_class_ratios(df, config)
    
    # 2. 클래스 비율에 따른 ID 그룹화
    positive_ids = id_class_ratios[id_class_ratios > 0].index.tolist()
    negative_ids = id_class_ratios[id_class_ratios == 0].index.tolist()
    
    logger.info(f"ID 그룹화 완료:")
    logger.info(f"  - 양성 클래스 ID 수: {len(positive_ids):,}")
    logger.info(f"  - 음성 클래스 ID 수: {len(negative_ids):,}")
    
    # 3. 양성 ID가 충분한지 확인
    if len(positive_ids) < num_folds * min_positive_samples:
        logger.warning(f"양성 ID 수({len(positive_ids)})가 폴드 수({num_folds}) * 최소 샘플({min_positive_samples})보다 적습니다.")
        logger.warning("일반 Group K-Fold로 대체합니다.")
        yield from get_group_kfold_splits(df, config)
        return
    
    # 4. 균형잡힌 폴드 생성
    # 양성 ID를 폴드에 균등 분배
    positive_folds = np.array_split(positive_ids, num_folds)
    
    # 음성 ID를 폴드에 균등 분배
    negative_folds = np.array_split(negative_ids, num_folds)
    
    # 5. 각 폴드 생성
    for fold_idx in range(num_folds):
        # 현재 폴드의 ID들
        fold_positive_ids = positive_folds[fold_idx].tolist()
        fold_negative_ids = negative_folds[fold_idx].tolist()
        
        # 폴드 데이터 생성
        fold_ids = fold_positive_ids + fold_negative_ids
        fold_data = df[df[id_column].isin(fold_ids)].copy()
        
        # 훈련/검증 분할
        train_fold_df, val_fold_df = split_fold_data(fold_data, config)
        
        # 폴드 정보 생성
        fold_info = {
            'fold_idx': fold_idx + 1,
            'strategy': 'stratified_group_kfold',
            'train_ids_count': len(train_fold_df[id_column].unique()),
            'val_ids_count': len(val_fold_df[id_column].unique()),
            'train_samples': len(train_fold_df),
            'val_samples': len(val_fold_df),
            'train_years_range': (int(train_fold_df[year_column].min()), int(train_fold_df[year_column].max())),
            'val_years_range': (int(val_fold_df[year_column].min()), int(val_fold_df[year_column].max())),
            'positive_ids_in_fold': len(fold_positive_ids),
            'negative_ids_in_fold': len(fold_negative_ids),
            'total_ids_in_fold': len(fold_ids)
        }
        
        # 클래스 분포 로깅
        target_columns = config['features']['target_columns']
        target_col = target_columns[0] if target_columns else 'suicide_a_next_year'
        
        train_positive_ratio = (train_fold_df[target_col] == 1).mean()
        val_positive_ratio = (val_fold_df[target_col] == 1).mean()
        
        logger.info(f"Stratified Group K-Fold 폴드 {fold_idx + 1} 생성 완료")
        logger.info(f"  - 훈련 데이터: {fold_info['train_samples']:,} 행 ({fold_info['train_ids_count']:,}개 ID)")
        logger.info(f"  - 검증 데이터: {fold_info['val_samples']:,} 행 ({fold_info['val_ids_count']:,}개 ID)")
        logger.info(f"  - 훈련 연도 범위: {fold_info['train_years_range']}")
        logger.info(f"  - 검증 연도 범위: {fold_info['val_years_range']}")
        logger.info(f"  - 폴드 내 양성 ID: {fold_info['positive_ids_in_fold']}개")
        logger.info(f"  - 훈련 양성 비율: {train_positive_ratio:.4f}")
        logger.info(f"  - 검증 양성 비율: {val_positive_ratio:.4f}")
        
        yield train_fold_df, val_fold_df, fold_info


def main():
    """
    테스트용 메인 함수
    """
    # 설정 로드
    config = load_config()
    
    # 데이터 로드 (처음 10,000행만 사용, 설정 파일의 data.file_path 사용)
    data_path = config['data']['file_path']
    df = pd.read_csv(data_path, nrows=10000)
    
    logger.info(f"테스트 데이터 로드: {len(df):,} 행")
    
    # 1. 테스트 세트 분리
    train_val_df, test_df, test_ids = split_test_set(df, config)
    
    # 2. 분할 검증
    if not validate_splits(train_val_df, test_df, config):
        logger.error("분할 검증 실패!")
        return
    
    # 3. 분할 정보 로깅
    splits_info = log_splits_info(train_val_df, test_df, config)
    
    # 4. 교차 검증 폴드 생성 테스트
    fold_count = 0
    for train_fold_df, val_fold_df, fold_info in get_cv_splits(train_val_df, config):
        fold_count += 1
        logger.info(f"폴드 {fold_count} 생성 완료")
        
        # 첫 번째 폴드만 상세 로깅
        if fold_count == 1 and config['logging']['log_fold_details']:
            logger.info("=== 첫 번째 폴드 상세 정보 ===")
            for key, value in fold_info.items():
                logger.info(f"  {key}: {value}")
    
    logger.info(f"총 {fold_count}개의 폴드가 생성되었습니다.")


if __name__ == "__main__":
    main() 