"""
데이터 전처리 모듈

이 모듈은 데이터 유출을 방지하면서 결측치를 처리하는 전처리 파이프라인을 구현합니다.
- ID별 시계열 보간 (ffill, bfill)
- sklearn 기반 결측치 처리
- 범주형 인코딩 (One-Hot Encoding)
- 데이터 유출 방지 원칙 준수
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import warnings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_timeseries_imputation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    ID별로 시계열 보간을 적용합니다.
    
    Args:
        df: 입력 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        시계열 보간이 적용된 데이터프레임
    """
    id_column = config['time_series']['id_column']
    date_column = config['time_series']['date_column']
    
    # 시계열 보간 대상 컬럼
    ts_columns = config['preprocessing']['time_series_imputation']['columns']
    fallback_strategy = config['preprocessing']['time_series_imputation']['fallback_strategy']
    
    logger.info(f"시계열 보간 적용: {ts_columns}")
    logger.info(f"Fallback 전략: {fallback_strategy}")
    
    # 데이터 복사
    df_imputed = df.copy()
    
    # ID별로 그룹화하여 시계열 보간 적용
    for col in ts_columns:
        if col in df.columns:
            # ID별로 정렬 후 ffill, bfill 적용
            df_imputed[col] = df_imputed.groupby(id_column)[col].ffill().bfill()
            
            # 여전히 결측치가 있다면 fallback 전략 적용
            if df_imputed[col].isnull().any():
                if fallback_strategy == "mean":
                    fallback_val = df_imputed[col].mean()
                elif fallback_strategy == "median":
                    fallback_val = df_imputed[col].median()
                elif fallback_strategy == "constant":
                    fallback_val = 0  # 설정에서 constant_value를 추가할 수 있음
                else:
                    fallback_val = df_imputed[col].mean()  # 기본값
                
                df_imputed[col].fillna(fallback_val, inplace=True)
                logger.info(f"  {col}: {fallback_strategy} 값으로 추가 보간 ({fallback_val:.4f})")
    
    return df_imputed


def get_numerical_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    수치형 컬럼 목록을 반환합니다.
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        수치형 컬럼 목록
    """
    # 설정에서 명시된 수치형 컬럼
    numerical_cols = config['preprocessing']['numerical_imputation']['columns']
    
    # 실제 존재하는 컬럼만 필터링
    available_numerical = [col for col in numerical_cols if col in df.columns]
    
    logger.info(f"수치형 컬럼: {available_numerical}")
    return available_numerical


def get_categorical_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    범주형 컬럼 목록을 반환합니다.
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        범주형 컬럼 목록
    """
    # 설정에서 명시된 범주형 컬럼
    categorical_cols = config['preprocessing']['categorical_imputation']['columns']
    
    # 실제 존재하는 컬럼만 필터링
    available_categorical = [col for col in categorical_cols if col in df.columns]
    
    logger.info(f"범주형 컬럼: {available_categorical}")
    return available_categorical


def get_passthrough_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    전처리 파이프라인에서 그대로 통과시킬 컬럼 목록을 반환합니다.
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        통과시킬 컬럼 목록
    """
    # 기본적으로 제외할 컬럼들
    exclude_columns = [
        config['time_series']['id_column'],
        config['time_series']['date_column'],
        config['time_series']['year_column']
    ]
    
    # 타겟 컬럼들은 제외하지 않음 (피처 엔지니어링 후 분리하기 위해)
    
    # 이미 처리되는 컬럼들도 제외
    numerical_cols = get_numerical_columns(df, config)
    categorical_cols = get_categorical_columns(df, config)
    exclude_columns.extend(numerical_cols)
    exclude_columns.extend(categorical_cols)
    
    # 통과시킬 컬럼들 (존재하는 컬럼만)
    passthrough_cols = [col for col in df.columns if col not in exclude_columns]
    
    logger.info(f"통과시킬 컬럼: {passthrough_cols}")
    return passthrough_cols


def create_preprocessing_pipeline(config: Dict[str, Any]) -> ColumnTransformer:
    """
    전처리 파이프라인을 생성합니다.
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        전처리 파이프라인
    """
    # 수치형 결측치 처리 + 스케일링
    numerical_strategy = config['preprocessing']['numerical_imputation']['strategy']
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=numerical_strategy)),
        ('scaler', StandardScaler())
    ])
    
    # 범주형 결측치 처리 + 인코딩
    categorical_strategy = config['preprocessing']['categorical_imputation']['strategy']
    categorical_encoding = config['preprocessing']['categorical_encoding']['method']
    
    if categorical_encoding == 'onehot':
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
    elif categorical_encoding == 'ordinal':
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    elif categorical_encoding == 'none':
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing'))
        ])
    else:
        raise ValueError(f"지원하지 않는 범주형 인코딩 방법: {categorical_encoding}. 'onehot', 'ordinal', 'none' 중 선택하세요.")
    
    # 컬럼 변환기 생성 (placeholder로 시작)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, 'placeholder_numerical'),
            ('cat', categorical_transformer, 'placeholder_categorical')
        ],
        remainder='passthrough'  # 처리되지 않은 컬럼은 그대로 유지
    )
    
    logger.info(f"전처리 파이프라인 생성 완료")
    logger.info(f"  - 수치형 전략: {numerical_strategy}")
    logger.info(f"  - 범주형 전략: {categorical_strategy}")
    logger.info(f"  - 범주형 인코딩: {categorical_encoding}")
    
    return preprocessor


def fit_preprocessing_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """
    전처리 파이프라인을 학습하고 적용합니다.
    
    Args:
        df: 훈련 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        학습된 파이프라인과 전처리된 데이터프레임
    """
    logger.info("전처리 파이프라인 학습 시작")
    
    # 1단계: 시계열 보간 적용
    df_ts_imputed = apply_timeseries_imputation(df, config)
    
    # 2단계: 컬럼 분류
    numerical_cols = get_numerical_columns(df_ts_imputed, config)
    categorical_cols = get_categorical_columns(df_ts_imputed, config)
    passthrough_cols = get_passthrough_columns(df_ts_imputed, config)
    
    # 3단계: 전처리 파이프라인 생성 및 학습
    preprocessor = create_preprocessing_pipeline(config)
    
    # 실제 컬럼명으로 업데이트
    preprocessor.transformers[0] = ('num', preprocessor.transformers[0][1], numerical_cols)
    preprocessor.transformers[1] = ('cat', preprocessor.transformers[1][1], categorical_cols)
    
    # 파이프라인 학습 및 적용
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preprocessor.fit(df_ts_imputed)
        df_processed = pd.DataFrame(
            preprocessor.transform(df_ts_imputed),
            columns=preprocessor.get_feature_names_out(),
            index=df_ts_imputed.index
        )
    
    logger.info(f"전처리 완료: {df_processed.shape}")
    logger.info(f"결측치 수: {df_processed.isnull().sum().sum()}")
    logger.info(f"피처 이름: {list(df_processed.columns)}")
    
    return preprocessor, df_processed


def transform_data(df: pd.DataFrame, preprocessor: ColumnTransformer, config: Dict[str, Any]) -> pd.DataFrame:
    """
    학습된 전처리 파이프라인을 사용하여 데이터를 변환합니다.
    
    Args:
        df: 변환할 데이터프레임
        preprocessor: 학습된 전처리 파이프라인
        config: 설정 딕셔너리
        
    Returns:
        변환된 데이터프레임
    """
    logger.info("전처리 파이프라인 적용")
    
    # 1단계: 시계열 보간 적용
    df_ts_imputed = apply_timeseries_imputation(df, config)
    
    # 2단계: 파이프라인 적용
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_processed = pd.DataFrame(
            preprocessor.transform(df_ts_imputed),
            columns=preprocessor.get_feature_names_out(),
            index=df_ts_imputed.index
        )

    # XGBoost 호환을 위해 변환 가능한 컬럼만 float32로 변환
    for col in df_processed.columns:
        try:
            df_processed[col] = df_processed[col].astype('float32')
        except Exception:
            pass  # 변환 불가 컬럼은 그대로 둠

    logger.info(f"변환 완료: {df_processed.shape}")
    logger.info(f"결측치 수: {df_processed.isnull().sum().sum()}")
    
    return df_processed


def validate_preprocessing(df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """
    전처리 결과의 유효성을 검증합니다.
    
    Args:
        df: 전처리된 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        검증 통과 여부
    """
    # 1. 결측치 확인
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"전처리 후에도 {missing_count}개의 결측치가 남아있습니다.")
        return False
    
    # 2. 무한값 확인
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        logger.warning(f"전처리 후 {inf_count}개의 무한값이 발견되었습니다.")
        return False
    
    # 3. 데이터 타입 확인
    logger.info("전처리 검증 통과 ✓")
    return True


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
    
    # 전처리 파이프라인 학습 및 적용
    preprocessor, df_processed = fit_preprocessing_pipeline(df, config)
    
    # 검증
    if validate_preprocessing(df_processed, config):
        logger.info("전처리 테스트 성공!")
    else:
        logger.error("전처리 테스트 실패!")


if __name__ == "__main__":
    main() 