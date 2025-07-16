"""
데이터 분석 모듈

이 모듈은 원본 데이터에 대한 종합적인 탐색적 데이터 분석(EDA)을 수행합니다.
- 결측치 분석 및 시각화
- 이상치 분석 (IQR 기반)
- 시계열 길이 분석 (개인별 연도 수)
- 타겟 변수 분포 분석 (연속형/이진형)
- 데이터 타입 분석 및 변환
- 타겟 변수 생성 (다음 해 예측용)

참고: 피처 엔지니어링은 별도의 feature_engineering.py 모듈에서 담당합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mlflow
from datetime import datetime
import warnings
import yaml
import logging
from src.utils import setup_logging

# Configure warnings for EDA - suppress common plotting and pandas warnings
# while keeping important ones visible
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', message='.*SettingWithCopyWarning.*')
warnings.filterwarnings('ignore', message='.*DataFrame.*is highly fragmented.*')

# Keep important warnings visible
warnings.filterwarnings('always', category=DeprecationWarning)
warnings.filterwarnings('always', message='.*convergence.*')
warnings.filterwarnings('always', message='.*overflow.*')

# Set up paths
DATA_DIR = Path(__file__).parent.parent / "data" / "sourcedata"
ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "sourcedata_analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"
REPORTS_DIR = ANALYSIS_DIR / "reports"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Create directories if they don't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 로깅 설정
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

def load_config():
    """설정 파일을 로드합니다."""
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
        """)
        raise SystemExit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
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
    
    return config

def get_target_variables(config):
    """설정에서 타겟 변수명들을 가져옵니다."""
    target_vars = config.get('features', {}).get('target_variables', {})
    
    # 기본값 설정 (설정이 없을 경우)
    default_score_targets = ['anxiety_score', 'depress_score', 'sleep_score', 'comp']
    default_binary_targets = ['suicide_t', 'suicide_a']
    
    original_targets = target_vars.get('original_targets', {})
    score_targets = original_targets.get('score_targets', default_score_targets)
    binary_targets = original_targets.get('binary_targets', default_binary_targets)
    
    return score_targets, binary_targets

def load_data():
    """Load the raw data and perform initial inspection."""
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "data.csv")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    print("\n=== Missing Values Analysis ===")
    missing_stats = df.isnull().sum()
    missing_percentage = (missing_stats / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_stats,
        'Percentage': missing_percentage
    }).sort_values('Percentage', ascending=False)
    
    print("\nMissing values summary:")
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Save missing values analysis to CSV
    missing_df.to_csv(REPORTS_DIR / "missing_values_analysis.txt")
    
    # Create missing values visualization
    plt.figure(figsize=(12, 8))
    missing_percentage_plot = missing_percentage[missing_percentage > 0].sort_values(ascending=True)
    plt.barh(range(len(missing_percentage_plot)), missing_percentage_plot.values)
    plt.yticks(range(len(missing_percentage_plot)), missing_percentage_plot.index)
    plt.xlabel('Missing Percentage (%)')
    plt.title('Missing Values by Column')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'missing_values_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return missing_df

def analyze_outliers(df, config=None):
    """Analyze outliers in score variables."""
    print("\n=== Outlier Analysis ===")
    
    # 설정에서 점수 컬럼 가져오기
    score_targets, _ = get_target_variables(config or load_config())
    score_columns = score_targets
    
    outlier_report = {}
    
    for col in score_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            outlier_report[col] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound,
                'Outlier Count': outlier_count,
                'Outlier Percentage': outlier_percentage
            }
            
            print(f"\n{col}:")
            print(f"  Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
            print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Save outlier analysis
    outlier_df = pd.DataFrame(outlier_report).T
    outlier_df.to_csv(REPORTS_DIR / "outlier_analysis.txt")
    
    return outlier_report

def analyze_time_series_length(df):
    """Analyze the length of time series for each individual."""
    print("\n=== Time Series Length Analysis ===")
    ts_length = df.groupby('id')['yov'].nunique()
    
    print("\nTime series length statistics:")
    print(ts_length.describe())
    
    # Plot distribution of time series lengths
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.histplot(ts_length, bins=30)
    plt.title('Distribution of Time Series Length per Individual')
    plt.xlabel('Number of Years')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(y=ts_length)
    plt.title('Box Plot of Time Series Length')
    plt.ylabel('Number of Years')
    
    plt.subplot(2, 2, 3)
    ts_length.value_counts().sort_index().plot(kind='bar')
    plt.title('Frequency of Time Series Lengths')
    plt.xlabel('Number of Years')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    cumulative_dist = (ts_length.value_counts().sort_index().cumsum() / len(ts_length)) * 100
    cumulative_dist.plot(kind='line', marker='o')
    plt.title('Cumulative Distribution of Time Series Lengths')
    plt.xlabel('Number of Years')
    plt.ylabel('Cumulative Percentage (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'time_series_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save time series length statistics
    ts_stats = pd.DataFrame({
        'statistic': ts_length.describe().index,
        'value': ts_length.describe().values
    })
    ts_stats.to_csv(REPORTS_DIR / "time_series_length_stats.txt", index=False)
    
    return ts_length

def analyze_target_distribution(df, config=None):
    """Analyze the distribution of target variables."""
    print("\n=== Target Variable Distribution Analysis ===")
    
    # 설정에서 타겟 변수명 가져오기
    score_targets, binary_targets = get_target_variables(config or load_config())
    
    # Analyze continuous targets
    continuous_targets = score_targets
    
    for target in continuous_targets:
        if target not in df.columns:
            logger.warning(f"타겟 변수 {target}가 데이터에 없습니다. 건너뜁니다.")
            continue
            
        print(f"\n{target} statistics:")
        print(df[target].describe())
        
        plt.figure(figsize=(15, 10))
        
        # Histogram
        plt.subplot(2, 3, 1)
        sns.histplot(df[target], bins=50)
        plt.title(f'Distribution of {target}')
        plt.xlabel(target)
        plt.ylabel('Count')
        
        # Box plot
        plt.subplot(2, 3, 2)
        sns.boxplot(y=df[target])
        plt.title(f'Box Plot of {target}')
        plt.ylabel(target)
        
        # Q-Q plot
        plt.subplot(2, 3, 3)
        from scipy import stats
        stats.probplot(df[target].dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {target}')
        
        # Distribution by year
        plt.subplot(2, 3, 4)
        yearly_stats = df.groupby('yov')[target].mean()
        yearly_stats.plot(kind='line', marker='o')
        plt.title(f'Mean {target} by Year')
        plt.xlabel('Year')
        plt.ylabel(f'Mean {target}')
        plt.xticks(rotation=45)
        
        # Distribution by age group
        plt.subplot(2, 3, 5)
        df['age_group'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '80+'])
        age_stats = df.groupby('age_group')[target].mean()
        age_stats.plot(kind='bar')
        plt.title(f'Mean {target} by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel(f'Mean {target}')
        plt.xticks(rotation=45)
        
        # Distribution by sex
        plt.subplot(2, 3, 6)
        sex_stats = df.groupby('sex')[target].mean()
        sex_stats.plot(kind='bar')
        plt.title(f'Mean {target} by Sex')
        plt.xlabel('Sex')
        plt.ylabel(f'Mean {target}')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'{target}_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Analyze binary targets
    for target in binary_targets:
        if target not in df.columns:
            logger.warning(f"타겟 변수 {target}가 데이터에 없습니다. 건너뜁니다.")
            continue
            
        print(f"\n{target} distribution:")
        print(df[target].value_counts(normalize=True))
        
        plt.figure(figsize=(15, 10))
        
        # Overall distribution
        plt.subplot(2, 3, 1)
        sns.countplot(data=df, x=target)
        plt.title(f'Distribution of {target}')
        
        # Distribution by year
        plt.subplot(2, 3, 2)
        yearly_dist = df.groupby('yov')[target].mean()
        yearly_dist.plot(kind='line', marker='o')
        plt.title(f'{target} Rate by Year')
        plt.xlabel('Year')
        plt.ylabel(f'{target} Rate')
        plt.xticks(rotation=45)
        
        # Distribution by age group
        plt.subplot(2, 3, 3)
        age_dist = df.groupby('age_group')[target].mean()
        age_dist.plot(kind='bar')
        plt.title(f'{target} Rate by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel(f'{target} Rate')
        plt.xticks(rotation=45)
        
        # Distribution by sex
        plt.subplot(2, 3, 4)
        sex_dist = df.groupby('sex')[target].mean()
        sex_dist.plot(kind='bar')
        plt.title(f'{target} Rate by Sex')
        plt.xlabel('Sex')
        plt.ylabel(f'{target} Rate')
        plt.xticks(rotation=0)
        
        # Distribution by psychiatric category
        plt.subplot(2, 3, 5)
        if 'psychia_cate' in df.columns:
            psych_dist = df.groupby('psychia_cate')[target].mean()
            psych_dist.plot(kind='bar')
            plt.title(f'{target} Rate by Psychiatric Category')
            plt.xlabel('Psychiatric Category')
            plt.ylabel(f'{target} Rate')
            plt.xticks(rotation=45)
        
        # Correlation with continuous scores
        plt.subplot(2, 3, 6)
        correlations = []
        for score_col in score_targets:
            if score_col in df.columns:
                corr = df[target].corr(df[score_col])
                correlations.append(corr)
            else:
                correlations.append(0.0)
        
        plt.bar(score_targets, correlations)
        plt.title(f'Correlation with {target}')
        plt.xlabel('Score Variables')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'{target}_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_data_types_and_conversion(df):
    """Analyze data types and prepare for conversion."""
    print("\n=== Data Type Analysis ===")
    
    # Current data types
    print("\nCurrent data types:")
    print(df.dtypes)
    
    # Convert dov to datetime
    if 'dov' in df.columns:
        print("\nConverting 'dov' to datetime...")
        df['dov'] = pd.to_datetime(df['dov'], errors='coerce')
        print(f"Date range: {df['dov'].min()} to {df['dov'].max()}")
    
    # Analyze categorical variables
    categorical_cols = ['sex', 'psychia_cate']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col} unique values:")
            print(df[col].value_counts())
    
    # Save data type analysis
    dtype_analysis = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes,
        'unique_count': [df[col].nunique() for col in df.columns],
        'null_count': df.isnull().sum()
    })
    dtype_analysis.to_csv(REPORTS_DIR / "data_type_analysis.txt", index=False)
    
    return df

def create_target_variables(df, config=None):
    """Create target variables for next year prediction."""
    print("\n=== Target Variable Creation ===")
    
    # 설정에서 타겟 변수명 가져오기
    score_targets, binary_targets = get_target_variables(config or load_config())
    target_cols = score_targets + binary_targets
    
    # Sort by id and yov
    df = df.sort_values(['id', 'yov'])
    
    # Create target variables (next year's values)
    for col in target_cols:
        if col in df.columns:
            df[f'{col}_next_year'] = df.groupby('id')[col].shift(-1)
    
    # Analyze target variable creation
    target_analysis = {}
    for col in target_cols:
        if f'{col}_next_year' in df.columns:
            target_analysis[col] = {
                'total_records': len(df),
                'records_with_target': df[f'{col}_next_year'].notna().sum(),
                'missing_targets': df[f'{col}_next_year'].isna().sum(),
                'target_availability_rate': (df[f'{col}_next_year'].notna().sum() / len(df)) * 100
            }
    
    target_df = pd.DataFrame(target_analysis).T
    target_df.to_csv(REPORTS_DIR / "target_variable_analysis.txt")
    
    print("\nTarget variable creation summary:")
    print(target_df)
    
    return df



def main():
    """Main function to run all analysis."""
    # 설정 로드
    config = load_config()
    
    # Start MLflow run
    with mlflow.start_run(run_name="comprehensive_data_analysis"):
        # Load data
        df = load_data()
        
        # Log data shape
        mlflow.log_param("data_shape", df.shape)
        mlflow.log_param("total_individuals", df['id'].nunique())
        mlflow.log_param("year_range", f"{df['yov'].min()}-{df['yov'].max()}")
        
        # Analyze missing values
        missing_df = analyze_missing_values(df)
        mlflow.log_artifact(str(REPORTS_DIR / "missing_values_analysis.txt"))
        mlflow.log_artifact(str(FIGURES_DIR / "missing_values_analysis.png"))
        
        # Analyze outliers
        outlier_report = analyze_outliers(df, config)
        mlflow.log_artifact(str(REPORTS_DIR / "outlier_analysis.txt"))
        
        # Analyze time series length
        ts_length = analyze_time_series_length(df)
        mlflow.log_artifact(str(FIGURES_DIR / "time_series_length_analysis.png"))
        mlflow.log_artifact(str(REPORTS_DIR / "time_series_length_stats.txt"))
        
        # Analyze target distributions
        analyze_target_distribution(df, config)
        
        # Data type analysis and conversion
        df = analyze_data_types_and_conversion(df)
        mlflow.log_artifact(str(REPORTS_DIR / "data_type_analysis.txt"))
        
        # Create target variables
        df = create_target_variables(df, config)
        mlflow.log_artifact(str(REPORTS_DIR / "target_variable_analysis.txt"))
        
        # Save processed data in the correct location
        df.to_csv(PROCESSED_DIR / "processed_data_with_features.csv", index=False)
        mlflow.log_artifact(str(PROCESSED_DIR / "processed_data_with_features.csv"))
        
        # Log all figures
        for fig_file in FIGURES_DIR.glob("*.png"):
            mlflow.log_artifact(str(fig_file))
        
        print("\n=== Analysis Complete ===")
        print(f"All results saved to: {ANALYSIS_DIR}")
        print(f"Figures saved to: {FIGURES_DIR}")
        print(f"Reports saved to: {REPORTS_DIR}")
        print(f"Processed data saved to: {PROCESSED_DIR}")

if __name__ == "__main__":
    main() 