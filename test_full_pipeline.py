#!/usr/bin/env python3
"""
통합 피처 엔지니어링 + 전처리 파이프라인 end-to-end 테스트 스크립트
"""

import pandas as pd
import yaml
from src.feature_engineering import fit_feature_engineering
from src.preprocessing import fit_preprocessing_pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_full_pipeline():
    print("=== 통합 피처 엔지니어링 + 전처리 파이프라인 테스트 시작 ===")
    # 1. 설정 로드
    print("\n1. 설정 파일 로드...")
    with open('configs/base/common.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ 설정 파일 로드 완료")
    # 2. 데이터 로드
    print("\n2. 데이터 로드...")
    df = pd.read_csv('data/sourcedata/data.csv')
    print(f"✅ 데이터 로드 완료: {df.shape}")
    # 3. 피처 엔지니어링
    print("\n3. 피처 엔지니어링 실행...")
    try:
        df_engineered, feature_info = fit_feature_engineering(df, config)
        print(f"✅ 피처 엔지니어링 완료: {df_engineered.shape}")
    except Exception as e:
        print(f"❌ 피처 엔지니어링 실패: {e}")
        return
    # 4. 전처리 파이프라인 실행
    print("\n4. 전처리 파이프라인 실행...")
    try:
        preprocessor, transformed_df = fit_preprocessing_pipeline(df_engineered, config)
        print(f"✅ 전처리 파이프라인 완료: {transformed_df.shape}")
    except Exception as e:
        print(f"❌ 전처리 파이프라인 실패: {e}")
        return
    # 5. 결측치 플래그 컬럼 확인
    print("\n5. 결측치 플래그 컬럼 확인...")
    flag_cols = [col for col in transformed_df.columns if col.endswith('_missing')]
    print(f"  - 플래그 컬럼 수: {len(flag_cols)}")
    print(f"  - 플래그 컬럼 목록: {flag_cols}")
    for col in flag_cols:
        flag_count = transformed_df[col].sum()
        flag_rate = flag_count / len(transformed_df) * 100
        print(f"  - {col}: {flag_count}개 ({flag_rate:.4f}%)")
    print("\n=== 통합 파이프라인 테스트 완료 ===")

if __name__ == "__main__":
    test_full_pipeline() 