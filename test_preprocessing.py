#!/usr/bin/env python3
"""
전처리 파이프라인 테스트 스크립트
결측치 플래그 생성 기능을 포함한 전처리 파이프라인을 테스트합니다.
"""

import pandas as pd
import yaml
from src.preprocessing import fit_preprocessing_pipeline
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_preprocessing_pipeline():
    """전처리 파이프라인을 테스트합니다."""
    
    print("=== 전처리 파이프라인 테스트 시작 ===")
    
    # 1. 설정 로드
    print("\n1. 설정 파일 로드...")
    with open('configs/base/common.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ 설정 파일 로드 완료")
    
    # 2. 데이터 로드
    print("\n2. 데이터 로드...")
    df = pd.read_csv('data/sourcedata/data.csv')
    print(f"✅ 데이터 로드 완료: {df.shape}")
    
    # 3. 원본 데이터 결측치 확인
    print("\n3. 원본 데이터 결측치 확인...")
    target_cols = ['sleep_score', 'depress_score', 'anxiety_score']
    for col in target_cols:
        missing_count = df[col].isnull().sum()
        missing_rate = missing_count / len(df) * 100
        print(f"  - {col}: {missing_count}개 ({missing_rate:.4f}%)")
    
    # 4. 전처리 파이프라인 실행
    print("\n4. 전처리 파이프라인 실행...")
    try:
        preprocessor, transformed_df = fit_preprocessing_pipeline(df, config)
        print("✅ 전처리 파이프라인 실행 완료")
    except Exception as e:
        print(f"❌ 전처리 파이프라인 실행 실패: {e}")
        return
    
    # 5. 전처리 후 데이터 확인
    print("\n5. 전처리 후 데이터 확인...")
    print(f"  - 데이터 크기: {transformed_df.shape}")
    
    # 6. 전처리 후 결측치 확인
    print("\n6. 전처리 후 결측치 확인...")
    for col in target_cols:
        if col in transformed_df.columns:
            missing_count = transformed_df[col].isnull().sum()
            print(f"  - {col}: {missing_count}개")
        else:
            print(f"  - {col}: 컬럼이 존재하지 않음")
    
    # 7. 생성된 플래그 컬럼 확인
    print("\n7. 생성된 플래그 컬럼 확인...")
    flag_cols = [col for col in transformed_df.columns if col.endswith('_missing')]
    print(f"  - 플래그 컬럼 수: {len(flag_cols)}")
    print(f"  - 플래그 컬럼 목록: {flag_cols}")
    
    # 8. 플래그 컬럼 통계
    print("\n8. 플래그 컬럼 통계...")
    for col in flag_cols:
        flag_count = transformed_df[col].sum()
        flag_rate = flag_count / len(transformed_df) * 100
        print(f"  - {col}: {flag_count}개 ({flag_rate:.4f}%)")
    
    # 9. 플래그 컬럼과 자살 관련성 확인
    print("\n9. 플래그 컬럼과 자살 관련성 확인...")
    suicide_cols = ['suicide_t', 'suicide_a']
    for flag_col in flag_cols:
        print(f"\n  {flag_col} 분석:")
        for suicide_col in suicide_cols:
            if suicide_col in transformed_df.columns:
                # 플래그가 1인 경우의 자살 양성 비율
                flag_positive = transformed_df[transformed_df[flag_col] == 1]
                if len(flag_positive) > 0:
                    suicide_positive = flag_positive[suicide_col].sum()
                    suicide_rate = suicide_positive / len(flag_positive) * 100
                    print(f"    - {suicide_col} 양성: {suicide_positive}개 ({suicide_rate:.2f}%)")
                
                # 전체 자살 양성 비율
                total_suicide_positive = transformed_df[suicide_col].sum()
                total_suicide_rate = total_suicide_positive / len(transformed_df) * 100
                print(f"    - 전체 {suicide_col} 양성 비율: {total_suicide_rate:.2f}%")
    
    # 10. 컬럼 목록 확인
    print("\n10. 최종 컬럼 목록 (처음 20개)...")
    print(f"  - 총 컬럼 수: {len(transformed_df.columns)}")
    print(f"  - 컬럼 목록: {list(transformed_df.columns[:20])}")
    if len(transformed_df.columns) > 20:
        print(f"  - ... (총 {len(transformed_df.columns)}개 컬럼)")
    
    print("\n=== 전처리 파이프라인 테스트 완료 ===")

if __name__ == "__main__":
    test_preprocessing_pipeline() 