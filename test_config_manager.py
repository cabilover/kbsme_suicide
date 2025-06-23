#!/usr/bin/env python3
"""
ConfigManager 테스트 스크립트
새로운 설정 관리 시스템의 기능을 검증
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config_manager import ConfigManager
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_config_manager():
    """ConfigManager 기능 테스트"""
    print("="*60)
    print("ConfigManager 테스트 시작")
    print("="*60)
    
    # ConfigManager 초기화
    config_manager = ConfigManager()
    
    # 사용 가능한 모델과 실험 목록 출력
    print("\n1. 사용 가능한 모델 목록:")
    models = config_manager.get_available_models()
    for model in models:
        print(f"  - {model}")
    
    print("\n2. 사용 가능한 실험 목록:")
    experiments = config_manager.get_available_experiments()
    for experiment in experiments:
        print(f"  - {experiment}")
    
    # 다양한 설정 조합 테스트
    test_cases = [
        ("xgboost", None, "기본 XGBoost 실험"),
        ("catboost", None, "기본 CatBoost 실험"),
        ("lightgbm", None, "기본 LightGBM 실험"),
        ("random_forest", None, "기본 Random Forest 실험"),
        ("xgboost", "focal_loss", "XGBoost + Focal Loss 실험"),
        ("xgboost", "resampling", "XGBoost + 리샘플링 실험"),
        ("xgboost", "hyperparameter_tuning", "XGBoost + 하이퍼파라미터 튜닝"),
    ]
    
    print("\n3. 설정 생성 테스트:")
    for model_type, experiment_type, description in test_cases:
        print(f"\n{description}:")
        try:
            config = config_manager.create_experiment_config(model_type, experiment_type)
            
            # 설정 유효성 검증
            is_valid = config_manager.validate_config(config)
            print(f"  ✓ 설정 생성 성공 (유효성: {is_valid})")
            
            # 설정 요약 출력
            config_manager.print_config_summary(config)
            
            # 설정 저장 (테스트용)
            output_path = f"test_config_{model_type}_{experiment_type or 'default'}.yaml"
            config_manager.save_config(config, output_path)
            print(f"  ✓ 설정 파일 저장: {output_path}")
            
        except Exception as e:
            print(f"  ✗ 설정 생성 실패: {e}")
    
    print("\n" + "="*60)
    print("ConfigManager 테스트 완료")
    print("="*60)

if __name__ == "__main__":
    test_config_manager() 