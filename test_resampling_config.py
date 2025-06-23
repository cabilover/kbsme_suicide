#!/usr/bin/env python3
"""
리샘플링 설정 테스트 스크립트
모델별 리샘플링 설정이 올바르게 로드되는지 확인
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config_manager import ConfigManager

def test_resampling_configs():
    """모든 모델의 리샘플링 설정을 테스트"""
    
    models = ['xgboost', 'catboost', 'lightgbm', 'random_forest']
    config_manager = ConfigManager()
    
    print("=== 리샘플링 설정 테스트 ===\n")
    
    for model in models:
        print(f"--- {model.upper()} 모델 설정 테스트 ---")
        
        try:
            # 리샘플링 실험 설정 생성
            config = config_manager.create_experiment_config(model, 'resampling')
            
            # 불균형 데이터 처리 설정 확인
            if 'imbalanced_data' in config:
                print(f"✓ {model}: imbalanced_data 설정 존재")
                
                resampling_config = config['imbalanced_data'].get('resampling', {})
                if resampling_config:
                    print(f"  - resampling.enabled: {resampling_config.get('enabled', False)}")
                    print(f"  - resampling.method: {resampling_config.get('method', 'N/A')}")
                else:
                    print(f"  - resampling 설정 없음")
            else:
                print(f"✗ {model}: imbalanced_data 설정 없음")
            
            # 모델별 리샘플링 설정 확인
            model_specific = config.get('model_specific', {}).get(model, {})
            if model_specific:
                print(f"✓ {model}: model_specific 설정 존재")
                for key, value in model_specific.items():
                    print(f"  - {key}: {value}")
            else:
                print(f"✗ {model}: model_specific 설정 없음")
            
            # 하이퍼파라미터 검색 범위 확인
            params_key = f"{model}_params"
            if params_key in config:
                print(f"✓ {model}: {params_key} 설정 존재")
                param_count = len(config[params_key])
                print(f"  - 검색 파라미터 수: {param_count}")
            else:
                print(f"✗ {model}: {params_key} 설정 없음")
            
            print()
            
        except Exception as e:
            print(f"✗ {model}: 설정 로드 실패 - {str(e)}")
            print()

if __name__ == "__main__":
    test_resampling_configs() 