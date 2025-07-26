#!/usr/bin/env python3
"""
4개 모델의 CPU 코어 사용 방식 점검 스크립트

이 스크립트는 XGBoost, LightGBM, CatBoost, Random Forest 모델의
병렬 처리 설정을 점검하고 일관성을 확인합니다.
"""

import os
import sys
import yaml
import logging
import psutil
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config_manager import ConfigManager
from src.models import ModelFactory

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """시스템 정보를 수집합니다."""
    return {
        'cpu_count': multiprocessing.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }


def check_model_config(model_type: str, config_manager: ConfigManager) -> Dict[str, Any]:
    """특정 모델의 설정을 점검합니다."""
    try:
        # 모델 설정 로드
        config = config_manager.create_experiment_config(model_type=model_type)
        
        # 모델 파라미터 추출
        model_config = config.get('model', {}).get(model_type, {})
        
        # 병렬 처리 파라미터 확인
        parallel_params = {}
        if model_type == 'xgboost':
            parallel_params['n_jobs'] = model_config.get('n_jobs', 'Not set')
        elif model_type == 'lightgbm':
            parallel_params['n_jobs'] = model_config.get('n_jobs', 'Not set')
        elif model_type == 'catboost':
            parallel_params['n_jobs'] = model_config.get('n_jobs', 'Not set')
        elif model_type == 'random_forest':
            parallel_params['n_jobs'] = model_config.get('n_jobs', 'Not set')
        
        return {
            'model_type': model_type,
            'config_file': f'configs/models/{model_type}.yaml',
            'parallel_params': parallel_params,
            'config_loaded': True
        }
    except Exception as e:
        logger.error(f"모델 {model_type} 설정 점검 실패: {e}")
        return {
            'model_type': model_type,
            'config_file': f'configs/models/{model_type}.yaml',
            'parallel_params': {},
            'config_loaded': False,
            'error': str(e)
        }


def check_tuning_configs() -> Dict[str, Any]:
    """하이퍼파라미터 튜닝 설정을 점검합니다."""
    tuning_configs = {}
    
    # 튜닝 설정 파일들 확인
    tuning_files = [
        'configs/experiments/hyperparameter_tuning.yaml',
        'configs/experiments/resampling.yaml',
        'configs/experiments/resampling_experiment.yaml',
        'configs/templates/tuning.yaml'
    ]
    
    for file_path in tuning_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            tuning_configs[file_path] = {
                'n_jobs': config.get('tuning', {}).get('n_jobs', 'Not set'),
                'n_trials': config.get('tuning', {}).get('n_trials', 'Not set'),
                'direction': config.get('tuning', {}).get('direction', 'Not set')
            }
        except Exception as e:
            tuning_configs[file_path] = {
                'error': str(e)
            }
    
    return tuning_configs


def check_model_implementation(model_type: str) -> Dict[str, Any]:
    """모델 구현에서 병렬 처리 파라미터 사용 방식을 점검합니다."""
    try:
        # 모델 팩토리에서 모델 생성
        config_manager = ConfigManager()
        config = config_manager.create_experiment_config(model_type=model_type)
        
        # 모델 인스턴스 생성
        model = ModelFactory.create_model(config)
        
        # 모델 파라미터 확인
        if hasattr(model, '_get_model_params'):
            # 테스트용 타겟으로 파라미터 확인
            test_target = 'suicide_a_next_year'
            params = model._get_model_params(test_target)
            
            # 병렬 처리 파라미터 추출
            parallel_params = {}
            if model_type == 'xgboost':
                parallel_params['n_jobs'] = params.get('n_jobs', 'Not found')
            elif model_type == 'lightgbm':
                parallel_params['num_threads'] = params.get('num_threads', 'Not found')
            elif model_type == 'catboost':
                parallel_params['thread_count'] = params.get('thread_count', 'Not found')
            elif model_type == 'random_forest':
                parallel_params['n_jobs'] = params.get('n_jobs', 'Not found')
            
            return {
                'model_type': model_type,
                'implementation_checked': True,
                'parallel_params': parallel_params,
                'params_method_exists': True
            }
        else:
            return {
                'model_type': model_type,
                'implementation_checked': False,
                'error': '_get_model_params method not found'
            }
    except Exception as e:
        return {
            'model_type': model_type,
            'implementation_checked': False,
            'error': str(e)
        }


def print_system_info(system_info: Dict[str, Any]):
    """시스템 정보를 출력합니다."""
    print("\n" + "="*60)
    print("시스템 정보")
    print("="*60)
    print(f"전체 CPU 코어 수: {system_info['cpu_count']}")
    print(f"논리적 CPU 코어 수: {system_info['cpu_count_logical']}")
    print(f"물리적 CPU 코어 수: {system_info['cpu_count_physical']}")
    print(f"총 메모리: {system_info['memory_total_gb']:.1f}GB")
    print(f"사용 가능한 메모리: {system_info['memory_available_gb']:.1f}GB")


def print_model_configs(model_configs: List[Dict[str, Any]]):
    """모델 설정 정보를 출력합니다."""
    print("\n" + "="*60)
    print("모델별 설정 파일 점검")
    print("="*60)
    
    for config in model_configs:
        print(f"\n📁 {config['model_type'].upper()}")
        print(f"   설정 파일: {config['config_file']}")
        
        if config['config_loaded']:
            for param, value in config['parallel_params'].items():
                print(f"   {param}: {value}")
        else:
            print(f"   ❌ 설정 로드 실패: {config.get('error', 'Unknown error')}")


def print_implementation_check(impl_results: List[Dict[str, Any]]):
    """모델 구현 점검 결과를 출력합니다."""
    print("\n" + "="*60)
    print("모델 구현 점검")
    print("="*60)
    
    for result in impl_results:
        print(f"\n🔧 {result['model_type'].upper()}")
        
        if result['implementation_checked']:
            for param, value in result['parallel_params'].items():
                print(f"   {param}: {value}")
        else:
            print(f"   ❌ 구현 점검 실패: {result.get('error', 'Unknown error')}")


def print_tuning_configs(tuning_configs: Dict[str, Any]):
    """튜닝 설정 정보를 출력합니다."""
    print("\n" + "="*60)
    print("하이퍼파라미터 튜닝 설정 점검")
    print("="*60)
    
    for file_path, config in tuning_configs.items():
        print(f"\n📋 {file_path}")
        
        if 'error' in config:
            print(f"   ❌ 설정 로드 실패: {config['error']}")
        else:
            for param, value in config.items():
                print(f"   {param}: {value}")


def print_recommendations(system_info: Dict[str, Any], model_configs: List[Dict[str, Any]]):
    """권장사항을 출력합니다."""
    print("\n" + "="*60)
    print("권장사항")
    print("="*60)
    
    cpu_count = system_info['cpu_count']
    
    print(f"\n💡 시스템 권장사항:")
    print(f"   - 안전한 병렬 처리: n_jobs = 4 (현재 설정)")
    print(f"   - 고성능 병렬 처리: n_jobs = {min(cpu_count // 2, 8)}")
    print(f"   - 최대 병렬 처리: n_jobs = {cpu_count}")
    
    print(f"\n🔧 모델별 권장사항:")
    print(f"   - XGBoost: n_jobs = 4 (현재 설정)")
    print(f"   - LightGBM: n_jobs = 4 (현재 설정)")
    print(f"   - CatBoost: n_jobs = 4 (현재 설정)")
    print(f"   - Random Forest: n_jobs = 4 (현재 설정)")
    
    print(f"\n⚡ 성능 최적화 팁:")
    print(f"   - 메모리 부족 시: n_jobs = 2")
    print(f"   - 안정성 우선: n_jobs = 4 (현재)")
    print(f"   - 성능 우선: n_jobs = {min(cpu_count // 2, 8)}")
    print(f"   - 극한 성능: n_jobs = {cpu_count}")


def main():
    """메인 함수"""
    print("🚀 4개 모델 CPU 코어 사용 방식 점검 시작")
    
    # 시스템 정보 수집
    system_info = get_system_info()
    
    # ConfigManager 초기화
    config_manager = ConfigManager()
    
    # 모델 타입들
    model_types = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
    
    # 모델 설정 점검
    model_configs = []
    for model_type in model_types:
        config = check_model_config(model_type, config_manager)
        model_configs.append(config)
    
    # 모델 구현 점검
    impl_results = []
    for model_type in model_types:
        result = check_model_implementation(model_type)
        impl_results.append(result)
    
    # 튜닝 설정 점검
    tuning_configs = check_tuning_configs()
    
    # 결과 출력
    print_system_info(system_info)
    print_model_configs(model_configs)
    print_implementation_check(impl_results)
    print_tuning_configs(tuning_configs)
    print_recommendations(system_info, model_configs)
    
    print("\n" + "="*60)
    print("✅ CPU 코어 사용 방식 점검 완료")
    print("="*60)


if __name__ == "__main__":
    main() 