#!/usr/bin/env python3
"""
하이퍼파라미터 튜닝 실행 스크립트

Optuna를 활용한 하이퍼파라미터 최적화를 실행합니다.
MLflow와 연동되어 실험 추적이 가능하며, 다양한 튜닝 전략을 지원합니다.
리샘플링 기법 비교 실험 기능도 포함되어 있습니다.
"""

import argparse
import logging
import sys
from pathlib import Path
import mlflow
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.hyperparameter_tuning import HyperparameterTuner
from src.splits import (
    load_config, 
    split_test_set, 
    validate_splits, 
    log_splits_info
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mlflow_experiment(experiment_name: str):
    """
    MLflow 실험을 설정합니다.
    
    Args:
        experiment_name: 실험 이름
        
    Returns:
        experiment_id: 실험 ID
    """
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    logger.info(f"MLflow 실험 설정: {experiment_name} (ID: {experiment_id})")
    return experiment_id


def log_tuning_params(tuning_config: Dict[str, Any], base_config: Dict[str, Any]):
    """
    튜닝 파라미터를 MLflow에 로깅합니다.
    
    Args:
        tuning_config: 튜닝 설정 딕셔너리
        base_config: 기본 설정 딕셔너리
    """
    # 튜닝 파라미터
    tuning_params = {
        "n_trials": tuning_config['tuning']['n_trials'],
        "timeout": tuning_config['tuning'].get('timeout', None),
        "sampler": tuning_config['sampler']['type'],
        "optimization_direction": tuning_config['evaluation']['optimization_direction'],
        "primary_metric": tuning_config['evaluation']['primary_metric']
    }
    
    # XGBoost 파라미터 범위
    xgboost_params = tuning_config['xgboost_params']
    for param, param_config in xgboost_params.items():
        if 'range' in param_config:
            tuning_params[f"tune_{param}_min"] = param_config['range'][0]
            tuning_params[f"tune_{param}_max"] = param_config['range'][1]
        elif 'choices' in param_config:
            tuning_params[f"tune_{param}_choices"] = str(param_config['choices'])
    
    # 기본 설정에서 리샘플링 파라미터 추출
    resampling_config = base_config.get('resampling', {})
    resampling_params = {
        "resampling_enabled": resampling_config.get('enabled', False),
        "resampling_method": resampling_config.get('method', 'none'),
        "resampling_random_state": resampling_config.get('random_state', 42)
    }
    
    # 모든 파라미터 로깅
    all_params = {**tuning_params, **resampling_params}
    mlflow.log_params(all_params)
    logger.info("튜닝 파라미터 로깅 완료")


def run_resampling_tuning_comparison(tuning_config_path: str, base_config_path: str,
                                   data_path: str, resampling_methods: List[str],
                                   nrows: int = None) -> Dict[str, Any]:
    """
    다양한 리샘플링 기법에 대해 하이퍼파라미터 튜닝을 비교하는 실험을 실행합니다.
    
    Args:
        tuning_config_path: 튜닝 설정 파일 경로
        base_config_path: 기본 설정 파일 경로
        data_path: 데이터 파일 경로
        resampling_methods: 테스트할 리샘플링 기법 리스트
        nrows: 사용할 데이터 행 수 (None이면 전체 사용)
        
    Returns:
        비교 실험 결과
    """
    logger.info("=== 리샘플링 하이퍼파라미터 튜닝 비교 실험 시작 ===")
    
    # 기본 설정 로드
    base_config = load_config(base_config_path)
    
    # 데이터 로드
    logger.info(f"데이터 로드 중: {data_path}")
    if nrows:
        df = pd.read_csv(data_path, nrows=nrows)
        logger.info(f"테스트용 데이터 로드: {len(df):,} 행")
    else:
        df = pd.read_csv(data_path)
        logger.info(f"전체 데이터 로드: {len(df):,} 행")
    
    # 테스트 세트 분리
    logger.info("=== 테스트 세트 분리 ===")
    train_val_df, test_df, test_ids = split_test_set(df, base_config)
    
    # 분할 검증
    if not validate_splits(train_val_df, test_df, base_config):
        logger.error("분할 검증 실패!")
        return {}
    
    comparison_results = {
        'methods': {},
        'best_method': None,
        'best_score': -np.inf,
        'summary': {}
    }
    
    for method in resampling_methods:
        logger.info(f"\n--- {method.upper()} 하이퍼파라미터 튜닝 시작 ---")
        
        # 설정 복사 및 리샘플링 설정 업데이트
        method_base_config = base_config.copy()
        method_base_config['resampling'] = {
            'enabled': method != 'none',
            'method': method if method != 'none' else 'smote',  # none이 아닌 경우에만 설정
            'random_state': 42
        }
        
        # 리샘플링 기법별 파라미터 설정
        if method == 'smote':
            method_base_config['resampling']['smote_k_neighbors'] = 5
        elif method == 'borderline_smote':
            method_base_config['resampling']['borderline_smote_k_neighbors'] = 5
        elif method == 'adasyn':
            method_base_config['resampling']['adasyn_k_neighbors'] = 5
        elif method == 'under_sampling':
            method_base_config['resampling']['under_sampling_strategy'] = 'random'
        elif method == 'hybrid':
            method_base_config['resampling']['hybrid_strategy'] = 'smote_tomek'
        
        # MLflow 중첩 실행
        with mlflow.start_run(run_name=f"resampling_tuning_{method}", nested=True):
            # 튜닝 설정 로드
            with open(tuning_config_path, 'r', encoding='utf-8') as f:
                method_tuning_config = yaml.safe_load(f)
            
            # 실험 파라미터 로깅
            log_tuning_params(method_tuning_config, method_base_config)
            
            # 하이퍼파라미터 튜너 생성 (임시 설정 파일 사용)
            import tempfile
            import os
            
            # 임시 기본 설정 파일 생성
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(method_base_config, f, default_flow_style=False)
                temp_base_config_path = f.name
            
            try:
                # 하이퍼파라미터 튜너 생성
                tuner = HyperparameterTuner(tuning_config_path, temp_base_config_path)
                
                # 최적화 실행
                logger.info(f"{method} 하이퍼파라미터 최적화 실행 중...")
                best_params, best_score = tuner.optimize()
                
                # 결과 저장
                logger.info(f"{method} 튜닝 결과 저장 중...")
                tuner.save_results()
                
                # 결과 수집
                method_results = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'tuning_config': method_tuning_config,
                    'base_config': method_base_config
                }
                
                comparison_results['methods'][method] = method_results
                
                # 최고 성능 업데이트
                if best_score > comparison_results['best_score']:
                    comparison_results['best_score'] = best_score
                    comparison_results['best_method'] = method
                
                logger.info(f"{method} 하이퍼파라미터 튜닝 완료: {best_score:.4f}")
                
            finally:
                # 임시 파일 정리
                if os.path.exists(temp_base_config_path):
                    os.unlink(temp_base_config_path)
    
    # 비교 결과 요약
    logger.info("\n=== 리샘플링 하이퍼파라미터 튜닝 비교 결과 요약 ===")
    for method, results in comparison_results['methods'].items():
        best_score = results['best_score']
        logger.info(f"{method}: 최고 성능 = {best_score:.4f}")
    
    logger.info(f"최고 성능 기법: {comparison_results['best_method']} (성능: {comparison_results['best_score']:.4f})")
    
    return comparison_results


def validate_config_files(tuning_config_path: str, base_config_path: str):
    """
    설정 파일들의 유효성을 검증합니다.
    
    Args:
        tuning_config_path: 튜닝 설정 파일 경로
        base_config_path: 기본 설정 파일 경로
    """
    # 파일 존재 확인
    if not Path(tuning_config_path).exists():
        raise FileNotFoundError(f"튜닝 설정 파일을 찾을 수 없습니다: {tuning_config_path}")
    
    if not Path(base_config_path).exists():
        raise FileNotFoundError(f"기본 설정 파일을 찾을 수 없습니다: {base_config_path}")
    
    # YAML 파싱 테스트
    try:
        with open(tuning_config_path, 'r', encoding='utf-8') as f:
            tuning_config = yaml.safe_load(f)
        logger.info("튜닝 설정 파일 파싱 성공")
    except Exception as e:
        raise ValueError(f"튜닝 설정 파일 파싱 실패: {str(e)}")
    
    try:
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        logger.info("기본 설정 파일 파싱 성공")
    except Exception as e:
        raise ValueError(f"기본 설정 파일 파싱 실패: {str(e)}")
    
    # 필수 설정 확인
    required_tuning_keys = ['tuning', 'sampler', 'xgboost_params', 'evaluation', 'results', 'mlflow']
    for key in required_tuning_keys:
        if key not in tuning_config:
            raise ValueError(f"튜닝 설정에 필수 키가 없습니다: {key}")
    
    logger.info("설정 파일 검증 완료")


def run_hyperparameter_tuning(tuning_config_path: str, base_config_path: str, 
                             data_path: str = None, nrows: int = None,
                             resampling_comparison: bool = False, 
                             resampling_methods: List[str] = None):
    """
    하이퍼파라미터 튜닝을 실행합니다.
    
    Args:
        tuning_config_path: 튜닝 설정 파일 경로
        base_config_path: 기본 설정 파일 경로
        data_path: 데이터 파일 경로 (None이면 기본 경로 사용)
        nrows: 사용할 데이터 행 수 (None이면 전체 사용)
        resampling_comparison: 리샘플링 비교 실험 여부
        resampling_methods: 비교할 리샘플링 기법 리스트
    """
    logger.info("=== 하이퍼파라미터 튜닝 시작 ===")
    
    # 설정 파일 검증
    validate_config_files(tuning_config_path, base_config_path)
    
    # 튜닝 설정 로드
    with open(tuning_config_path, 'r', encoding='utf-8') as f:
        tuning_config = yaml.safe_load(f)
    
    # 데이터 경로 확인
    if data_path is None:
        data_path = "data/processed/processed_data_with_features.csv"
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    logger.info(f"데이터 파일: {data_path}")
    if nrows:
        logger.info(f"사용할 데이터 행 수: {nrows}")
    
    # MLflow 실험 설정
    experiment_name = "resampling_tuning_comparison" if resampling_comparison else tuning_config['mlflow']['experiment_name']
    setup_mlflow_experiment(experiment_name)
    
    if resampling_comparison:
        # 리샘플링 비교 실험
        logger.info("=== 리샘플링 하이퍼파라미터 튜닝 비교 실험 ===")
        if resampling_methods is None:
            resampling_methods = ['none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid']
        
        comparison_results = run_resampling_tuning_comparison(
            tuning_config_path, base_config_path, data_path, resampling_methods, nrows
        )
        
        # 비교 결과 로깅
        mlflow.log_param("resampling_tuning_comparison", True)
        mlflow.log_param("resampling_methods", str(resampling_methods))
        mlflow.log_param("best_resampling_method", comparison_results.get('best_method', 'none'))
        mlflow.log_metric("best_resampling_tuning_score", comparison_results.get('best_score', 0.0))
        
        logger.info("리샘플링 하이퍼파라미터 튜닝 비교 실험 완료")
        return comparison_results
    else:
        # 일반 하이퍼파라미터 튜닝
        # 하이퍼파라미터 튜너 생성
        tuner = HyperparameterTuner(tuning_config_path, base_config_path)
        
        # 최적화 실행
        logger.info("하이퍼파라미터 최적화 실행 중...")
        best_params, best_score = tuner.optimize()
        
        # 결과 저장
        logger.info("튜닝 결과 저장 중...")
        tuner.save_results()
        
        # 최종 결과 출력
        logger.info("=== 하이퍼파라미터 튜닝 완료 ===")
        logger.info(f"최고 성능: {best_score:.4f}")
        logger.info("최적 파라미터:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        
        return best_params, best_score


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="하이퍼파라미터 튜닝 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 설정으로 튜닝 실행 (리샘플링 통합 설정 사용)
  python scripts/run_hyperparameter_tuning.py
  
  # 커스텀 설정 파일로 튜닝 실행
  python scripts/run_hyperparameter_tuning.py \\
    --tuning_config configs/hyperparameter_tuning_tpe.yaml \\
    --base_config configs/default_config.yaml
  
  # 특정 모델 타입으로 튜닝 실행
  python scripts/run_hyperparameter_tuning.py \\
    --model-type lightgbm \\
    --tuning_config configs/resampling_config.yaml
  
  # 샘플 데이터로 빠른 테스트
  python scripts/run_hyperparameter_tuning.py \\
    --tuning_config configs/resampling_config.yaml \\
    --nrows 1000
  
  # 리샘플링 기법 비교 실험
  python scripts/run_hyperparameter_tuning.py \\
    --resampling-comparison \\
    --resampling-methods smote borderline_smote adasyn
        """
    )
    
    parser.add_argument(
        "--tuning_config", 
        type=str, 
        default="configs/resampling_config.yaml",
        help="튜닝 설정 파일 경로 (기본값: configs/resampling_config.yaml)"
    )
    
    parser.add_argument(
        "--base_config", 
        type=str, 
        default="configs/default_config.yaml",
        help="기본 설정 파일 경로 (기본값: configs/default_config.yaml)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['xgboost', 'lightgbm', 'random_forest', 'catboost'],
        default=None,
        help="사용할 모델 타입 (설정 파일의 model_type을 덮어씀)"
    )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None,
        help="데이터 파일 경로 (기본값: data/processed/processed_data_with_features.csv)"
    )
    
    parser.add_argument(
        "--nrows", 
        type=int, 
        default=None,
        help="사용할 데이터 행 수 (기본값: 전체 데이터)"
    )
    
    parser.add_argument(
        "--resampling-comparison",
        action="store_true",
        help="리샘플링 기법 비교 실험 실행"
    )
    
    parser.add_argument(
        "--resampling-methods",
        nargs="+",
        choices=['none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid'],
        default=['none', 'smote', 'borderline_smote', 'adasyn'],
        help="비교할 리샘플링 기법들"
    )
    
    parser.add_argument(
        "--mlflow_ui", 
        action="store_true",
        help="튜닝 완료 후 MLflow UI 실행"
    )
    
    args = parser.parse_args()
    
    try:
        # 기본 설정 로드 및 모델 타입 업데이트
        base_config = load_config(args.base_config)
        if args.model_type:
            base_config['model']['model_type'] = args.model_type
            logger.info(f"모델 타입을 {args.model_type}으로 설정했습니다.")
        
        # 하이퍼파라미터 튜닝 실행
        result = run_hyperparameter_tuning(
            tuning_config_path=args.tuning_config,
            base_config_path=args.base_config,
            data_path=args.data_path,
            nrows=args.nrows,
            resampling_comparison=args.resampling_comparison,
            resampling_methods=args.resampling_methods
        )
        
        if args.resampling_comparison:
            print(f"\n🎉 리샘플링 하이퍼파라미터 튜닝 비교 실험이 성공적으로 완료되었습니다!")
            print(f"📊 최고 성능 기법: {result.get('best_method', 'none')}")
            print(f"⚙️  최고 성능: {result.get('best_score', 0.0):.4f}")
            
            print(f"\n📈 각 기법별 성능:")
            for method, method_result in result.get('methods', {}).items():
                print(f"  {method}: {method_result['best_score']:.4f}")
        else:
            best_params, best_score = result
            print(f"\n🎉 하이퍼파라미터 튜닝이 성공적으로 완료되었습니다!")
            print(f"📊 최고 성능: {best_score:.4f}")
            print(f"⚙️  최적 파라미터: {best_params}")
        
        # MLflow UI 실행 옵션
        if args.mlflow_ui:
            print(f"\n🌐 MLflow UI를 실행합니다...")
            print(f"   브라우저에서 http://localhost:5000 으로 접속하세요.")
            print(f"   종료하려면 Ctrl+C를 누르세요.")
            
            import subprocess
            try:
                subprocess.run(["mlflow", "ui"], check=True)
            except KeyboardInterrupt:
                print(f"\n👋 MLflow UI를 종료합니다.")
            except Exception as e:
                print(f"❌ MLflow UI 실행 실패: {str(e)}")
        
    except Exception as e:
        logger.error(f"하이퍼파라미터 튜닝 실패: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 