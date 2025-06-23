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
from src.utils.config_manager import ConfigManager

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
                tuner = HyperparameterTuner(temp_base_config_path, temp_base_config_path)
                
                # 최적화 실행
                logger.info(f"{method} 하이퍼파라미터 최적화 실행 중...")
                best_params, best_score = tuner.optimize(start_mlflow_run=False)
                
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


def run_resampling_tuning_comparison_with_configmanager(
    model_type: str,
    experiment_type: str,
    data_path: str,
    resampling_methods: List[str],
    nrows: int = None
) -> Dict[str, Any]:
    """
    ConfigManager 기반 다양한 리샘플링 기법에 대해 하이퍼파라미터 튜닝을 비교하는 실험을 실행합니다.
    """
    logger.info("=== [ConfigManager] 리샘플링 하이퍼파라미터 튜닝 비교 실험 시작 ===")
    config_manager = ConfigManager()

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
    # config는 리샘플링과 무관하게 분할에만 사용
    base_config = config_manager.create_experiment_config(model_type, experiment_type)
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

    # 상위 MLflow run 시작
    with mlflow.start_run(run_name=f"resampling_comparison_{model_type}"):
        for method in resampling_methods:
            logger.info(f"\n--- {method.upper()} 하이퍼파라미터 튜닝 시작 ---")
            # config 생성 및 리샘플링 설정 수정
            config = config_manager.create_experiment_config(model_type, experiment_type)
            config['resampling'] = {
                'enabled': method != 'none',
                'method': method if method != 'none' else 'smote',
                'random_state': 42
            }
            # 리샘플링 기법별 파라미터 설정
            if method == 'smote':
                config['resampling']['smote_k_neighbors'] = 5
            elif method == 'borderline_smote':
                config['resampling']['borderline_smote_k_neighbors'] = 5
            elif method == 'adasyn':
                config['resampling']['adasyn_k_neighbors'] = 5
            elif method == 'under_sampling':
                config['resampling']['under_sampling_strategy'] = 'random'
            elif method == 'hybrid':
                config['resampling']['hybrid_strategy'] = 'smote_tomek'
            # 데이터 경로 반영
            config['data']['data_path'] = data_path
            # MLflow 중첩 실행
            with mlflow.start_run(run_name=f"resampling_tuning_{method}", nested=True):
                # 튜닝 파라미터 로깅 (tuning_config는 config에서 추출)
                log_tuning_params(config, config)
                # 임시 config 파일 생성 및 튜너 실행
                import tempfile, yaml, os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                    temp_config_path = f.name
                try:
                    tuner = HyperparameterTuner(temp_config_path, temp_config_path)
                    logger.info(f"{method} 하이퍼파라미터 최적화 실행 중...")
                    best_params, best_score = tuner.optimize(start_mlflow_run=False)
                    logger.info(f"{method} 튜닝 결과 저장 중...")
                    tuner.save_results()
                    method_results = {
                        'best_params': best_params,
                        'best_score': best_score,
                        'config': config
                    }
                    comparison_results['methods'][method] = method_results
                    if best_score > comparison_results['best_score']:
                        comparison_results['best_score'] = best_score
                        comparison_results['best_method'] = method
                    logger.info(f"{method} 하이퍼파라미터 튜닝 완료: {best_score:.4f}")
                finally:
                    if os.path.exists(temp_config_path):
                        os.unlink(temp_config_path)
    
    # 비교 결과 요약
    logger.info("\n=== 리샘플링 하이퍼파라미터 튜닝 비교 결과 요약 ===")
    for method, results in comparison_results['methods'].items():
        best_score = results['best_score']
        logger.info(f"{method}: 최고 성능 = {best_score:.4f}")
    logger.info(f"최고 성능 기법: {comparison_results['best_method']} (성능: {comparison_results['best_score']:.4f})")
    return comparison_results


def run_hyperparameter_tuning_with_config(config: Dict[str, Any], data_path: str = None, nrows: int = None, resampling_comparison: bool = False, resampling_methods: List[str] = None):
    """
    병합된 config 객체를 받아 하이퍼파라미터 튜닝을 실행합니다.
    """
    # 데이터 경로 확인
    if data_path is None:
        data_path = config.get('data', {}).get('data_path', "data/processed/processed_data_with_features.csv")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    logger.info(f"데이터 파일: {data_path}")
    if nrows:
        logger.info(f"사용할 데이터 행 수: {nrows}")
    
    # MLflow 실험 설정
    experiment_name = config.get('mlflow', {}).get('experiment_name', 'hyperparameter_tuning')
    setup_mlflow_experiment(experiment_name)
    
    # 리샘플링 비교 실험
    if resampling_comparison:
        logger.info("=== 리샘플링 하이퍼파라미터 튜닝 비교 실험 ===")
        if resampling_methods is None:
            resampling_methods = ['none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid']
        # ConfigManager 기반 비교 함수 호출
        model_type = config['model']['model_type'] if 'model' in config and 'model_type' in config['model'] else None
        experiment_type = config.get('experiment_type', 'resampling')
        return run_resampling_tuning_comparison_with_configmanager(
            model_type=model_type,
            experiment_type=experiment_type,
            data_path=data_path,
            resampling_methods=resampling_methods,
            nrows=nrows
        )
    else:
        # 일반 하이퍼파라미터 튜닝 (MLflow run 시작)
        with mlflow.start_run():
            # 기존 코드 그대로 유지
            import tempfile, yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                merged_config_path = f.name
            try:
                tuner = HyperparameterTuner(merged_config_path, merged_config_path)
                logger.info("하이퍼파라미터 최적화 실행 중...")
                best_params, best_score = tuner.optimize()
                logger.info("튜닝 결과 저장 중...")
                tuner.save_results()
                logger.info("=== 하이퍼파라미터 튜닝 완료 ===")
                logger.info(f"최고 성능: {best_score:.4f}")
                logger.info("최적 파라미터:")
                for param, value in best_params.items():
                    logger.info(f"  {param}: {value}")
                return best_params, best_score
            finally:
                Path(merged_config_path).unlink(missing_ok=True)


def main():
    """메인 함수 (ConfigManager 기반 계층적 config 병합 지원)"""
    parser = argparse.ArgumentParser(
        description="하이퍼파라미터 튜닝 실행 (ConfigManager 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 계층적 config로 튜닝 실행
  python scripts/run_hyperparameter_tuning.py --model-type xgboost --experiment-type hyperparameter_tuning
  # legacy 단일 파일 config로 튜닝 실행
  python scripts/run_hyperparameter_tuning.py --tuning_config configs/hyperparameter_tuning.yaml --base_config configs/default_config.yaml
        """
    )
    parser.add_argument("--model-type", type=str, choices=['xgboost', 'lightgbm', 'random_forest', 'catboost'], default=None, help="사용할 모델 타입 (계층적 config 병합)")
    parser.add_argument("--experiment-type", type=str, default=None, help="실험 타입 (focal_loss, resampling, hyperparameter_tuning 등)")
    parser.add_argument("--tuning_config", type=str, default=None, help="(legacy) 튜닝 설정 파일 경로")
    parser.add_argument("--base_config", type=str, default=None, help="(legacy) 기본 설정 파일 경로")
    parser.add_argument("--data_path", type=str, default=None, help="데이터 파일 경로")
    parser.add_argument("--nrows", type=int, default=None, help="사용할 데이터 행 수")
    parser.add_argument("--resampling-comparison", action="store_true", help="리샘플링 기법 비교 실험 실행")
    parser.add_argument("--resampling-methods", nargs="+", choices=['none', 'smote', 'borderline_smote', 'adasyn', 'under_sampling', 'hybrid'], default=None, help="비교할 리샘플링 기법들")
    parser.add_argument("--mlflow_ui", action="store_true", help="튜닝 완료 후 MLflow UI 실행")
    args = parser.parse_args()
    try:
        if args.model_type:
            # ConfigManager 기반 계층적 config 병합
            config_manager = ConfigManager()
            config = config_manager.create_experiment_config(args.model_type, args.experiment_type)
            if args.data_path:
                config['data']['data_path'] = args.data_path
            result = run_hyperparameter_tuning_with_config(
                config,
                data_path=args.data_path,
                nrows=args.nrows,
                resampling_comparison=args.resampling_comparison,
                resampling_methods=args.resampling_methods
            )
        elif args.tuning_config and args.base_config:
            # legacy: 단일 파일 기반
            result = run_resampling_tuning_comparison(
                tuning_config_path=args.tuning_config,
                base_config_path=args.base_config,
                data_path=args.data_path,
                resampling_methods=args.resampling_methods,
                nrows=args.nrows
            )
        else:
            raise ValueError("model-type 또는 tuning_config+base_config 중 하나는 반드시 지정해야 합니다.")
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