#!/usr/bin/env python3
"""
하이퍼파라미터 튜닝 실행 스크립트

Optuna를 활용한 하이퍼파라미터 최적화를 실행합니다.
MLflow와 연동되어 실험 추적이 가능하며, 다양한 튜닝 전략을 지원합니다.
"""

import argparse
import logging
import sys
from pathlib import Path
import mlflow
import yaml

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.hyperparameter_tuning import HyperparameterTuner

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
                             data_path: str = None, nrows: int = None):
    """
    하이퍼파라미터 튜닝을 실행합니다.
    
    Args:
        tuning_config_path: 튜닝 설정 파일 경로
        base_config_path: 기본 설정 파일 경로
        data_path: 데이터 파일 경로 (None이면 기본 경로 사용)
        nrows: 사용할 데이터 행 수 (None이면 전체 사용)
    """
    logger.info("=== 하이퍼파라미터 튜닝 시작 ===")
    
    # 설정 파일 검증
    validate_config_files(tuning_config_path, base_config_path)
    
    # 튜닝 설정 로드
    with open(tuning_config_path, 'r', encoding='utf-8') as f:
        tuning_config = yaml.safe_load(f)
    
    # MLflow 실험 설정
    experiment_name = tuning_config['mlflow']['experiment_name']
    setup_mlflow_experiment(experiment_name)
    
    # 데이터 경로 확인
    if data_path is None:
        data_path = "data/processed/processed_data_with_features.csv"
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    logger.info(f"데이터 파일: {data_path}")
    if nrows:
        logger.info(f"사용할 데이터 행 수: {nrows}")
    
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
  # 기본 설정으로 튜닝 실행
  python scripts/run_hyperparameter_tuning.py
  
  # 커스텀 설정 파일로 튜닝 실행
  python scripts/run_hyperparameter_tuning.py \\
    --tuning_config configs/hyperparameter_tuning_tpe.yaml \\
    --base_config configs/default_config.yaml
  
  # 샘플 데이터로 빠른 테스트
  python scripts/run_hyperparameter_tuning.py \\
    --tuning_config configs/hyperparameter_tuning.yaml \\
    --nrows 1000
        """
    )
    
    parser.add_argument(
        "--tuning_config", 
        type=str, 
        default="configs/hyperparameter_tuning.yaml",
        help="튜닝 설정 파일 경로 (기본값: configs/hyperparameter_tuning.yaml)"
    )
    
    parser.add_argument(
        "--base_config", 
        type=str, 
        default="configs/default_config.yaml",
        help="기본 설정 파일 경로 (기본값: configs/default_config.yaml)"
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
        "--mlflow_ui", 
        action="store_true",
        help="튜닝 완료 후 MLflow UI 실행"
    )
    
    args = parser.parse_args()
    
    try:
        # 하이퍼파라미터 튜닝 실행
        best_params, best_score = run_hyperparameter_tuning(
            tuning_config_path=args.tuning_config,
            base_config_path=args.base_config,
            data_path=args.data_path,
            nrows=args.nrows
        )
        
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