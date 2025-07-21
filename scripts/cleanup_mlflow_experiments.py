#!/usr/bin/env python3
"""
MLflow 실험 디렉토리 정리 스크립트

meta.yaml이 없는 실험 디렉토리를 찾아서 정리하고,
orphaned 실험들을 삭제하여 MLflow 경고를 해결합니다.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import mlflow
from datetime import datetime, timedelta


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/mlflow_cleanup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def find_orphaned_experiments(mlruns_path: str) -> List[str]:
    """
    meta.yaml이 없는 실험 디렉토리를 찾습니다.
    
    Args:
        mlruns_path: MLflow runs 디렉토리 경로
        
    Returns:
        orphaned 실험 디렉토리 경로 리스트
    """
    orphaned_dirs = []
    mlruns_path = Path(mlruns_path)
    
    if not mlruns_path.exists():
        logger.warning(f"MLruns 디렉토리가 존재하지 않습니다: {mlruns_path}")
        return orphaned_dirs
    
    # 실험 디렉토리 찾기 (숫자/문자로 된 디렉토리)
    for item in mlruns_path.iterdir():
        if item.is_dir() and item.name not in ['models', '.trash']:
            # meta.yaml 파일이 없는지 확인
            meta_file = item / 'meta.yaml'
            if not meta_file.exists():
                orphaned_dirs.append(str(item))
                logger.info(f"Orphaned 실험 발견: {item.name}")
    
    return orphaned_dirs


def backup_experiment_data(experiment_path: str, backup_dir: str) -> bool:
    """
    실험 데이터를 백업합니다.
    
    Args:
        experiment_path: 실험 디렉토리 경로
        backup_dir: 백업 디렉토리 경로
        
    Returns:
        백업 성공 여부
    """
    try:
        experiment_name = Path(experiment_path).name
        backup_path = Path(backup_dir) / f"backup_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if Path(experiment_path).exists():
            shutil.copytree(experiment_path, backup_path)
            logger.info(f"실험 데이터 백업 완료: {backup_path}")
            return True
        else:
            logger.warning(f"실험 디렉토리가 존재하지 않습니다: {experiment_path}")
            return False
    except Exception as e:
        logger.error(f"백업 실패: {e}")
        return False


def remove_orphaned_experiment(experiment_path: str, backup: bool = True) -> bool:
    """
    orphaned 실험 디렉토리를 삭제합니다.
    
    Args:
        experiment_path: 실험 디렉토리 경로
        backup: 백업 여부
        
    Returns:
        삭제 성공 여부
    """
    try:
        if backup:
            backup_dir = Path('mlruns/backups')
            backup_dir.mkdir(exist_ok=True)
            if not backup_experiment_data(experiment_path, str(backup_dir)):
                logger.warning(f"백업 실패로 인해 삭제를 건너뜁니다: {experiment_path}")
                return False
        
        # 디렉토리 삭제
        shutil.rmtree(experiment_path)
        logger.info(f"Orphaned 실험 삭제 완료: {experiment_path}")
        return True
        
    except Exception as e:
        logger.error(f"실험 삭제 실패: {e}")
        return False


def cleanup_old_runs(mlruns_path: str, days_old: int = 30) -> int:
    """
    오래된 run들을 정리합니다.
    
    Args:
        mlruns_path: MLflow runs 디렉토리 경로
        days_old: 삭제할 기준 일수
        
    Returns:
        삭제된 run 수
    """
    cutoff_date = datetime.now() - timedelta(days=days_old)
    deleted_count = 0
    
    mlruns_path = Path(mlruns_path)
    
    for experiment_dir in mlruns_path.iterdir():
        if not experiment_dir.is_dir() or experiment_dir.name in ['models', '.trash', 'backups']:
            continue
            
        # meta.yaml이 있는 실험만 처리
        meta_file = experiment_dir / 'meta.yaml'
        if not meta_file.exists():
            continue
            
        # run 디렉토리들 확인
        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            run_meta_file = run_dir / 'meta.yaml'
            if run_meta_file.exists():
                try:
                    # run 생성 시간 확인
                    with open(run_meta_file, 'r') as f:
                        for line in f:
                            if line.startswith('start_time:'):
                                start_time_str = line.split(':', 1)[1].strip()
                                start_time = datetime.fromtimestamp(int(start_time_str) / 1000)
                                
                                if start_time < cutoff_date:
                                    shutil.rmtree(run_dir)
                                    deleted_count += 1
                                    logger.info(f"오래된 run 삭제: {run_dir}")
                                break
                except Exception as e:
                    logger.warning(f"Run 메타데이터 읽기 실패: {run_dir} - {e}")
    
    return deleted_count


def create_missing_meta_yaml(experiment_path: str, experiment_name: str = None) -> bool:
    """
    누락된 meta.yaml 파일을 생성합니다.
    
    Args:
        experiment_path: 실험 디렉토리 경로
        experiment_name: 실험 이름 (None이면 디렉토리명 사용)
        
    Returns:
        생성 성공 여부
    """
    try:
        experiment_path = Path(experiment_path)
        experiment_id = experiment_path.name
        
        if experiment_name is None:
            experiment_name = f"Experiment_{experiment_id}"
        
        meta_content = f"""artifact_location: file://{experiment_path.absolute()}
creation_time: {int(datetime.now().timestamp() * 1000)}
experiment_id: '{experiment_id}'
last_update_time: {int(datetime.now().timestamp() * 1000)}
lifecycle_stage: active
name: {experiment_name}
"""
        
        meta_file = experiment_path / 'meta.yaml'
        with open(meta_file, 'w') as f:
            f.write(meta_content)
        
        logger.info(f"meta.yaml 생성 완료: {meta_file}")
        return True
        
    except Exception as e:
        logger.error(f"meta.yaml 생성 실패: {e}")
        return False


def list_experiments_info(mlruns_path: str) -> None:
    """
    모든 실험 정보를 출력합니다.
    
    Args:
        mlruns_path: MLflow runs 디렉토리 경로
    """
    mlruns_path = Path(mlruns_path)
    
    print("\n=== MLflow 실험 정보 ===")
    print(f"{'실험 ID':<20} {'이름':<30} {'상태':<10} {'Run 수':<8}")
    print("-" * 70)
    
    for experiment_dir in mlruns_path.iterdir():
        if not experiment_dir.is_dir() or experiment_dir.name in ['models', '.trash', 'backups']:
            continue
            
        experiment_id = experiment_dir.name
        meta_file = experiment_dir / 'meta.yaml'
        
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    lines = f.readlines()
                    name = "Unknown"
                    lifecycle = "Unknown"
                    
                    for line in lines:
                        if line.startswith('name:'):
                            name = line.split(':', 1)[1].strip()
                        elif line.startswith('lifecycle_stage:'):
                            lifecycle = line.split(':', 1)[1].strip()
                    
                    # run 수 계산
                    run_count = len([d for d in experiment_dir.iterdir() if d.is_dir()])
                    
                    print(f"{experiment_id:<20} {name:<30} {lifecycle:<10} {run_count:<8}")
                    
            except Exception as e:
                print(f"{experiment_id:<20} {'Error':<30} {'Error':<10} {'Error':<8}")
        else:
            print(f"{experiment_id:<20} {'Orphaned':<30} {'Missing':<10} {'N/A':<8}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="MLflow 실험 디렉토리 정리")
    parser.add_argument("--mlruns-path", type=str, default="mlruns", help="MLflow runs 디렉토리 경로")
    parser.add_argument("--action", type=str, choices=['list', 'cleanup', 'backup', 'create-meta'], 
                       default='list', help="수행할 작업")
    parser.add_argument("--backup", action="store_true", help="삭제 전 백업 (cleanup 모드에서)")
    parser.add_argument("--force", action="store_true", help="확인 없이 실행")
    parser.add_argument("--days-old", type=int, default=30, help="오래된 run 삭제 기준 일수")
    parser.add_argument("--experiment-name", type=str, help="생성할 실험 이름 (create-meta 모드에서)")
    
    args = parser.parse_args()
    
    # 로깅 설정
    global logger
    logger = setup_logging()
    
    # logs 디렉토리 생성
    Path('logs').mkdir(exist_ok=True)
    
    logger.info(f"MLflow 정리 작업 시작: {args.action}")
    
    if args.action == 'list':
        list_experiments_info(args.mlruns_path)
        
    elif args.action == 'cleanup':
        # Orphaned 실험 찾기
        orphaned_experiments = find_orphaned_experiments(args.mlruns_path)
        
        if not orphaned_experiments:
            logger.info("정리할 orphaned 실험이 없습니다.")
            return
        
        print(f"\n발견된 orphaned 실험: {len(orphaned_experiments)}개")
        for exp in orphaned_experiments:
            print(f"  - {exp}")
        
        if not args.force:
            response = input(f"\n이 실험들을 삭제하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                logger.info("사용자가 삭제를 취소했습니다.")
                return
        
        # 삭제 실행
        deleted_count = 0
        for experiment_path in orphaned_experiments:
            if remove_orphaned_experiment(experiment_path, backup=args.backup):
                deleted_count += 1
        
        logger.info(f"정리 완료: {deleted_count}/{len(orphaned_experiments)} 실험 삭제")
        
        # 오래된 run 정리
        old_runs_deleted = cleanup_old_runs(args.mlruns_path, args.days_old)
        logger.info(f"오래된 run 정리 완료: {old_runs_deleted}개 삭제")
        
    elif args.action == 'backup':
        orphaned_experiments = find_orphaned_experiments(args.mlruns_path)
        
        if not orphaned_experiments:
            logger.info("백업할 orphaned 실험이 없습니다.")
            return
        
        backup_dir = Path('mlruns/backups')
        backup_dir.mkdir(exist_ok=True)
        
        for experiment_path in orphaned_experiments:
            backup_experiment_data(experiment_path, str(backup_dir))
            
    elif args.action == 'create-meta':
        orphaned_experiments = find_orphaned_experiments(args.mlruns_path)
        
        if not orphaned_experiments:
            logger.info("meta.yaml을 생성할 orphaned 실험이 없습니다.")
            return
        
        for experiment_path in orphaned_experiments:
            create_missing_meta_yaml(experiment_path, args.experiment_name)
    
    logger.info("MLflow 정리 작업 완료")


if __name__ == "__main__":
    main() 