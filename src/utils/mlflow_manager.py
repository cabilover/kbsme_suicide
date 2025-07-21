"""
MLflow 실험 관리 유틸리티

MLflow 실험의 생성, 정리, 백업 등을 관리하여
meta.yaml 관련 경고를 방지하고 실험 디렉토리를 깔끔하게 유지합니다.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

import mlflow
import pandas as pd


logger = logging.getLogger(__name__)


class MLflowExperimentManager:
    """
    MLflow 실험 관리 클래스
    
    실험 생성, 정리, 백업 등의 기능을 제공하여
    MLflow 디렉토리 관리를 자동화합니다.
    """
    
    def __init__(self, tracking_uri: str = "file:./mlruns", backup_dir: str = "mlruns_backups"):
        """
        MLflow 실험 관리자 초기화
        
        Args:
            tracking_uri: MLflow tracking URI
            backup_dir: 백업 디렉토리 경로
        """
        self.tracking_uri = tracking_uri
        self.backup_dir = Path(backup_dir)
        self.mlruns_path = Path(tracking_uri.replace("file:", ""))
        
        # MLflow 설정
        mlflow.set_tracking_uri(tracking_uri)
        
        # 백업 디렉토리 생성
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MLflow 실험 관리자 초기화 완료")
        logger.info(f"  - Tracking URI: {tracking_uri}")
        logger.info(f"  - 백업 디렉토리: {backup_dir}")
    
    def create_experiment_safely(self, experiment_name: str, artifact_location: str = None) -> str:
        """
        안전하게 실험을 생성합니다.
        
        Args:
            experiment_name: 실험 이름
            artifact_location: 아티팩트 저장 위치
            
        Returns:
            실험 ID
        """
        try:
            # 기존 실험 확인
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                # 새 실험 생성
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                logger.info(f"새 실험 생성: {experiment_name} (ID: {experiment_id})")
            else:
                # 기존 실험 사용
                experiment_id = experiment.experiment_id
                logger.info(f"기존 실험 사용: {experiment_name} (ID: {experiment_id})")
                
                # 실험이 삭제된 상태인지 확인
                if experiment.lifecycle_stage == "deleted":
                    logger.warning(f"실험이 삭제된 상태입니다: {experiment_name}")
                    # 실험을 다시 활성화하거나 새로 생성
                    experiment_id = mlflow.create_experiment(
                        experiment_name,
                        artifact_location=artifact_location
                    )
                    logger.info(f"삭제된 실험을 새로 생성: {experiment_name} (ID: {experiment_id})")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"실험 생성 실패: {e}")
            raise
    
    def cleanup_orphaned_experiments(self, backup: bool = True) -> List[str]:
        """
        Orphaned 실험들을 정리합니다.
        
        Args:
            backup: 삭제 전 백업 여부
            
        Returns:
            삭제된 실험 ID 리스트
        """
        orphaned_experiments = self._find_orphaned_experiments()
        deleted_experiments = []
        
        if not orphaned_experiments:
            logger.info("정리할 orphaned 실험이 없습니다.")
            return deleted_experiments
        
        logger.info(f"발견된 orphaned 실험: {len(orphaned_experiments)}개")
        
        for experiment_path in orphaned_experiments:
            experiment_id = Path(experiment_path).name
            
            try:
                if backup:
                    self._backup_experiment(experiment_path)
                
                # 디렉토리 삭제
                shutil.rmtree(experiment_path)
                deleted_experiments.append(experiment_id)
                logger.info(f"Orphaned 실험 삭제 완료: {experiment_id}")
                
            except Exception as e:
                logger.error(f"실험 삭제 실패: {experiment_id} - {e}")
        
        logger.info(f"정리 완료: {len(deleted_experiments)}/{len(orphaned_experiments)} 실험 삭제")
        return deleted_experiments
    
    def cleanup_old_runs(self, days_old: int = 30) -> int:
        """
        오래된 run들을 정리합니다.
        
        Args:
            days_old: 삭제할 기준 일수
            
        Returns:
            삭제된 run 수
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        logger.info(f"{days_old}일 이상 오래된 run 정리 시작")
        
        for experiment_dir in self.mlruns_path.iterdir():
            if not experiment_dir.is_dir() or experiment_dir.name in ['models', '.trash', 'backups']:
                continue
            
            # meta.yaml이 있는 실험만 처리
            meta_file = experiment_dir / 'meta.yaml'
            if not meta_file.exists():
                continue
            
            experiment_id = experiment_dir.name
            
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
                                        logger.debug(f"오래된 run 삭제: {run_dir.name}")
                                    break
                    except Exception as e:
                        logger.warning(f"Run 메타데이터 읽기 실패: {run_dir} - {e}")
        
        logger.info(f"오래된 run 정리 완료: {deleted_count}개 삭제")
        return deleted_count
    

    
    def print_experiment_summary(self) -> None:
        """실험 요약 정보를 출력합니다."""
        summary = self.get_experiment_summary()
        
        print("\n=== MLflow 실험 요약 ===")
        print(f"총 실험 수: {summary['total_experiments']}")
        print(f"  - 활성 실험: {summary['active_experiments']}")
        print(f"  - 삭제된 실험: {summary['deleted_experiments']}")
        print(f"  - Orphaned 실험: {summary['orphaned_experiments']}")
        print(f"총 Run 수: {summary['total_runs']}")
        
        if summary['experiments']:
            print("\n=== 실험 상세 정보 ===")
            print(f"{'실험 ID':<20} {'이름':<30} {'상태':<12} {'Run 수':<8} {'메타':<6}")
            print("-" * 80)
            
            for exp in summary['experiments']:
                meta_status = "✓" if exp['has_meta'] else "✗"
                print(f"{exp['experiment_id']:<20} {exp['name']:<30} {exp['lifecycle_stage']:<12} {exp['run_count']:<8} {meta_status:<6}")
    
    def _find_orphaned_experiments(self) -> List[str]:
        """meta.yaml이 없는 실험 디렉토리를 찾습니다."""
        orphaned_dirs = []
        
        for item in self.mlruns_path.iterdir():
            if item.is_dir() and item.name not in ['models', '.trash', 'backups']:
                meta_file = item / 'meta.yaml'
                if not meta_file.exists():
                    orphaned_dirs.append(str(item))
        
        return orphaned_dirs
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        실험 요약 정보를 반환합니다.
        
        Returns:
            실험 요약 정보 딕셔너리
        """
        summary = {
            'total_experiments': 0,
            'active_experiments': 0,
            'deleted_experiments': 0,
            'orphaned_experiments': 0,
            'total_runs': 0,
            'experiments': []
        }
        
        for experiment_dir in self.mlruns_path.iterdir():
            if not experiment_dir.is_dir() or experiment_dir.name in ['models', '.trash', 'backups']:
                continue
            
            experiment_id = experiment_dir.name
            meta_file = experiment_dir / 'meta.yaml'
            
            experiment_info = {
                'experiment_id': experiment_id,
                'name': 'Unknown',
                'lifecycle_stage': 'orphaned',
                'run_count': 0,
                'has_meta': meta_file.exists()
            }
            
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        lines = f.readlines()
                        
                        for line in lines:
                            if line.startswith('name:'):
                                experiment_info['name'] = line.split(':', 1)[1].strip()
                            elif line.startswith('lifecycle_stage:'):
                                experiment_info['lifecycle_stage'] = line.split(':', 1)[1].strip()
                    
                    # run 수 계산
                    run_count = len([d for d in experiment_dir.iterdir() if d.is_dir()])
                    experiment_info['run_count'] = run_count
                    summary['total_runs'] += run_count
                    
                    # 상태별 카운트
                    if experiment_info['lifecycle_stage'] == 'active':
                        summary['active_experiments'] += 1
                    elif experiment_info['lifecycle_stage'] == 'deleted':
                        summary['deleted_experiments'] += 1
                    
                except Exception as e:
                    logger.warning(f"실험 메타데이터 읽기 실패: {experiment_id} - {e}")
            else:
                summary['orphaned_experiments'] += 1
            
            summary['experiments'].append(experiment_info)
            summary['total_experiments'] += 1
        
        return summary
    
    def _backup_experiment(self, experiment_path: str) -> bool:
        """실험 데이터를 백업합니다."""
        try:
            experiment_name = Path(experiment_path).name
            backup_path = self.backup_dir / f"backup_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
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


def setup_mlflow_experiment_safely(experiment_name: str, tracking_uri: str = "file:./mlruns") -> str:
    """
    안전하게 MLflow 실험을 설정합니다.
    
    Args:
        experiment_name: 실험 이름
        tracking_uri: MLflow tracking URI
        
    Returns:
        실험 ID
    """
    manager = MLflowExperimentManager(tracking_uri)
    return manager.create_experiment_safely(experiment_name)


def cleanup_mlflow_experiments(tracking_uri: str = "file:./mlruns", backup: bool = True, days_old: int = 30) -> Dict[str, Any]:
    """
    MLflow 실험들을 정리합니다.
    
    Args:
        tracking_uri: MLflow tracking URI
        backup: 삭제 전 백업 여부
        days_old: 오래된 run 삭제 기준 일수
        
    Returns:
        정리 결과 딕셔너리
    """
    manager = MLflowExperimentManager(tracking_uri)
    
    # Orphaned 실험 정리
    deleted_experiments = manager.cleanup_orphaned_experiments(backup=backup)
    
    # 오래된 run 정리
    deleted_runs = manager.cleanup_old_runs(days_old=days_old)
    
    return {
        'deleted_experiments': deleted_experiments,
        'deleted_runs_count': deleted_runs,
        'total_deleted_experiments': len(deleted_experiments)
    } 