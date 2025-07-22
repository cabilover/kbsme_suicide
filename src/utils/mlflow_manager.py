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
from contextlib import contextmanager

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

    def print_experiment_summary(self):
        """현재 실험들의 요약 정보를 출력합니다."""
        try:
            experiments = mlflow.search_experiments()
            
            print("=== MLflow 실험 정보 ===")
            print(f"{'실험 ID':<20} {'이름':<40} {'상태':<10} {'Run 수':<8}")
            print("-" * 78)
            
            for exp in experiments:
                # Run 수 계산
                try:
                    runs = mlflow.search_runs(exp.experiment_id, max_results=1)
                    run_count = len(runs) if runs is not None else 0
                except:
                    run_count = 0
                
                print(f"{exp.experiment_id:<20} {exp.name:<40} {exp.lifecycle_stage:<10} {run_count:<8}")
                
        except Exception as e:
            logger.error(f"실험 요약 출력 실패: {e}")

    def cleanup_orphaned_experiments(self, backup: bool = True) -> List[str]:
        """
        Orphaned 실험들을 정리합니다.
        
        Args:
            backup: 삭제 전 백업 여부
            
        Returns:
            삭제된 실험 ID 목록
        """
        deleted_experiments = []
        
        for experiment_dir in self.mlruns_path.iterdir():
            if not experiment_dir.is_dir():
                continue
                
            # 관리용 디렉토리는 제외
            if experiment_dir.name in ['models', '.trash', 'backups']:
                continue
                
            meta_file = experiment_dir / 'meta.yaml'
            
            if not meta_file.exists():
                logger.warning(f"Orphaned 실험 발견: {experiment_dir.name}")
                
                if backup:
                    # 백업 생성
                    backup_path = self.backup_dir / f"{experiment_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    try:
                        shutil.copytree(experiment_dir, backup_path)
                        logger.info(f"Orphaned 실험 백업: {experiment_dir.name} -> {backup_path}")
                    except Exception as e:
                        logger.error(f"백업 생성 실패: {experiment_dir.name} - {e}")
                
                # 실험 디렉토리 삭제
                try:
                    shutil.rmtree(experiment_dir)
                    deleted_experiments.append(experiment_dir.name)
                    logger.info(f"Orphaned 실험 삭제: {experiment_dir.name}")
                except Exception as e:
                    logger.error(f"실험 삭제 실패: {experiment_dir.name} - {e}")
        
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
        
        for experiment_dir in self.mlruns_path.iterdir():
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
                                        logger.debug(f"오래된 run 삭제: {run_dir.name}")
                                    break
                    except Exception as e:
                        logger.warning(f"Run 메타데이터 읽기 실패: {run_dir} - {e}")
        
        logger.info(f"오래된 run 정리 완료: {deleted_count}개 삭제")
        return deleted_count

    def backup_experiment(self, experiment_id: str, backup_name: str = None) -> str:
        """
        실험을 백업합니다.
        
        Args:
            experiment_id: 실험 ID
            backup_name: 백업 이름 (None이면 자동 생성)
            
        Returns:
            백업 경로
        """
        experiment_dir = self.mlruns_path / experiment_id
        
        if not experiment_dir.exists():
            raise ValueError(f"실험 디렉토리가 존재하지 않습니다: {experiment_id}")
        
        if backup_name is None:
            backup_name = f"{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copytree(experiment_dir, backup_path)
            logger.info(f"실험 백업 완료: {experiment_id} -> {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"실험 백업 실패: {experiment_id} - {e}")
            raise

    def restore_experiment(self, backup_path: str, experiment_id: str = None) -> str:
        """
        백업된 실험을 복원합니다.
        
        Args:
            backup_path: 백업 경로
            experiment_id: 복원할 실험 ID (None이면 백업에서 추출)
            
        Returns:
            복원된 실험 ID
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise ValueError(f"백업 경로가 존재하지 않습니다: {backup_path}")
        
        if experiment_id is None:
            experiment_id = backup_path.name
        
        restore_path = self.mlruns_path / experiment_id
        
        try:
            if restore_path.exists():
                shutil.rmtree(restore_path)
            
            shutil.copytree(backup_path, restore_path)
            logger.info(f"실험 복원 완료: {backup_path} -> {experiment_id}")
            return experiment_id
        except Exception as e:
            logger.error(f"실험 복원 실패: {backup_path} - {e}")
            raise

    def validate_experiment_integrity(self, experiment_id: str) -> bool:
        """
        실험의 무결성을 검증합니다.
        
        Args:
            experiment_id: 실험 ID
            
        Returns:
            무결성 검증 결과
        """
        experiment_dir = self.mlruns_path / experiment_id
        
        if not experiment_dir.exists():
            return False
        
        meta_file = experiment_dir / 'meta.yaml'
        if not meta_file.exists():
            return False
        
        try:
            # meta.yaml 파일 읽기 테스트
            with open(meta_file, 'r') as f:
                content = f.read()
                if not content.strip():
                    return False
        except Exception:
            return False
        
        return True

    def repair_experiment(self, experiment_id: str) -> bool:
        """
        손상된 실험을 복구합니다.
        
        Args:
            experiment_id: 실험 ID
            
        Returns:
            복구 성공 여부
        """
        experiment_dir = self.mlruns_path / experiment_id
        
        if not experiment_dir.exists():
            return False
        
        meta_file = experiment_dir / 'meta.yaml'
        
        if not meta_file.exists() or meta_file.stat().st_size == 0:
            try:
                # 기본 meta.yaml 생성
                basic_meta = f"""artifact_uri: ./mlruns/{experiment_id}
experiment_id: {experiment_id}
lifecycle_stage: active
name: experiment_{experiment_id}
"""
                with open(meta_file, 'w') as f:
                    f.write(basic_meta)
                
                logger.info(f"실험 복구 완료: {experiment_id}")
                return True
            except Exception as e:
                logger.error(f"실험 복구 실패: {experiment_id} - {e}")
                return False
        
        return True


@contextmanager
def safe_mlflow_run(experiment_id: str = None, run_name: str = None, nested: bool = False):
    """
    안전한 MLflow run 관리를 위한 컨텍스트 매니저
    
    Args:
        experiment_id: 실험 ID
        run_name: run 이름
        nested: 중첩 실행 여부
        
    Yields:
        MLflow run 객체
    """
    run = None
    try:
        # MLflow run 시작
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested
        )
        logger.info(f"MLflow Run 시작: {run.info.run_id}")
        
        yield run
        
    except Exception as e:
        logger.error(f"MLflow Run 실행 중 오류 발생: {e}")
        # run이 시작되었지만 예외가 발생한 경우
        if run is not None:
            try:
                mlflow.end_run(status="FAILED")
                logger.info(f"MLflow Run 종료 (FAILED): {run.info.run_id}")
            except Exception as end_error:
                logger.error(f"MLflow Run 종료 실패: {end_error}")
        raise
    else:
        # 정상 종료
        if run is not None:
            try:
                mlflow.end_run(status="FINISHED")
                logger.info(f"MLflow Run 종료 (FINISHED): {run.info.run_id}")
            except Exception as end_error:
                logger.error(f"MLflow Run 종료 실패: {end_error}")


def safe_log_param(param_name: str, param_value: Any, logger_instance: logging.Logger = None):
    """
    안전한 MLflow 파라미터 로깅
    
    Args:
        param_name: 파라미터 이름
        param_value: 파라미터 값
        logger_instance: 로거 인스턴스
    """
    if logger_instance is None:
        logger_instance = logger
    
    try:
        mlflow.log_param(param_name, param_value)
    except Exception as e:
        logger_instance.warning(f"MLflow 파라미터 로깅 실패 ({param_name}): {e}")


def safe_log_metric(metric_name: str, metric_value: Any, step: int = None, logger_instance: logging.Logger = None):
    """
    안전한 MLflow 메트릭 로깅
    
    Args:
        metric_name: 메트릭 이름
        metric_value: 메트릭 값
        step: 스텝 번호
        logger_instance: 로거 인스턴스
    """
    if logger_instance is None:
        logger_instance = logger
    
    try:
        mlflow.log_metric(metric_name, metric_value, step=step)
    except Exception as e:
        logger_instance.warning(f"MLflow 메트릭 로깅 실패 ({metric_name}): {e}")


def safe_log_artifact(local_path: str, artifact_path: str = None, logger_instance: logging.Logger = None):
    """
    안전한 MLflow 아티팩트 로깅
    
    Args:
        local_path: 로컬 파일 경로
        artifact_path: 아티팩트 경로
        logger_instance: 로거 인스턴스
    """
    if logger_instance is None:
        logger_instance = logger
    
    try:
        mlflow.log_artifact(local_path, artifact_path)
    except Exception as e:
        logger_instance.warning(f"MLflow 아티팩트 로깅 실패 ({local_path}): {e}")


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