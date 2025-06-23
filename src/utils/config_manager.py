"""
설정 관리 시스템
계층적 설정 파일 로딩 및 병합을 위한 유틸리티
"""

import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    계층적 설정 파일 관리 클래스
    기본 설정, 모델 설정, 실험 설정을 병합하여 완전한 설정을 생성
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        ConfigManager 초기화
        
        Args:
            config_dir: 설정 파일 디렉토리 경로
        """
        self.config_dir = Path(config_dir)
        self.base_dir = self.config_dir / "base"
        self.models_dir = self.config_dir / "models"
        self.experiments_dir = self.config_dir / "experiments"
        self.templates_dir = self.config_dir / "templates"
        
        # 디렉토리 존재 확인
        self._validate_directories()
    
    def _validate_directories(self):
        """필요한 디렉토리들이 존재하는지 확인"""
        required_dirs = [self.base_dir, self.models_dir, self.experiments_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.warning(f"디렉토리가 존재하지 않습니다: {dir_path}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        설정 파일 로딩
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            로딩된 설정 딕셔너리
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.debug(f"설정 파일 로딩 완료: {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML 파싱 오류: {config_path} - {e}")
            raise
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """
        깊은 병합 구현
        dict2의 값이 dict1을 덮어씀 (중첩된 딕셔너리도 재귀적으로 병합)
        
        Args:
            dict1: 기본 딕셔너리
            dict2: 덮어쓸 딕셔너리
            
        Returns:
            병합된 딕셔너리
        """
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def merge_configs(self, base_configs: List[str], model_config: str, 
                     experiment_config: Optional[str] = None) -> Dict[str, Any]:
        """
        여러 설정 파일을 병합
        
        Args:
            base_configs: 기본 설정 파일 경로 리스트
            model_config: 모델 설정 파일 경로
            experiment_config: 실험 설정 파일 경로 (선택사항)
            
        Returns:
            병합된 설정 딕셔너리
        """
        merged_config = {}
        
        # 기본 설정들 병합
        for base_config in base_configs:
            if Path(base_config).exists():
                base_data = self.load_config(base_config)
                merged_config = self._deep_merge(merged_config, base_data)
                logger.debug(f"기본 설정 병합: {base_config}")
            else:
                logger.warning(f"기본 설정 파일이 존재하지 않습니다: {base_config}")
        
        # 모델 설정 병합
        if Path(model_config).exists():
            model_data = self.load_config(model_config)
            merged_config = self._deep_merge(merged_config, model_data)
            logger.debug(f"모델 설정 병합: {model_config}")
        else:
            logger.warning(f"모델 설정 파일이 존재하지 않습니다: {model_config}")
        
        # 실험 설정 병합 (있는 경우)
        if experiment_config and Path(experiment_config).exists():
            experiment_data = self.load_config(experiment_config)
            
            # overrides 섹션이 있으면 적용
            if 'overrides' in experiment_data:
                overrides = experiment_data['overrides']
                merged_config = self._deep_merge(merged_config, overrides)
                logger.debug(f"실험 오버라이드 적용: {experiment_config}")
            
            # overrides 외의 다른 설정들도 병합
            experiment_config_clean = {k: v for k, v in experiment_data.items() if k != 'overrides'}
            if experiment_config_clean:
                merged_config = self._deep_merge(merged_config, experiment_config_clean)
                logger.debug(f"실험 설정 병합: {experiment_config}")
        elif experiment_config:
            logger.warning(f"실험 설정 파일이 존재하지 않습니다: {experiment_config}")
        
        return merged_config
    
    def create_experiment_config(self, model_type: str, experiment_type: str = None) -> Dict[str, Any]:
        """
        실험 설정 생성
        
        Args:
            model_type: 모델 타입 ('xgboost', 'catboost', 'lightgbm', 'random_forest')
            experiment_type: 실험 타입 ('focal_loss', 'resampling', 'hyperparameter_tuning')
            
        Returns:
            완전한 실험 설정 딕셔너리
        """
        # 기본 설정 파일들
        base_configs = [
            str(self.base_dir / "common.yaml"),
            str(self.base_dir / "validation.yaml"),
            str(self.base_dir / "evaluation.yaml"),
            str(self.base_dir / "mlflow.yaml")
        ]
        
        # 모델 설정 파일
        model_config = str(self.models_dir / f"{model_type}.yaml")
        
        # 실험 설정 파일 (있는 경우)
        experiment_config = None
        if experiment_type:
            experiment_config = str(self.experiments_dir / f"{experiment_type}.yaml")
        
        # 설정 병합
        config = self.merge_configs(base_configs, model_config, experiment_config)
        
        logger.info(f"실험 설정 생성 완료: model={model_type}, experiment={experiment_type}")
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        설정 유효성 검증
        
        Args:
            config: 검증할 설정 딕셔너리
            
        Returns:
            유효성 여부
        """
        required_sections = ['data', 'features', 'model', 'validation']
        
        for section in required_sections:
            if section not in config:
                logger.error(f"필수 섹션이 누락되었습니다: {section}")
                return False
        
        # 모델 타입 확인
        if 'model' in config and 'model_type' not in config['model']:
            logger.error("모델 타입이 지정되지 않았습니다")
            return False
        
        logger.info("설정 유효성 검증 통과")
        return True
    
    def save_config(self, config: Dict[str, Any], output_path: str):
        """
        설정을 파일로 저장
        
        Args:
            config: 저장할 설정 딕셔너리
            output_path: 저장할 파일 경로
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            logger.info(f"설정 파일 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"설정 파일 저장 실패: {output_path} - {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        사용 가능한 모델 목록 반환
        
        Returns:
            사용 가능한 모델 타입 리스트
        """
        if not self.models_dir.exists():
            return []
        
        model_files = list(self.models_dir.glob("*.yaml"))
        models = [f.stem for f in model_files]
        return models
    
    def get_available_experiments(self) -> List[str]:
        """
        사용 가능한 실험 목록 반환
        
        Returns:
            사용 가능한 실험 타입 리스트
        """
        if not self.experiments_dir.exists():
            return []
        
        experiment_files = list(self.experiments_dir.glob("*.yaml"))
        experiments = [f.stem for f in experiment_files]
        return experiments
    
    def print_config_summary(self, config: Dict[str, Any]):
        """
        설정 요약 출력
        
        Args:
            config: 출력할 설정 딕셔너리
        """
        print("\n" + "="*50)
        print("설정 요약")
        print("="*50)
        
        # 모델 정보
        if 'model' in config:
            model_type = config['model'].get('model_type', 'Unknown')
            print(f"모델 타입: {model_type}")
        
        # 피처 정보
        if 'features' in config:
            target_cols = config['features'].get('target_columns', [])
            print(f"타겟 변수: {len(target_cols)}개")
            print(f"  - {', '.join(target_cols)}")
        
        # 검증 정보
        if 'validation' in config:
            strategy = config['validation'].get('strategy', 'Unknown')
            n_folds = config['validation'].get('num_cv_folds', 'Unknown')
            print(f"검증 전략: {strategy} ({n_folds}폴드)")
        
        # 실험 정보
        if 'experiment' in config:
            exp_name = config['experiment'].get('name', 'Unknown')
            print(f"실험 이름: {exp_name}")
        
        print("="*50) 