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
        logger.info("=== 설정 요약 ===")
        logger.info(f"모델 타입: {config.get('model', {}).get('model_type', 'N/A')}")
        logger.info(f"데이터 경로: {config.get('data', {}).get('data_path', 'N/A')}")
        logger.info(f"검증 전략: {config.get('validation', {}).get('strategy', 'N/A')}")
        logger.info(f"CV 폴드 수: {config.get('validation', {}).get('num_cv_folds', 'N/A')}")
        logger.info(f"튜닝 시도 횟수: {config.get('tuning', {}).get('n_trials', 'N/A')}")
    
    def apply_command_line_args(self, config: Dict[str, Any], args: Any) -> Dict[str, Any]:
        """
        명령행 인자를 config에 적용
        
        Args:
            config: 기본 설정 딕셔너리
            args: argparse.Namespace 객체
            
        Returns:
            인자가 적용된 설정 딕셔너리
        """
        config = copy.deepcopy(config)
        
        # 데이터 관련 인자
        if hasattr(args, 'data_path') and args.data_path:
            config['data']['data_path'] = args.data_path
            logger.info(f"데이터 경로 업데이트: {args.data_path}")
        
        if hasattr(args, 'nrows') and args.nrows:
            config['data']['nrows'] = args.nrows
            logger.info(f"데이터 행 수 제한: {args.nrows}")
        
        # 검증 관련 인자
        if hasattr(args, 'split_strategy') and args.split_strategy:
            config['validation']['strategy'] = args.split_strategy
            logger.info(f"분할 전략 업데이트: {args.split_strategy}")
        
        if hasattr(args, 'cv_folds') and args.cv_folds:
            config['validation']['num_cv_folds'] = args.cv_folds
            logger.info(f"CV 폴드 수 업데이트: {args.cv_folds}")
        
        if hasattr(args, 'test_size') and args.test_size:
            config['validation']['test_size'] = args.test_size
            logger.info(f"테스트 세트 비율 업데이트: {args.test_size}")
        
        if hasattr(args, 'random_state') and args.random_state:
            config['validation']['random_state'] = args.random_state
            logger.info(f"랜덤 시드 업데이트: {args.random_state}")
        
        # 튜닝 관련 인자
        if hasattr(args, 'n_trials') and args.n_trials:
            config['tuning']['n_trials'] = args.n_trials
            logger.info(f"튜닝 시도 횟수 업데이트: {args.n_trials}")
        
        if hasattr(args, 'tuning_direction') and args.tuning_direction:
            config['tuning']['direction'] = args.tuning_direction
            logger.info(f"튜닝 방향 업데이트: {args.tuning_direction}")
        
        if hasattr(args, 'primary_metric') and args.primary_metric:
            config['evaluation']['primary_metric'] = args.primary_metric
            logger.info(f"주요 평가 지표 업데이트: {args.primary_metric}")
        
        if hasattr(args, 'n_jobs') and args.n_jobs:
            config['tuning']['n_jobs'] = args.n_jobs
            logger.info(f"병렬 처리 작업 수 업데이트: {args.n_jobs}")
        
        if hasattr(args, 'timeout') and args.timeout:
            config['tuning']['timeout'] = args.timeout
            logger.info(f"튜닝 타임아웃 업데이트: {args.timeout}초")
        
        # Early stopping 관련 인자
        if hasattr(args, 'early_stopping') and args.early_stopping:
            if 'training' not in config:
                config['training'] = {}
            config['training']['early_stopping'] = True
            logger.info("Early stopping 활성화")
        
        if hasattr(args, 'early_stopping_rounds') and args.early_stopping_rounds:
            if 'training' not in config:
                config['training'] = {}
            config['training']['early_stopping_rounds'] = args.early_stopping_rounds
            logger.info(f"Early stopping 라운드 수 업데이트: {args.early_stopping_rounds}")
        
        # 피처 선택 관련 인자
        if hasattr(args, 'feature_selection') and args.feature_selection:
            if 'features' not in config:
                config['features'] = {}
            config['features']['enable_feature_selection'] = True
            logger.info("피처 선택 활성화")
        
        if hasattr(args, 'feature_selection_method') and args.feature_selection_method:
            if 'features' not in config:
                config['features'] = {}
            config['features']['feature_selection_method'] = args.feature_selection_method
            logger.info(f"피처 선택 방법 업데이트: {args.feature_selection_method}")
        
        if hasattr(args, 'feature_selection_k') and args.feature_selection_k:
            if 'features' not in config:
                config['features'] = {}
            config['features']['feature_selection_k'] = args.feature_selection_k
            logger.info(f"선택할 피처 수 업데이트: {args.feature_selection_k}")
        
        # 리샘플링 관련 인자
        if hasattr(args, 'resampling_enabled') and args.resampling_enabled:
            if 'resampling' not in config:
                config['resampling'] = {}
            config['resampling']['enabled'] = True
            logger.info("리샘플링 활성화")
        
        if hasattr(args, 'resampling_method') and args.resampling_method:
            if 'resampling' not in config:
                config['resampling'] = {}
            config['resampling']['method'] = args.resampling_method
            logger.info(f"리샘플링 방법 업데이트: {args.resampling_method}")
        
        if hasattr(args, 'resampling_ratio') and args.resampling_ratio:
            if 'resampling' not in config:
                config['resampling'] = {}
            config['resampling']['target_ratio'] = args.resampling_ratio
            logger.info(f"리샘플링 목표 비율 업데이트: {args.resampling_ratio}")
        
        # MLflow 관련 인자
        if hasattr(args, 'experiment_name') and args.experiment_name:
            if 'mlflow' not in config:
                config['mlflow'] = {}
            config['mlflow']['experiment_name'] = args.experiment_name
            logger.info(f"MLflow 실험 이름 업데이트: {args.experiment_name}")
        
        # 모델 저장 관련 인자
        if hasattr(args, 'save_model') and args.save_model:
            if 'model' not in config:
                config['model'] = {}
            config['model']['save_model'] = True
            logger.info("모델 저장 활성화")
        
        if hasattr(args, 'save_predictions') and args.save_predictions:
            if 'evaluation' not in config:
                config['evaluation'] = {}
            config['evaluation']['save_predictions'] = True
            logger.info("예측 결과 저장 활성화")
        
        # 로그 레벨 관련 인자
        if hasattr(args, 'verbose') and args.verbose is not None:
            if 'logging' not in config:
                config['logging'] = {}
            config['logging']['level'] = args.verbose
            logger.info(f"로그 레벨 업데이트: {args.verbose}")
        
        logger.info("명령행 인자 적용 완료")
        return config 