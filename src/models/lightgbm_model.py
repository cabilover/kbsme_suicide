"""
LightGBM 모델 구현

이 모듈은 다중 출력 회귀 및 분류를 위한 LightGBM 모델을 구현합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import warnings
from src.utils import find_column_with_remainder
from .base_model import BaseModel
from .model_factory import register_model
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_model("lightgbm")
class LightGBMModel(BaseModel):
    """
    LightGBM 기반 다중 출력 모델
    
    회귀와 분류 문제를 모두 지원하며, 각 타겟별로 별도의 모델을 학습합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        LightGBM 모델 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        # BaseModel 초기화
        super().__init__(config)
        
        # 모델 파라미터 (설정에서만 가져옴, 코드 내 하드코딩 금지)
        self.model_params = config['model']['lightgbm']
        logger.info(f"  - Focal Loss 사용: False")
    
    def _get_model_params(self, target: str) -> Dict[str, Any]:
        """
        특정 타겟에 대한 모델 파라미터를 반환합니다.
        
        Args:
            target: 타겟 컬럼명
            
        Returns:
            모델 파라미터 딕셔너리
        """
        params = self.model_params.copy()
        params.pop('focal_loss', None)  # LightGBM은 focal_loss를 지원하지 않음
        params.pop('use_focal_loss', None)  # 내부 플래그도 제거
        # 병렬 파라미터 추가
        params['num_threads'] = self.model_params.get('num_threads', 4)

        # 타겟 접두사 제거
        clean_target = target
        for prefix in ['pass__', 'num__', 'cat__']:
            if target.startswith(prefix):
                clean_target = target[len(prefix):]
                break
        # classification_targets도 접두사 제거해서 비교
        clean_classification_targets = [
            t[len(prefix):] if any(t.startswith(prefix) for prefix in ['pass__', 'num__', 'cat__']) else t
            for t in self.classification_targets
        ]
        
        # 분류 문제인 경우
        if clean_target in clean_classification_targets:
            # 반드시 config에서만 파라미터를 가져옴 (코드 내 하드코딩 금지)
            scale_pos_weight = self.model_params.get('scale_pos_weight', 1.0)
            is_unbalance = self.model_params.get('is_unbalance', False)
            class_weight = self.model_params.get('class_weight', None)
            # 파라미터 설정 (is_unbalance와 scale_pos_weight 충돌 방지)
            if scale_pos_weight != 1.0 and scale_pos_weight is not None and scale_pos_weight != "auto":
                params['scale_pos_weight'] = scale_pos_weight
                params.pop('is_unbalance', None)  # scale_pos_weight 우선 시 is_unbalance 제거
                logger.info(f"  - scale_pos_weight: {scale_pos_weight}")
                logger.info(f"  - is_unbalance: False (scale_pos_weight와 충돌 방지)")
            elif is_unbalance:
                params['is_unbalance'] = is_unbalance
                params.pop('scale_pos_weight', None)  # is_unbalance 우선 시 scale_pos_weight 제거
                logger.info(f"  - is_unbalance: {is_unbalance}")
                logger.info(f"  - scale_pos_weight: 1.0 (is_unbalance와 충돌 방지)")
            else:
                params.pop('is_unbalance', None)
                params.pop('scale_pos_weight', None)
                logger.info(f"  - scale_pos_weight: 1.0 (기본값)")
                logger.info(f"  - is_unbalance: False (기본값)")
            if class_weight and class_weight != "None":
                params['class_weight'] = class_weight
                logger.info(f"  - class_weight: {class_weight}")
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            # 설정된 파라미터 요약
            logger.info(f"=== {target} 클래스 불균형 처리 설정 ===")
            logger.info(f"  - scale_pos_weight: {scale_pos_weight}")
            logger.info(f"  - is_unbalance: {is_unbalance}")
            logger.info(f"  - class_weight: {class_weight}")
        else:
            # 회귀 문제
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        
        return params
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, 
            X_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> 'LightGBMModel':
        """
        모델을 학습합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 데이터프레임
            X_val: 검증 피처 데이터프레임 (Early Stopping용, 선택사항)
            y_val: 검증 타겟 데이터프레임 (Early Stopping용, 선택사항)
            
        Returns:
            학습된 모델
        """
        logger.info("LightGBM 모델 학습 시작")
        logger.info(f"입력 데이터 형태: X={X.shape}, y={y.shape}")
        
        # Early Stopping 설정
        early_stopping_rounds = self.model_params.get('early_stopping_rounds', None)
        use_early_stopping = early_stopping_rounds is not None and X_val is not None and y_val is not None
        if use_early_stopping:
            logger.info(f"Early Stopping 활성화 (rounds: {early_stopping_rounds})")
            logger.info(f"검증 데이터 형태: X_val={X_val.shape}, y_val={y_val.shape}")
        else:
            logger.info("Early Stopping 비활성화")
        
        # 입력 데이터 검증 및 전처리
        logger.info("입력 데이터 검증 및 전처리 중...")
        if y_val is not None:
            X, y = self._validate_input_data(X, y)
            X_val, y_val = self._validate_input_data(X_val, y_val)
        else:
            X = self._validate_input_data(X)
            # y 데이터는 타겟이므로 컬럼을 제거하지 않음 (중요!)
            if isinstance(y, pd.Series):
                y = y.to_frame()
            y = y.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"전처리 후 데이터 형태: X={X.shape}, y={y.shape}")
        
        # 사용 가능한 타겟 컬럼 찾기 (접두사 포함)
        available_targets = self._find_available_targets(y)
        logger.info(f"사용 가능한 타겟 컬럼: {available_targets}")

        if y.shape[1] == 0:
            logger.error("y 데이터프레임에 컬럼이 없습니다! 타겟 컬럼 매칭을 확인하세요.")
            return self
        if not available_targets:
            logger.error("사용 가능한 타겟 컬럼이 없습니다!")
            return self
        
        for target in available_targets:
            logger.info(f"=== 타겟 {target} 모델 학습 시작 ===")
            
            # 타겟 데이터 정리 (inf, nan 제거)
            y_target = y[target].replace([np.inf, -np.inf], np.nan).dropna()
            X_target = X.loc[y_target.index]
            
            logger.info(f"타겟 {target} 데이터 정리 후: X={X_target.shape}, y={len(y_target)}")
            
            # 검증 데이터 정리
            if use_early_stopping and target in y_val.columns:
                y_val_target = y_val[target].replace([np.inf, -np.inf], np.nan).dropna()
                X_val_target = X_val.loc[y_val_target.index]
                logger.info(f"검증 데이터 정리 후: X_val={X_val_target.shape}, y_val={len(y_val_target)}")
            else:
                y_val_target = None
                X_val_target = None
                logger.info("검증 데이터 없음")
            
            # 모델 파라미터 준비
            params = self._get_model_params(target)
            
            # scale_pos_weight가 "auto"인 경우 자동 계산
            if params.get('scale_pos_weight') == "auto":
                # 양성 클래스와 음성 클래스 비율 계산
                positive_count = (y_target == 1).sum()
                negative_count = (y_target == 0).sum()
                if positive_count > 0 and negative_count > 0:
                    auto_scale_pos_weight = negative_count / positive_count
                    params['scale_pos_weight'] = auto_scale_pos_weight
                    logger.info(f"  - scale_pos_weight 자동 계산: {auto_scale_pos_weight:.2f} (음성:양성 = {negative_count}:{positive_count})")
                else:
                    logger.warning(f"  - scale_pos_weight 자동 계산 실패: 양성 또는 음성 클래스가 없음")
                    params['scale_pos_weight'] = 1.0
            
            # 실제 적용되는 파라미터 로깅
            logger.info(f"=== {target} 모델 파라미터 ===")
            for key, value in params.items():
                logger.info(f"  {key}: {value}")
            
            # LightGBM 데이터셋 생성
            logger.info("LightGBM 데이터셋 생성 중...")
            train_data = lgb.Dataset(X_target, label=y_target)
            
            # 검증 데이터셋 생성 (Early Stopping용)
            valid_data = None
            if use_early_stopping and X_val_target is not None and len(X_val_target) > 0:
                valid_data = lgb.Dataset(X_val_target, label=y_val_target, reference=train_data)
                logger.info("검증 데이터셋 생성 완료")
            
            # 모델 학습
            logger.info(f"LightGBM 모델 학습 시작 (타겟: {target})")
            if use_early_stopping and valid_data is not None:
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[valid_data],
                    valid_names=['valid'],
                    callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
                )
            else:
                # Early Stopping 없이 학습
                logger.info("Early Stopping 없이 학습")
                model = lgb.train(
                    params,
                    train_data,
                    callbacks=[lgb.log_evaluation(0)]
                )
            
            # 모델 저장
            self.models[target] = model
            logger.info(f"타겟 {target} 모델 학습 완료")
            
            # 모델 정보 로깅
            if hasattr(model, 'best_score'):
                logger.info(f"  - Best Score: {model.best_score}")
            if hasattr(model, 'best_iteration'):
                logger.info(f"  - Best Iteration: {model.best_iteration}")
        
        logger.info("LightGBM 모델 학습 완료")
        self.is_fitted = True  # 모델 학습 완료 표시
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        예측을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            
        Returns:
            예측 결과 데이터프레임
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        logger.info("LightGBM 모델 예측 시작")
        
        # 입력 데이터 검증 및 전처리
        X = self._validate_input_data(X)
        
        predictions = {}
        
        # 실제 학습된 모델의 키를 사용
        for target in self.models.keys():
            model = self.models[target]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = model.predict(X)
                predictions[target] = pred
                logger.info(f"  - {target} 예측 완료")
            except Exception as e:
                logger.warning(f"{target} 예측 중 예외 발생: {e}")
        
        result_df = pd.DataFrame(predictions, index=X.index)
        logger.info(f"예측 완료: {result_df.shape}")
        return result_df
    
    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        분류 문제에서 확률 예측을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            
        Returns:
            타겟별 확률 예측 결과
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        logger.info("LightGBM 모델 확률 예측 시작")
        
        # 입력 데이터 검증 및 전처리
        X = self._validate_input_data(X)
        
        proba_predictions = {}
        
        # 실제 학습된 모델의 키를 사용 (분류 모델만)
        for target in self.models.keys():
            if target not in self.classification_targets:
                continue
            model = self.models[target]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    proba = model.predict(X)
                    proba_reshaped = np.column_stack([1 - proba, proba])
                proba_predictions[target] = proba_reshaped
                logger.info(f"  - {target} 확률 예측 완료")
            except Exception as e:
                logger.warning(f"{target} 확률 예측 중 예외 발생: {e}")
        return proba_predictions
    
    def get_feature_importance(self, target: str = None, aggregate: bool = False) -> Dict[str, pd.DataFrame]:
        """
        피처 중요도를 반환합니다.
        
        Args:
            target: 특정 타겟 (None이면 모든 타겟)
            aggregate: 모든 타겟의 피처 중요도를 집계할지 여부
            
        Returns:
            타겟별 피처 중요도 데이터프레임 또는 집계된 피처 중요도
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        importance_dict = {}
        
        # 분석할 타겟 결정
        if target is not None:
            if target not in self.target_columns:
                raise ValueError(f"타겟 {target}이 모델에 존재하지 않습니다. 사용 가능한 타겟: {self.target_columns}")
            targets_to_analyze = [target]
        else:
            targets_to_analyze = self.target_columns
        
        # 각 타겟별 피처 중요도 추출
        for t in targets_to_analyze:
            if t not in self.models:
                logger.warning(f"타겟 {t}에 대한 모델이 없습니다. 건너뜁니다.")
                continue
            
            model = self.models[t]
            
            # LightGBM 피처 중요도 추출
            importance = model.feature_importance(importance_type='gain')
            feature_names = model.feature_name()
            
            # 데이터프레임으로 변환
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            importance_dict[t] = importance_df
        
        # 집계 요청이 있고 여러 타겟이 있는 경우
        if aggregate and len(importance_dict) > 1:
            return self._aggregate_feature_importance(importance_dict)
        
        return importance_dict
    
    def _aggregate_feature_importance(self, importance_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        여러 타겟의 피처 중요도를 집계합니다.
        
        Args:
            importance_dict: 타겟별 피처 중요도 딕셔너리
            
        Returns:
            집계된 피처 중요도
        """
        # 모든 피처 이름 수집
        all_features = set()
        for df in importance_dict.values():
            all_features.update(df['feature'].tolist())
        
        # 피처별 평균 중요도 계산
        feature_avg_importance = {}
        feature_std_importance = {}
        
        for feature in all_features:
            importances = []
            for df in importance_dict.values():
                feature_importance = df[df['feature'] == feature]['importance']
                if not feature_importance.empty:
                    importances.append(feature_importance.iloc[0])
            
            if importances:
                feature_avg_importance[feature] = np.mean(importances)
                feature_std_importance[feature] = np.std(importances)
            else:
                feature_avg_importance[feature] = 0.0
                feature_std_importance[feature] = 0.0
        
        # 집계된 데이터프레임 생성
        aggregated_df = pd.DataFrame({
            'feature': list(all_features),
            'mean_importance': [feature_avg_importance[f] for f in all_features],
            'std_importance': [feature_std_importance[f] for f in all_features]
        }).sort_values('mean_importance', ascending=False)
        
        return {'aggregated': aggregated_df}

    def _validate_input_data(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """
        LightGBM에 최적화된 입력 데이터 검증 및 전처리
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 데이터프레임 (선택사항)
            
        Returns:
            전처리된 피처 데이터프레임 (또는 튜플)
        """
        logger.info(f"[DEBUG] LightGBM _validate_input_data 입력 X shape: {X.shape}")
        logger.info(f"[DEBUG] LightGBM _validate_input_data 입력 X 컬럼: {list(X.columns)}")
        logger.info(f"[DEBUG] LightGBM _validate_input_data 입력 X dtypes: {X.dtypes.value_counts()}")
        
        # LightGBM은 범주형 변수를 지원하므로 object 타입을 그대로 유지
        X_cleaned = X.copy()
        
        # inf 값만 처리 (범주형 변수는 보존)
        X_cleaned = X_cleaned.replace([np.inf, -np.inf], np.nan)
        
        if y is not None:
            # y가 Series면 DataFrame으로 변환
            if isinstance(y, pd.Series):
                y = y.to_frame()
            
            # y 데이터 처리: 타겟 컬럼은 그대로 유지하되 inf 값만 처리
            y_cleaned = y.copy()
            y_cleaned = y_cleaned.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"[DEBUG] LightGBM _validate_input_data 출력 X shape: {X_cleaned.shape}, y shape: {y_cleaned.shape}")
            logger.info(f"[DEBUG] LightGBM _validate_input_data 출력 y 컬럼: {list(y_cleaned.columns)}")
            logger.info(f"[DEBUG] LightGBM _validate_input_data 출력 X dtypes: {X_cleaned.dtypes.value_counts()}")
            return X_cleaned, y_cleaned
        
        logger.info(f"[DEBUG] LightGBM _validate_input_data 출력 X shape: {X_cleaned.shape}")
        logger.info(f"[DEBUG] LightGBM _validate_input_data 출력 X dtypes: {X_cleaned.dtypes.value_counts()}")
        return X_cleaned


def main():
    """테스트용 메인 함수"""
    # 설정 파일 로드
    config_path = Path("configs/base/common.yaml")
    
    if not config_path.exists():
        logger.error("설정 파일을 찾을 수 없습니다!")
        logger.error("해결 방법:")
        logger.error("1. configs/base/common.yaml 파일이 존재하는지 확인하세요")
        logger.error("2. 파일이 없다면 다음 구조로 생성하세요:")
        logger.error("""
features:
  target_variables:
    original_targets:
      score_targets: ["anxiety_score", "depress_score", "sleep_score", "comp"]
      binary_targets: ["suicide_t", "suicide_a"]
    next_year_targets:
      score_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
      binary_targets: ["suicide_t_next_year", "suicide_a_next_year"]
  
  target_types:
    regression_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
    classification_targets: ["suicide_t_next_year", "suicide_a_next_year"]
  
  target_columns: ["suicide_a_next_year"]
        """)
        raise SystemExit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            import yaml
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {e}")
        logger.error("해결 방법:")
        logger.error("1. configs/base/common.yaml 파일의 YAML 문법을 확인하세요")
        logger.error("2. 들여쓰기가 올바른지 확인하세요 (탭 대신 스페이스 사용)")
        logger.error("3. 콜론(:) 뒤에 공백이 있는지 확인하세요")
        raise SystemExit(1)
    
    # 필수 설정 검증
    required_sections = ['features']
    for section in required_sections:
        if section not in config:
            logger.error(f"설정 파일에 필수 섹션 '{section}'이 없습니다!")
            logger.error("해결 방법:")
            logger.error(f"configs/base/common.yaml 파일에 '{section}' 섹션을 추가하세요")
            raise SystemExit(1)
    
    features_config = config.get('features', {})
    required_feature_sections = ['target_columns', 'target_variables', 'target_types']
    missing_sections = [section for section in required_feature_sections if section not in features_config]
    
    if missing_sections:
        logger.error(f"설정 파일에 필수 피처 섹션이 없습니다: {missing_sections}")
        logger.error("해결 방법:")
        logger.error("configs/base/common.yaml 파일의 features 섹션에 다음을 추가하세요:")
        logger.error("""
  target_columns: ["suicide_a_next_year"]
  target_variables:
    original_targets:
      score_targets: ["anxiety_score", "depress_score", "sleep_score", "comp"]
      binary_targets: ["suicide_t", "suicide_a"]
    next_year_targets:
      score_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
      binary_targets: ["suicide_t_next_year", "suicide_a_next_year"]
  
  target_types:
    regression_targets: ["anxiety_score_next_year", "depress_score_next_year", "sleep_score_next_year"]
    classification_targets: ["suicide_t_next_year", "suicide_a_next_year"]
        """)
        raise SystemExit(1)
    
    # 모델 설정 추가
    config['model'] = {
        'model_type': 'lightgbm',
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
        }
    }
    
    # 모델 생성
    model = LightGBMModel(config)
    print(f"모델 타입: {model.model_type}")
    print(f"회귀 타겟: {model.regression_targets}")
    print(f"분류 타겟: {model.classification_targets}")
    print("✅ 설정 파일 검증 완료 - 모델이 정상적으로 초기화되었습니다.")


if __name__ == "__main__":
    main() 