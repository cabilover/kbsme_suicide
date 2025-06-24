"""
Random Forest 모델 구현

이 모듈은 다중 출력 회귀 및 분류를 위한 Random Forest 모델을 구현합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import warnings
from src.utils import find_column_with_remainder
from .base_model import BaseModel
from .model_factory import register_model
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_model("random_forest")
class RandomForestModel(BaseModel):
    """
    Random Forest 기반 다중 출력 모델
    
    회귀와 분류 문제를 모두 지원하며, 각 타겟별로 별도의 모델을 학습합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Random Forest 모델 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        # BaseModel 초기화
        super().__init__(config)
        
        # 모델 파라미터
        self.model_params = config['model']['random_forest']
        
        logger.info(f"  - Focal Loss 사용: False")
    
    def _calculate_class_weights(self, y_target: pd.Series) -> Dict[int, float]:
        """
        클래스 가중치를 계산합니다.
        
        Args:
            y_target: 타겟 시리즈
            
        Returns:
            클래스별 가중치 딕셔너리
        """
        # 기존 방식: 비율 기반 가중치
        neg_count = (y_target == 0).sum()
        pos_count = (y_target == 1).sum()
        if pos_count > 0:
            return {0: 1.0, 1: neg_count / pos_count}
        else:
            return {0: 1.0, 1: 1.0}
    
    def _strip_prefix(self, col):
        for prefix in ['pass__', 'num__', 'cat__']:
            if col.startswith(prefix):
                return col[len(prefix):]
        return col

    def _find_available_targets(self, y):
        # 접두사 제거 후 비교
        y_cols = list(y.columns)
        clean_y_cols = [self._strip_prefix(col) for col in y_cols]
        clean_class_targets = [self._strip_prefix(t) for t in self.classification_targets]
        return [col for col, clean_col in zip(y_cols, clean_y_cols) if clean_col in clean_class_targets]

    def _get_model_params(self, target: str) -> dict:
        params = self.model_params.copy()
        # 접두사 제거 후 비교
        clean_target = self._strip_prefix(target)
        clean_class_targets = [self._strip_prefix(t) for t in self.classification_targets]
        
        # 공통 파라미터 설정
        params['n_jobs'] = self.model_params.get('n_jobs', 28)
        
        if clean_target in clean_class_targets:
            # 분류 모델용 파라미터
            params['class_weight'] = self.model_params.get('class_weight', 'balanced')
            params['criterion'] = self.model_params.get('criterion', 'gini')
            
            # 분류 모델에서 지원하지 않는 파라미터 제거
            params.pop('max_features_regression', None)
            params.pop('min_impurity_decrease_regression', None)
            params.pop('criterion_regression', None)
        else:
            # 회귀 모델용 파라미터
            params['criterion'] = self.model_params.get('criterion', 'mse')
            
            # 회귀 모델에서 지원하지 않는 파라미터 제거
            params.pop('class_weight', None)
            params.pop('criterion_classification', None)
        
        # 공통적으로 제거할 파라미터들
        params.pop('sample_weight', None)  # fit()에서만 사용
        params.pop('use_focal_loss', None)
        params.pop('focal_loss', None)
        
        return params
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, 
            X_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> 'RandomForestModel':
        """
        모델을 학습합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 데이터프레임
            X_val: 검증 피처 데이터프레임 (사용하지 않음, 호환성을 위해 유지)
            y_val: 검증 타겟 데이터프레임 (사용하지 않음, 호환성을 위해 유지)
            
        Returns:
            학습된 모델
        """
        logger.info("Random Forest 모델 학습 시작")
        logger.info(f"입력 데이터 형태: X={X.shape}, y={y.shape}")
        
        # 입력 데이터 검증 및 전처리
        logger.info("입력 데이터 검증 및 전처리 중...")
        if y_val is not None:
            X, y = self._validate_input_data(X, y)
            X_val, y_val = self._validate_input_data(X_val, y_val)
        else:
            X = self._validate_input_data(X)
            y = y.select_dtypes(include=['number', 'bool', 'category'])
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
            
            # 클래스 분포 확인 (분류 문제인 경우)
            if target in self.classification_targets:
                class_counts = y_target.value_counts()
                logger.info(f"클래스 분포: {dict(class_counts)}")
                logger.info(f"불균형 비율: {class_counts.min() / class_counts.max():.3f}")
            
            # 모델 파라미터 준비
            params = self._get_model_params(target)
            # sample_weight 분리
            sample_weight = None
            if 'sample_weight' in params:
                sample_weight = params.pop('sample_weight')

            # random_state 중복 제거
            if 'random_state' in params:
                random_state = params.pop('random_state')
            else:
                random_state = self.config.get('data_split', {}).get('random_state', 42)
            
            # 실제 적용되는 파라미터 로깅
            logger.info(f"=== {target} 모델 파라미터 ===")
            for key, value in params.items():
                logger.info(f"  {key}: {value}")
            if sample_weight is not None:
                logger.info(f"  sample_weight: {sample_weight}")
            
            # Random Forest 모델 생성
            logger.info(f"Random Forest 모델 생성 중... (타겟: {target})")
            clean_target = self._strip_prefix(target)
            clean_class_targets = [self._strip_prefix(t) for t in self.classification_targets]
            if clean_target in clean_class_targets:
                logger.info("분류 모델 (RandomForestClassifier) 사용")
                model = RandomForestClassifier(
                    **params,
                    random_state=random_state
                )
            else:
                logger.info("회귀 모델 (RandomForestRegressor) 사용")
                model = RandomForestRegressor(
                    **params,
                    random_state=random_state
                )
            
            # 모델 학습
            logger.info(f"Random Forest 모델 학습 시작 (타겟: {target})")
            if sample_weight is not None:
                model.fit(X_target, y_target, sample_weight=sample_weight)
            else:
                model.fit(X_target, y_target)
            
            # 모델 정보 로깅
            logger.info(f"타겟 {target} 모델 학습 완료")
            if hasattr(model, 'n_estimators'):
                logger.info(f"  - 트리 개수: {model.n_estimators}")
            if hasattr(model, 'n_features_in_'):
                logger.info(f"  - 피처 개수: {model.n_features_in_}")
            if hasattr(model, 'classes_') and model.classes_ is not None:
                logger.info(f"  - 클래스: {model.classes_}")
            
            self.models[target] = model
        
        self.is_fitted = True
        logger.info(f"모든 모델 학습 완료 ({len(self.models)}개)")
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
        
        logger.info("Random Forest 모델 예측 시작")
        
        # 입력 데이터 검증 및 전처리
        X = self._validate_input_data(X)
        
        predictions = {}
        
        for target in self.target_columns:
            if target not in self.models:
                logger.warning(f"타겟 {target}에 대한 모델이 없습니다. 건너뜁니다.")
                continue
            
            model = self.models[target]
            
            # 예측 수행
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = model.predict(X)
            
            predictions[target] = pred
            logger.info(f"  - {target} 예측 완료")
        
        # 데이터프레임으로 변환
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
        
        logger.info("Random Forest 모델 확률 예측 시작")
        
        # 입력 데이터 검증 및 전처리
        X = self._validate_input_data(X)
        
        proba_predictions = {}
        
        for target in self.classification_targets:
            if target not in self.models:
                continue
            
            model = self.models[target]
            
            # 확률 예측 수행
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba = model.predict_proba(X)
            
            proba_predictions[target] = proba
            logger.info(f"  - {target} 확률 예측 완료")
        
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
            
            # Random Forest 피처 중요도 추출
            importance = model.feature_importances_
            
            # 피처 이름 처리
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                # 피처 이름이 없는 경우
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
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
    
    def get_model_interpretability_info(self) -> Dict[str, Any]:
        """
        모델 해석 가능성 정보를 반환합니다.
        
        Returns:
            모델 해석 가능성 정보
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        interpretability_info = {
            'model_type': 'Random Forest',
            'advantages': [
                '피처 중요도 제공',
                '과적합에 강함',
                '해석 가능성 높음',
                '불균형 데이터 처리 가능',
                '범주형 변수 자동 처리'
            ],
            'feature_importance_available': True,
            'partial_dependence_available': True,
            'shap_values_available': True,
            'num_models': len(self.models)
        }
        
        return interpretability_info


def main():
    """테스트용 메인 함수"""
    # 설정 예시
    config = {
        'features': {
            'target_columns': ['suicide_a_next_year']
        },
        'model': {
            'model_type': 'random_forest',
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True
            }
        }
    }
    
    # 모델 생성
    model = RandomForestModel(config)
    print(f"모델 타입: {model.model_type}")
    print(f"회귀 타겟: {model.regression_targets}")
    print(f"분류 타겟: {model.classification_targets}")


if __name__ == "__main__":
    main() 