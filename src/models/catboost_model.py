"""
CatBoost 모델 구현

이 모듈은 다중 출력 회귀 및 분류를 위한 CatBoost 모델을 구현합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import catboost as cb
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import warnings
from src.utils import find_column_with_remainder
from .base_model import BaseModel
from .model_factory import register_model

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_model("catboost")
class CatBoostModel(BaseModel):
    """
    CatBoost 기반 다중 출력 모델
    
    회귀와 분류 문제를 모두 지원하며, 각 타겟별로 별도의 모델을 학습합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        CatBoost 모델 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        # BaseModel 초기화
        super().__init__(config)
        
        # 모델 파라미터
        self.model_params = config['model']['catboost']
        
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
        
        # 분류 문제인 경우
        if target in self.classification_targets:
            # 기존 방식: class_weights 사용
            params['class_weights'] = [1.0, self.model_params.get('scale_pos_weight', 1.0)]
            params['loss_function'] = 'Logloss'
            params['eval_metric'] = 'Logloss'
        else:
            # 회귀 문제
            params['loss_function'] = 'RMSE'
            params['eval_metric'] = 'RMSE'
        
        return params
    
    def _calculate_class_weights(self, y_target: pd.Series) -> List[float]:
        """
        Focal Loss 설정을 기반으로 클래스 가중치를 계산합니다.
        
        Args:
            y_target: 타겟 시리즈
            
        Returns:
            클래스별 가중치 리스트 [negative_weight, positive_weight]
        """
        # 기존 방식: 비율 기반 가중치
        neg_count = (y_target == 0).sum()
        pos_count = (y_target == 1).sum()
        if pos_count > 0:
            return [1, neg_count / pos_count]
        else:
            return [1, 1]
    
    def _calculate_focal_loss_metric(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Focal Loss를 사용한 평가 지표를 계산합니다.
        
        Args:
            y_true: 실제 값
            y_pred_proba: 예측 확률
            
        Returns:
            Focal Loss 값
        """
        focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        return focal_loss(y_true, y_pred_proba)
    
    def _identify_categorical_features(self, X: pd.DataFrame) -> List[int]:
        """
        범주형 피처의 인덱스를 식별합니다.
        
        Args:
            X: 피처 데이터프레임
            
        Returns:
            범주형 피처 인덱스 리스트
        """
        categorical_features = []
        
        for i, col in enumerate(X.columns):
            # object 타입이거나 카디널리티가 낮은 수치형 변수
            if X[col].dtype == 'object' or X[col].dtype == 'category':
                categorical_features.append(i)
            elif X[col].dtype in ['int64', 'int32']:
                # 정수형 변수 중 카디널리티가 낮은 경우 범주형으로 처리
                unique_ratio = X[col].nunique() / len(X[col])
                if unique_ratio < 0.1:  # 10% 미만의 고유값 비율
                    categorical_features.append(i)
        
        return categorical_features
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, 
            X_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> 'CatBoostModel':
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
        logger.info("CatBoost 모델 학습 시작")
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
            y = y.select_dtypes(include=['number', 'bool', 'category'])
            y = y.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"전처리 후 데이터 형태: X={X.shape}, y={y.shape}")
        
        # 범주형 피처 식별
        categorical_features = self._identify_categorical_features(X)
        logger.info(f"범주형 피처 인덱스: {categorical_features}")
        logger.info(f"범주형 피처 개수: {len(categorical_features)}")
        
        # 사용 가능한 타겟 컬럼 찾기
        available_targets = []
        for target in self.target_columns:
            if target in y.columns:
                available_targets.append(target)
        logger.info(f"사용 가능한 타겟 컬럼: {available_targets}")
        
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
            
            # 클래스 분포 확인 (분류 문제인 경우)
            if target in self.classification_targets:
                class_counts = y_target.value_counts()
                logger.info(f"클래스 분포: {dict(class_counts)}")
                logger.info(f"불균형 비율: {class_counts.min() / class_counts.max():.3f}")
            
            # 모델 파라미터 준비
            params = self._get_model_params(target)
            
            # 실제 적용되는 파라미터 로깅
            logger.info(f"=== {target} 모델 파라미터 ===")
            for key, value in params.items():
                logger.info(f"  {key}: {value}")
            
            # random_seed 중복 제거
            if 'random_seed' in params:
                random_seed = params.pop('random_seed')
            else:
                random_seed = self.config.get('data_split', {}).get('random_state', 42)
            
            # CatBoost 모델 생성
            logger.info(f"CatBoost 모델 생성 중... (타겟: {target})")
            if target in self.regression_targets:
                logger.info("회귀 모델 (CatBoostRegressor) 사용")
                model = CatBoostRegressor(
                    **params,
                    random_seed=random_seed,
                    verbose=False
                )
            else:
                # 분류 문제인 경우 클래스 가중치 설정
                logger.info("분류 모델 (CatBoostClassifier) 사용")
                model = CatBoostClassifier(
                    **params,
                    random_seed=random_seed,
                    verbose=False
                )
            
            # 모델 학습
            logger.info(f"CatBoost 모델 학습 시작 (타겟: {target})")
            if use_early_stopping and X_val_target is not None and len(X_val_target) > 0:
                logger.info("Early Stopping과 함께 학습")
                model.fit(
                    X_target, y_target,
                    eval_set=(X_val_target, y_val_target),
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            else:
                logger.info("Early Stopping 없이 학습")
                model.fit(X_target, y_target, verbose=False)
            
            # 모델 정보 로깅
            logger.info(f"타겟 {target} 모델 학습 완료")
            if hasattr(model, 'best_iteration_'):
                logger.info(f"  - Best Iteration: {model.best_iteration_}")
            if hasattr(model, 'best_score_'):
                logger.info(f"  - Best Score: {model.best_score_}")
            if hasattr(model, 'tree_count_'):
                logger.info(f"  - 트리 개수: {model.tree_count_}")
            
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
        
        logger.info("CatBoost 모델 예측 시작")
        
        # 입력 데이터 검증 및 전처리
        X = self._validate_input_data(X)
        
        # 범주형 피처 식별
        categorical_features = self._identify_categorical_features(X)
        
        predictions = {}
        
        for target in self.target_columns:
            if target not in self.models:
                logger.warning(f"타겟 {target}에 대한 모델이 없습니다. 건너뜁니다.")
                continue
            
            model = self.models[target]
            
            # 예측 수행
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = model.predict(X, thread_count=-1)
            
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
        
        logger.info("CatBoost 모델 확률 예측 시작")
        
        # 입력 데이터 검증 및 전처리
        X = self._validate_input_data(X)
        
        # 범주형 피처 식별
        categorical_features = self._identify_categorical_features(X)
        
        proba_predictions = {}
        
        for target in self.classification_targets:
            if target not in self.models:
                continue
            
            model = self.models[target]
            
            # 확률 예측 수행
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba = model.predict_proba(X, thread_count=-1)
            
            proba_predictions[target] = proba
            logger.info(f"  - {target} 확률 예측 완료")
        
        return proba_predictions
    
    def get_feature_importance(self, target: str = None, aggregate: bool = False, 
                             importance_type: str = 'PredictionValuesChange') -> Dict[str, pd.DataFrame]:
        """
        피처 중요도를 반환합니다.
        
        Args:
            target: 특정 타겟 (None이면 모든 타겟)
            aggregate: 모든 타겟의 피처 중요도를 집계할지 여부
            importance_type: 중요도 타입 ('PredictionValuesChange', 'LossFunctionChange', 'ShapValues')
            
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
            
            # CatBoost 피처 중요도 추출
            importance = model.get_feature_importance(type=importance_type)
            feature_names = model.feature_names_
            
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
    
    def get_permutation_importance(self, X: pd.DataFrame, y: pd.DataFrame, 
                                  target: str = None, n_repeats: int = 5) -> Dict[str, pd.DataFrame]:
        """
        순열 중요도를 계산합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 데이터프레임
            target: 특정 타겟 (None이면 모든 타겟)
            n_repeats: 순열 반복 횟수
            
        Returns:
            타겟별 순열 중요도 데이터프레임
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        from sklearn.inspection import permutation_importance
        
        permutation_dict = {}
        
        # 분석할 타겟 결정
        if target is not None:
            if target not in self.target_columns:
                raise ValueError(f"타겟 {target}이 모델에 존재하지 않습니다. 사용 가능한 타겟: {self.target_columns}")
            targets_to_analyze = [target]
        else:
            targets_to_analyze = self.target_columns
        
        # 입력 데이터 검증 및 전처리
        X = self._validate_input_data(X)
        y = y.select_dtypes(include=['number', 'bool', 'category'])
        y = y.replace([np.inf, -np.inf], np.nan)
        
        for t in targets_to_analyze:
            if t not in self.models:
                logger.warning(f"타겟 {t}에 대한 모델이 없습니다. 건너뜁니다.")
                continue
            
            model = self.models[t]
            
            # 타겟 데이터 정리
            y_target = y[t].replace([np.inf, -np.inf], np.nan).dropna()
            X_target = X.loc[y_target.index]
            
            # 순열 중요도 계산
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                perm_importance = permutation_importance(
                    model, X_target, y_target, 
                    n_repeats=n_repeats, 
                    random_state=self.config['data_split']['random_state'],
                    n_jobs=-1
                )
            
            # 데이터프레임으로 변환
            perm_df = pd.DataFrame({
                'feature': X_target.columns,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            permutation_dict[t] = perm_df
        
        return permutation_dict


def main():
    """테스트용 메인 함수"""
    # 설정 예시
    config = {
        'features': {
            'target_columns': ['suicide_a_next_year']
        },
        'model': {
            'model_type': 'catboost',
            'catboost': {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3,
                'border_count': 254,
                'bagging_temperature': 1,
                'random_strength': 1,
                'early_stopping_rounds': 10
            }
        }
    }
    
    # 모델 생성
    model = CatBoostModel(config)
    print(f"모델 타입: {model.model_type}")
    print(f"회귀 타겟: {model.regression_targets}")
    print(f"분류 타겟: {model.classification_targets}")


if __name__ == "__main__":
    main() 