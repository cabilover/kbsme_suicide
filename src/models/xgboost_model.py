"""
XGBoost 모델 구현

이 모듈은 다중 출력 회귀 및 분류를 위한 XGBoost 모델을 구현합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import warnings
from src.utils import find_column_with_remainder
from .base_model import BaseModel
from .model_factory import register_model

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_model("xgboost")
class XGBoostModel(BaseModel):
    """
    XGBoost 기반 다중 출력 모델
    
    회귀와 분류 문제를 모두 지원하며, 각 타겟별로 별도의 모델을 학습합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        XGBoost 모델 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        # BaseModel 초기화
        super().__init__(config)
        
        # 모델 파라미터
        self.model_params = config['model']['xgboost']
        
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
            # 기존 방식: scale_pos_weight 사용
            params['scale_pos_weight'] = self.model_params.get('scale_pos_weight', 1.0)
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
        else:
            # 회귀 문제
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
        
        return params
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, 
            X_val: pd.DataFrame = None, y_val: pd.DataFrame = None) -> 'XGBoostModel':
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
        logger.info("XGBoost 모델 학습 시작")
        early_stopping_rounds = self.model_params.get('early_stopping_rounds', None)
        use_early_stopping = early_stopping_rounds is not None and X_val is not None and y_val is not None
        if use_early_stopping:
            logger.info(f"Early Stopping 활성화 (rounds: {early_stopping_rounds})")
        else:
            logger.info("Early Stopping 비활성화")
        
        # 입력 데이터 검증 및 전처리
        if y_val is not None:
            logger.info(f"[DEBUG] _validate_input_data 호출 전 y 컬럼: {list(y.columns)}")
            logger.info(f"[DEBUG] _validate_input_data 호출 전 X shape: {X.shape}")
            X, y = self._validate_input_data(X, y)
            logger.info(f"[DEBUG] _validate_input_data 호출 후 y 컬럼: {list(y.columns)}")
            logger.info(f"[DEBUG] _validate_input_data 호출 후 X shape: {X.shape}")
            X_val, y_val = self._validate_input_data(X_val, y_val)
            logger.info(f"[DEBUG] _validate_input_data 호출 후 X_val shape: {X_val.shape}")
        else:
            logger.info(f"[DEBUG] _validate_input_data 호출 전 X shape: {X.shape}")
            X = self._validate_input_data(X)
            logger.info(f"[DEBUG] _validate_input_data 호출 후 X shape: {X.shape}")
            logger.info(f"[DEBUG] _validate_input_data 호출 전 y 컬럼: {list(y.columns)}")
            y = y.select_dtypes(include=['number', 'bool', 'category'])
            y = y.replace([np.inf, -np.inf], np.nan)
            logger.info(f"[DEBUG] _validate_input_data 호출 후 y 컬럼: {list(y.columns)}")

        # 사용 가능한 타겟 컬럼 찾기 (접두사 포함)
        available_targets = self._find_available_targets(y)
        logger.info(f"사용 가능한 타겟 컬럼: {available_targets}")

        if y.shape[1] == 0:
            logger.error("y 데이터프레임에 컬럼이 없습니다! 타겟 컬럼 매칭을 확인하세요.")
            return self
        if not available_targets:
            logger.warning("사용 가능한 타겟 컬럼이 없습니다. 데이터를 확인해주세요.")
            logger.info(f"y 컬럼들: {list(y.columns)}")
            logger.info(f"찾고 있는 타겟들: {self.target_columns}")
            return self
        
        for target in available_targets:
            # pass__ 또는 remainder__ 접두사 제거하여 원본 타겟명 추출
            original_target = target
            if target.startswith('pass__'):
                original_target = target.replace('pass__', '')
            elif target.startswith('remainder__'):
                original_target = target.replace('remainder__', '')
            
            logger.info(f"타겟 {original_target} 모델 학습 중...")
            
            # 타겟 데이터 정리 (inf, nan 제거)
            y_target = y[target].replace([np.inf, -np.inf], np.nan).dropna()
            X_target = X.loc[y_target.index]
            
            # 검증 데이터 정리
            if use_early_stopping and target in y_val.columns:
                y_val_target = y_val[target].replace([np.inf, -np.inf], np.nan).dropna()
                X_val_target = X_val.loc[y_val_target.index]
            else:
                y_val_target = None
                X_val_target = None
            
            # 모델 파라미터 준비
            params = self._get_model_params(original_target)
            
            # 분류 문제인 경우 불균형 처리
            if original_target in self.classification_targets:
                # 기존 방식: scale_pos_weight 계산
                pos_weight = self._calculate_scale_pos_weight(y_target)
                params['scale_pos_weight'] = pos_weight
                logger.info(f"  - scale_pos_weight: {pos_weight:.2f}")
            
            # 실제 적용되는 파라미터 로깅
            logger.info(f"=== {original_target} 모델 파라미터 ===")
            for key, value in params.items():
                logger.info(f"  {key}: {value}")
            
            # Early Stopping 관련 파라미터를 fit에서 제외하고 모델 생성 시 사용할 파라미터에서 분리
            model_params = params.copy()
            fit_params = {}
            
            if 'early_stopping_rounds' in model_params:
                fit_params['early_stopping_rounds'] = model_params.pop('early_stopping_rounds')
            if 'verbose' in model_params:
                fit_params['verbose'] = model_params.pop('verbose')
            
            # 허용된 파라미터만 필터링하여 모델 생성
            allowed_reg_keys = xgb.XGBRegressor().get_params().keys()
            allowed_clf_keys = xgb.XGBClassifier().get_params().keys()
            
            if original_target in self.regression_targets:
                filtered_params = {k: v for k, v in model_params.items() if k in allowed_reg_keys}
                model = xgb.XGBRegressor(**filtered_params, random_state=self.config['data_split']['random_state'])
            else:
                filtered_params = {k: v for k, v in model_params.items() if k in allowed_clf_keys}
                model = xgb.XGBClassifier(**filtered_params, random_state=self.config['data_split']['random_state'])
            
            logger.info(f"model 인스턴스 타입: {type(model)}")
            
            # 모델 학습 (Early Stopping 조건부 적용)
            if (
                use_early_stopping and target in y_val.columns and
                X_val_target is not None and len(X_val_target) > 0 and
                y_val_target is not None and len(y_val_target) > 0
            ):
                eval_set = [(X_val_target, y_val_target)]
                
                model.fit(
                    X_target, y_target,
                    eval_set=eval_set,
                    early_stopping_rounds=fit_params.get('early_stopping_rounds'),
                    verbose=fit_params.get('verbose', False)
                )
            else:
                # Early Stopping 없이 학습
                logger.info("Early Stopping 없이 학습")
                
                model.fit(X_target, y_target)
            
            self.models[original_target] = model
            logger.info(f"  - {original_target} 모델 학습 완료")
        
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
        
        logger.info("XGBoost 모델 예측 시작")
        
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
        
        logger.info("XGBoost 모델 확률 예측 시작")
        
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
            
            # 피처 중요도 추출
            importance = model.feature_importances_
            
            # 전처리된 피처 이름 처리
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                # XGBoost 모델에서 피처 이름이 없는 경우
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

    def _validate_input_data(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """
        XGBoost에 최적화된 입력 데이터 검증 및 전처리
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 데이터프레임 (선택사항)
            
        Returns:
            전처리된 피처 데이터프레임 (또는 튜플)
        """
        logger.info(f"[DEBUG] XGBoost _validate_input_data 입력 X shape: {X.shape}")
        logger.info(f"[DEBUG] XGBoost _validate_input_data 입력 X 컬럼: {list(X.columns)}")
        logger.info(f"[DEBUG] XGBoost _validate_input_data 입력 X dtypes: {X.dtypes.value_counts()}")
        
        # XGBoost는 범주형 변수를 지원하지 않으므로 object 타입을 숫자로 변환
        X_cleaned = X.copy()
        
        # object 타입 컬럼들을 숫자로 변환 시도
        for col in X_cleaned.columns:
            if X_cleaned[col].dtype == 'object':
                try:
                    # 숫자로 변환 가능한지 확인
                    pd.to_numeric(X_cleaned[col], errors='raise')
                    X_cleaned[col] = pd.to_numeric(X_cleaned[col], errors='coerce')
                    logger.info(f"[DEBUG] XGBoost 컬럼 {col}을 숫자로 변환 성공")
                except (ValueError, TypeError):
                    # 숫자로 변환할 수 없는 경우 그대로 유지
                    logger.info(f"[DEBUG] XGBoost 컬럼 {col}은 숫자로 변환 불가, 그대로 유지")
        
        # 숫자형 컬럼만 선택 (XGBoost 호환)
        X_cleaned = X_cleaned.select_dtypes(include=['number', 'bool', 'category'])
        
        # 제거된 컬럼 확인
        removed_columns = set(X.columns) - set(X_cleaned.columns)
        if removed_columns:
            logger.warning(f"XGBoost에서 제거된 컬럼들: {removed_columns}")
        
        # inf 값 처리
        X_cleaned = X_cleaned.replace([np.inf, -np.inf], np.nan)
        
        if y is not None:
            # y가 Series면 DataFrame으로 변환
            if isinstance(y, pd.Series):
                y = y.to_frame()
            
            # y 데이터 처리: 타겟 컬럼은 그대로 유지하되 inf 값만 처리
            y_cleaned = y.copy()
            y_cleaned = y_cleaned.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"[DEBUG] XGBoost _validate_input_data 출력 X shape: {X_cleaned.shape}, y shape: {y_cleaned.shape}")
            logger.info(f"[DEBUG] XGBoost _validate_input_data 출력 y 컬럼: {list(y_cleaned.columns)}")
            logger.info(f"[DEBUG] XGBoost _validate_input_data 출력 X dtypes: {X_cleaned.dtypes.value_counts()}")
            return X_cleaned, y_cleaned
        
        logger.info(f"[DEBUG] XGBoost _validate_input_data 출력 X shape: {X_cleaned.shape}")
        logger.info(f"[DEBUG] XGBoost _validate_input_data 출력 X dtypes: {X_cleaned.dtypes.value_counts()}")
        return X_cleaned


def main():
    """테스트용 메인 함수"""
    # 설정 예시
    config = {
        'features': {
            'target_columns': ['suicide_a_next_year']
        },
        'model': {
            'model_type': 'xgboost',
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
            }
        }
    }
    
    # 모델 생성
    model = XGBoostModel(config)
    print(f"모델 타입: {model.model_type}")
    print(f"회귀 타겟: {model.regression_targets}")
    print(f"분류 타겟: {model.classification_targets}")


if __name__ == "__main__":
    main() 