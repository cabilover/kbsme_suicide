"""
LightGBM 모델 구현

이 모듈은 다중 출력 회귀 및 분류를 위한 LightGBM 모델을 구현합니다.
빠른 학습 속도와 높은 성능, 범주형 변수 자동 처리, 메모리 효율성을 특징으로 합니다.
Focal Loss를 통한 불균형 데이터 처리도 지원합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import warnings
from .base_model import BaseModel
from .model_factory import register_model
from .loss_functions import FocalLoss, validate_focal_loss_parameters

# LightGBM 임포트 (설치되지 않은 경우를 대비한 예외 처리)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_model("lightgbm")
class LightGBMModel(BaseModel):
    """
    LightGBM 기반 다중 출력 모델
    
    회귀와 분류 문제를 모두 지원하며, 각 타겟별로 별도의 모델을 학습합니다.
    빠른 학습 속도와 높은 성능, 범주형 변수 자동 처리를 특징으로 합니다.
    Focal Loss를 통한 불균형 데이터 처리도 지원합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        LightGBM 모델 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM이 설치되지 않았습니다. "
                "설치하려면: pip install lightgbm"
            )
        
        # BaseModel 초기화
        super().__init__(config)
        
        # Focal Loss 설정 확인
        self.use_focal_loss = config.get('model', {}).get('lightgbm', {}).get('use_focal_loss', False)
        if self.use_focal_loss:
            focal_config = config.get('model', {}).get('lightgbm', {}).get('focal_loss', {})
            self.focal_alpha = focal_config.get('alpha', 0.25)
            self.focal_gamma = focal_config.get('gamma', 2.0)
            
            # Focal Loss 파라미터 검증
            if not validate_focal_loss_parameters(self.focal_alpha, self.focal_gamma):
                logger.warning("Focal Loss 파라미터가 유효하지 않습니다. 기본값을 사용합니다.")
                self.focal_alpha = 0.25
                self.focal_gamma = 2.0
            
            logger.info(f"Focal Loss 활성화: alpha={self.focal_alpha}, gamma={self.focal_gamma}")
        
        # 모델 파라미터
        self.model_params = config['model']['lightgbm']
        
        logger.info(f"  - Focal Loss 사용: {self.use_focal_loss}")
    
    def _focal_loss_objective(self, predt: np.ndarray, dtrain: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        LightGBM용 Focal Loss objective 함수
        
        Args:
            predt: 예측값
            dtrain: LightGBM 데이터셋
            
        Returns:
            (gradient, hessian) 튜플
        """
        y_true = dtrain.get_label()
        
        # 시그모이드 적용
        predt = 1.0 / (1.0 + np.exp(-predt))
        
        # 수치 안정성을 위한 클리핑
        predt = np.clip(predt, 1e-7, 1.0 - 1e-7)
        
        # 예측 확률
        pt = y_true * predt + (1 - y_true) * (1 - predt)
        
        # Focal Loss 가중치
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_weight = self.focal_alpha * y_true + (1 - self.focal_alpha) * (1 - y_true)
        
        # 그래디언트와 헤시안 계산
        grad = alpha_weight * focal_weight * (predt - y_true)
        hess = alpha_weight * focal_weight * predt * (1 - predt)
        
        return grad, hess
    
    def _focal_loss_eval(self, predt: np.ndarray, dtrain: lgb.Dataset) -> Tuple[str, float, bool]:
        """
        LightGBM용 Focal Loss 평가 함수
        
        Args:
            predt: 예측값
            dtrain: LightGBM 데이터셋
            
        Returns:
            (metric_name, metric_value, is_higher_better) 튜플
        """
        y_true = dtrain.get_label()
        
        # 시그모이드 적용
        predt = 1.0 / (1.0 + np.exp(-predt))
        
        # Focal Loss 계산
        focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        loss_value = focal_loss(y_true, predt)
        
        return 'focal_loss', loss_value, False
    
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
            if self.use_focal_loss:
                # Focal Loss 사용 시: custom objective 사용
                params['objective'] = 'none'  # custom objective 사용
                params['metric'] = 'none'     # custom metric 사용
                # Focal Loss 관련 파라미터는 별도로 관리
                params['focal_loss_enabled'] = True
                params['focal_alpha'] = self.focal_alpha
                params['focal_gamma'] = self.focal_gamma
                logger.info(f"  - Focal Loss 활성화: alpha={self.focal_alpha}, gamma={self.focal_gamma}")
            else:
                # 기존 방식: class_weight 사용
                params['objective'] = 'binary'
                params['metric'] = 'binary_logloss'
                params['class_weight'] = 'balanced'
                logger.info(f"  - class_weight: balanced")
        else:
            # 회귀 문제
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        
        return params
    
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
            y = y.select_dtypes(include=['number', 'bool', 'category'])
            y = y.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"전처리 후 데이터 형태: X={X.shape}, y={y.shape}")
        
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
            
            # 모델 파라미터 준비
            params = self._get_model_params(target)
            
            # 분류 문제인 경우 불균형 처리
            if target in self.classification_targets:
                if self.use_focal_loss:
                    # Focal Loss 사용 시: custom objective 사용
                    logger.info(f"  - Focal Loss 사용으로 class_weight 비활성화")
                else:
                    # 기존 방식: class_weight 사용
                    logger.info(f"  - class_weight: balanced")
            
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
            
            # Focal Loss 관련 파라미터 제거 (LightGBM에서 지원하지 않음)
            focal_loss_params = ['focal_loss_enabled', 'focal_alpha', 'focal_gamma']
            for param in focal_loss_params:
                if param in params:
                    params.pop(param)
            
            # 모델 학습
            logger.info(f"LightGBM 모델 학습 시작 (타겟: {target})")
            if use_early_stopping and valid_data is not None:
                if self.use_focal_loss and target in self.classification_targets:
                    # Focal Loss 사용 시: custom objective와 eval 함수 사용
                    logger.info("Focal Loss custom objective 사용")
                    model = lgb.train(
                        params,
                        train_data,
                        valid_sets=[valid_data],
                        valid_names=['valid'],
                        fobj=self._focal_loss_objective,
                        feval=self._focal_loss_eval,
                        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
                    )
                else:
                    # 일반적인 학습
                    logger.info("일반 학습 모드")
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
                if self.use_focal_loss and target in self.classification_targets:
                    logger.info("Focal Loss custom objective 사용")
                    model = lgb.train(
                        params,
                        train_data,
                        fobj=self._focal_loss_objective,
                        feval=self._focal_loss_eval,
                        callbacks=[lgb.log_evaluation(0)]
                    )
                else:
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
        
        logger.info("LightGBM 모델 확률 예측 시작")
        
        # 입력 데이터 검증 및 전처리
        X = self._validate_input_data(X)
        
        proba_predictions = {}
        
        for target in self.classification_targets:
            if target not in self.models:
                continue
            
            model = self.models[target]
            
            # 확률 예측 수행 (LightGBM은 predict_proba 대신 predict 사용)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # LightGBM의 predict는 기본적으로 확률을 반환
                proba = model.predict(X)
                # 이진 분류의 경우 [0, 1] 확률을 [0, 1] 형태로 변환
                proba_reshaped = np.column_stack([1 - proba, proba])
            
            proba_predictions[target] = proba_reshaped
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


def main():
    """테스트용 메인 함수"""
    # 설정 예시
    config = {
        'features': {
            'target_columns': ['suicide_a_next_year']
        },
        'model': {
            'model_type': 'lightgbm',
            'lightgbm': {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'early_stopping_rounds': 10
            }
        }
    }
    
    # 모델 생성
    model = LightGBMModel(config)
    print(f"모델 타입: {model.model_type}")
    print(f"회귀 타겟: {model.regression_targets}")
    print(f"분류 타겟: {model.classification_targets}")


if __name__ == "__main__":
    main() 