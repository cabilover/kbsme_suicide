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
from sklearn.base import BaseEstimator
import warnings
from src.utils import find_column_with_remainder
from src.models.loss_functions import FocalLoss, validate_focal_loss_parameters

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel(BaseEstimator):
    """
    XGBoost 기반 다중 출력 모델
    
    회귀와 분류 문제를 모두 지원하며, 각 타겟별로 별도의 모델을 학습합니다.
    Focal Loss를 통한 불균형 데이터 처리도 지원합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        XGBoost 모델 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.models = {}
        self.target_columns = config['features']['target_columns']
        self.regression_targets = []
        self.classification_targets = []
        self.is_fitted = False
        
        # Focal Loss 설정 확인
        self.use_focal_loss = config.get('model', {}).get('xgboost', {}).get('use_focal_loss', False)
        if self.use_focal_loss:
            focal_config = config.get('model', {}).get('xgboost', {}).get('focal_loss', {})
            self.focal_alpha = focal_config.get('alpha', 0.25)
            self.focal_gamma = focal_config.get('gamma', 2.0)
            
            # Focal Loss 파라미터 검증
            if not validate_focal_loss_parameters(self.focal_alpha, self.focal_gamma):
                logger.warning("Focal Loss 파라미터가 유효하지 않습니다. 기본값을 사용합니다.")
                self.focal_alpha = 0.25
                self.focal_gamma = 2.0
            
            logger.info(f"Focal Loss 활성화: alpha={self.focal_alpha}, gamma={self.focal_gamma}")
        
        # 타겟 타입 분류
        self._classify_targets()
        
        # 모델 파라미터
        self.model_params = config['model']['xgboost']
        
        logger.info(f"XGBoost 모델 초기화 완료")
        logger.info(f"  - 회귀 타겟: {self.regression_targets}")
        logger.info(f"  - 분류 타겟: {self.classification_targets}")
        logger.info(f"  - Focal Loss 사용: {self.use_focal_loss}")
    
    def _classify_targets(self):
        """타겟을 회귀와 분류로 분류합니다."""
        for target in self.target_columns:
            if target.endswith('_next_year'):
                base_target = target.replace('_next_year', '')
                if base_target in ['anxiety_score', 'depress_score', 'sleep_score']:
                    self.regression_targets.append(target)
                elif base_target in ['suicide_t', 'suicide_a']:
                    self.classification_targets.append(target)
    
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
            # Focal Loss 사용 여부에 따른 파라미터 설정
            if self.use_focal_loss:
                # Focal Loss 사용 시: 기본 objective 유지, Focal Loss는 별도 처리
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
                # Focal Loss 관련 파라미터는 별도로 관리
                params['focal_loss_enabled'] = True
                params['focal_alpha'] = self.focal_alpha
                params['focal_gamma'] = self.focal_gamma
                logger.info(f"  - Focal Loss 활성화: alpha={self.focal_alpha}, gamma={self.focal_gamma}")
            else:
                # 기존 방식: scale_pos_weight 사용
                params['scale_pos_weight'] = self.model_params.get('scale_pos_weight', 1.0)
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
        else:
            # 회귀 문제
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
        
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
        
        # XGBoost 입력에 object 컬럼이 포함되지 않도록 처리
        X = X.select_dtypes(include=['number', 'bool', 'category'])
        if X_val is not None:
            X_val = X_val.select_dtypes(include=['number', 'bool', 'category'])
        y = y.select_dtypes(include=['number', 'bool', 'category'])
        if y_val is not None:
            y_val = y_val.select_dtypes(include=['number', 'bool', 'category'])
        
        # 사용 가능한 타겟 컬럼 찾기
        available_targets = []
        for target in self.target_columns:
            col = find_column_with_remainder(list(y.columns), target)
            if col:
                available_targets.append(col)
        logger.info(f"사용 가능한 타겟 컬럼: {available_targets}")
        
        for target in available_targets:
            original_target = target.replace('remainder__', '') if target.startswith('remainder__') else target
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
                if self.use_focal_loss:
                    # Focal Loss 사용 시: scale_pos_weight는 사용하지 않음
                    logger.info(f"  - Focal Loss 사용으로 scale_pos_weight 비활성화")
                else:
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
            
            # Focal Loss 관련 파라미터 제거 (XGBoost에서 지원하지 않음)
            focal_loss_params = ['focal_loss_enabled', 'focal_alpha', 'focal_gamma']
            for param in focal_loss_params:
                if param in model_params:
                    model_params.pop(param)
            
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
                model.fit(X_target, y_target)
            
            self.models[original_target] = model
            logger.info(f"  - {original_target} 모델 학습 완료")
        
        self.is_fitted = True
        logger.info(f"모든 모델 학습 완료 ({len(self.models)}개)")
        return self
    
    def _calculate_scale_pos_weight(self, y: pd.Series) -> float:
        """
        분류 문제에서 scale_pos_weight를 계산합니다.
        
        Args:
            y: 타겟 시리즈
            
        Returns:
            scale_pos_weight 값
        """
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        
        if pos_count == 0:
            return 1.0
        
        return neg_count / pos_count
    
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
        
        # XGBoost 입력에 object 컬럼이 포함되지 않도록 처리
        X = X.select_dtypes(include=['number', 'bool', 'category'])
        
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
    
    def save_model(self, filepath: str):
        """
        모델을 저장합니다.
        
        Args:
            filepath: 저장할 파일 경로
        """
        import joblib
        
        model_data = {
            'models': self.models,
            'target_columns': self.target_columns,
            'regression_targets': self.regression_targets,
            'classification_targets': self.classification_targets,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"모델 저장 완료: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'XGBoostModel':
        """
        저장된 모델을 로드합니다.
        
        Args:
            filepath: 모델 파일 경로
            
        Returns:
            로드된 모델 인스턴스
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        # 모델 인스턴스 생성
        model = cls(model_data['config'])
        model.models = model_data['models']
        model.target_columns = model_data['target_columns']
        model.regression_targets = model_data['regression_targets']
        model.classification_targets = model_data['classification_targets']
        model.is_fitted = model_data['is_fitted']
        
        logger.info(f"모델 로드 완료: {filepath}")
        return model


def main():
    """
    테스트용 메인 함수
    """
    import yaml
    
    # 설정 로드
    with open("configs/default_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    
    # 타겟 데이터 생성
    y = pd.DataFrame({
        'anxiety_score_next_year': np.random.randn(n_samples),
        'depress_score_next_year': np.random.randn(n_samples),
        'sleep_score_next_year': np.random.randn(n_samples),
        'suicide_t_next_year': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
        'suicide_a_next_year': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    })
    
    # 모델 생성 및 학습
    model = XGBoostModel(config)
    model.fit(X, y)
    
    # 예측
    predictions = model.predict(X)
    
    # 피처 중요도
    importance = model.get_feature_importance()
    
    logger.info("XGBoost 모델 테스트 완료!")


if __name__ == "__main__":
    main() 