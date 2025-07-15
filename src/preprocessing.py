"""
데이터 전처리 모듈

이 모듈은 데이터 유출을 방지하면서 결측치를 처리하는 전처리 파이프라인을 구현합니다.
- ID별 시계열 보간 (ffill, bfill)
- sklearn 기반 결측치 처리
- 범주형 인코딩 (One-Hot Encoding)
- 데이터 유출 방지 원칙 준수
- 불균형 데이터 처리를 위한 리샘플링 기법들
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
import warnings
from src.utils import setup_logging

# 로깅 설정
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


class BaseResampler:
    """리샘플링 기법들의 기본 클래스"""
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 재현성을 위한 시드
        """
        self.random_state = random_state
        self.rs = np.random.RandomState(random_state)
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, 
                    id_column: str = 'id') -> Tuple[pd.DataFrame, pd.Series]:
        """
        리샘플링을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 시리즈
            id_column: ID 컬럼명
            
        Returns:
            리샘플링된 (X, y) 튜플
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다")
    
    def _validate_input(self, X: pd.DataFrame, y: pd.Series, id_column: str) -> None:
        """입력 데이터 검증"""
        if len(X) != len(y):
            raise ValueError("X와 y의 길이가 일치하지 않습니다")
        if id_column not in X.columns:
            raise ValueError(f"ID 컬럼 '{id_column}'이 X에 존재하지 않습니다")
    
    def _get_class_distribution(self, y: pd.Series) -> Dict[int, int]:
        """클래스 분포를 반환합니다"""
        return y.value_counts().to_dict()


class SMOTEResampler(BaseResampler):
    """SMOTE (Synthetic Minority Over-sampling Technique) 구현"""
    
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        """
        Args:
            k_neighbors: K-NN에서 사용할 이웃 수
            random_state: 재현성을 위한 시드
        """
        super().__init__(random_state)
        self.k_neighbors = k_neighbors
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, 
                    id_column: str = 'id') -> Tuple[pd.DataFrame, pd.Series]:
        """
        SMOTE를 사용하여 소수 클래스 오버샘플링을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 시리즈
            id_column: ID 컬럼명
            
        Returns:
            리샘플링된 (X, y) 튜플
        """
        self._validate_input(X, y, id_column)
        
        # 클래스 분포 확인
        class_counts = self._get_class_distribution(y)
        logger.info(f"SMOTE 적용 전 클래스 분포: {class_counts}")
        
        if len(class_counts) != 2:
            raise ValueError("SMOTE는 이진 분류 문제에만 적용 가능합니다")
        
        # 다수 클래스와 소수 클래스 식별
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        # 소수 클래스 샘플들
        minority_samples = X[y == minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        
        if len(minority_samples) <= self.k_neighbors:
            logger.warning(f"소수 클래스 샘플 수({len(minority_samples)})가 k_neighbors({self.k_neighbors})보다 작습니다. k_neighbors를 {len(minority_samples)-1}로 조정합니다.")
            self.k_neighbors = min(self.k_neighbors, len(minority_samples) - 1)
        
        # 필요한 합성 샘플 수 계산
        target_count = class_counts[majority_class]
        synthetic_count = target_count - class_counts[minority_class]
        
        if synthetic_count <= 0:
            logger.info("이미 클래스가 균형잡혀 있습니다. SMOTE를 적용하지 않습니다.")
            return X, y
        
        # K-NN 모델 학습 (ID 컬럼 제외)
        feature_cols = [col for col in X.columns if col != id_column]
        knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='auto')
        knn.fit(minority_samples[feature_cols])
        
        # 합성 샘플 생성
        synthetic_samples = []
        synthetic_labels = []
        
        for _ in range(synthetic_count):
            # 랜덤하게 소수 클래스 샘플 선택
            random_idx = self.rs.randint(0, len(minority_samples))
            random_sample = minority_samples.iloc[random_idx]
            
            # K-NN으로 이웃 찾기
            distances, indices = knn.kneighbors([random_sample[feature_cols]])
            
            # 자기 자신을 제외하고 랜덤 이웃 선택
            neighbor_idx = self.rs.choice(indices[0][1:])  # 첫 번째는 자기 자신
            neighbor_sample = minority_samples.iloc[neighbor_idx]
            
            # 합성 샘플 생성 (선형 보간)
            alpha = self.rs.random()
            synthetic_sample = random_sample.copy()
            
            for col in feature_cols:
                synthetic_sample[col] = (alpha * random_sample[col] + 
                                       (1 - alpha) * neighbor_sample[col])
            
            synthetic_samples.append(synthetic_sample)
            synthetic_labels.append(minority_class)
        
        # 원본 데이터와 합성 데이터 결합
        synthetic_df = pd.DataFrame(synthetic_samples)
        synthetic_series = pd.Series(synthetic_labels, index=synthetic_df.index)
        
        X_resampled = pd.concat([X, synthetic_df], ignore_index=True)
        y_resampled = pd.concat([y, synthetic_series], ignore_index=True)
        
        # 결과 검증
        final_class_counts = self._get_class_distribution(y_resampled)
        logger.info(f"SMOTE 적용 후 클래스 분포: {final_class_counts}")
        logger.info(f"생성된 합성 샘플 수: {synthetic_count}")
        
        return X_resampled, y_resampled


class BorderlineSMOTEResampler(BaseResampler):
    """Borderline SMOTE 구현 - 경계선 근처의 소수 클래스 샘플에 집중"""
    
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        """
        Args:
            k_neighbors: K-NN에서 사용할 이웃 수
            random_state: 재현성을 위한 시드
        """
        super().__init__(random_state)
        self.k_neighbors = k_neighbors
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, 
                    id_column: str = 'id') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Borderline SMOTE를 사용하여 경계선 근처의 소수 클래스 오버샘플링을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 시리즈
            id_column: ID 컬럼명
            
        Returns:
            리샘플링된 (X, y) 튜플
        """
        self._validate_input(X, y, id_column)
        
        # 클래스 분포 확인
        class_counts = self._get_class_distribution(y)
        logger.info(f"Borderline SMOTE 적용 전 클래스 분포: {class_counts}")
        
        if len(class_counts) != 2:
            raise ValueError("Borderline SMOTE는 이진 분류 문제에만 적용 가능합니다")
        
        # 다수 클래스와 소수 클래스 식별
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        # 소수 클래스 샘플들
        minority_samples = X[y == minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        
        if len(minority_samples) <= self.k_neighbors:
            logger.warning(f"소수 클래스 샘플 수({len(minority_samples)})가 k_neighbors({self.k_neighbors})보다 작습니다. k_neighbors를 {len(minority_samples)-1}로 조정합니다.")
            self.k_neighbors = min(self.k_neighbors, len(minority_samples) - 1)
        
        # 필요한 합성 샘플 수 계산
        target_count = class_counts[majority_class]
        synthetic_count = target_count - class_counts[minority_class]
        
        if synthetic_count <= 0:
            logger.info("이미 클래스가 균형잡혀 있습니다. Borderline SMOTE를 적용하지 않습니다.")
            return X, y
        
        # K-NN 모델 학습 (전체 데이터에 대해)
        feature_cols = [col for col in X.columns if col != id_column]
        knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='auto')
        knn.fit(X[feature_cols])
        
        # 각 소수 클래스 샘플의 위험도 계산
        danger_samples = []
        safe_samples = []
        noise_samples = []
        
        for idx, sample in minority_samples.iterrows():
            # K-NN으로 이웃 찾기
            distances, indices = knn.kneighbors([sample[feature_cols]])
            
            # 이웃들의 클래스 확인
            neighbor_labels = y.iloc[indices[0][1:]]  # 자기 자신 제외
            majority_neighbors = sum(neighbor_labels == majority_class)
            
            # 위험도 분류
            if majority_neighbors >= self.k_neighbors * 0.5 and majority_neighbors < self.k_neighbors:
                danger_samples.append(idx)
            elif majority_neighbors < self.k_neighbors * 0.5:
                safe_samples.append(idx)
            else:
                noise_samples.append(idx)
        
        logger.info(f"위험도 분류 결과 - DANGER: {len(danger_samples)}, SAFE: {len(safe_samples)}, NOISE: {len(noise_samples)}")
        
        # DANGER 샘플들에 대해서만 합성 샘플 생성
        if len(danger_samples) == 0:
            logger.warning("DANGER 샘플이 없습니다. 일반 SMOTE를 적용합니다.")
            smote = SMOTEResampler(k_neighbors=self.k_neighbors, random_state=self.random_state)
            return smote.fit_resample(X, y, id_column)
        
        # 합성 샘플 생성
        synthetic_samples = []
        synthetic_labels = []
        
        for _ in range(synthetic_count):
            # DANGER 샘플 중 랜덤 선택
            danger_idx = self.rs.choice(danger_samples)
            danger_sample = minority_samples.loc[danger_idx]
            
            # 해당 샘플의 K-NN 이웃들 중 소수 클래스 이웃 찾기
            distances, indices = knn.kneighbors([danger_sample[feature_cols]])
            minority_neighbors = []
            
            for neighbor_idx in indices[0][1:]:  # 자기 자신 제외
                if y.iloc[neighbor_idx] == minority_class:
                    minority_neighbors.append(neighbor_idx)
            
            if len(minority_neighbors) == 0:
                continue
            
            # 랜덤 소수 클래스 이웃 선택
            neighbor_idx = self.rs.choice(minority_neighbors)
            neighbor_sample = X.iloc[neighbor_idx]
            
            # 합성 샘플 생성 (선형 보간)
            alpha = self.rs.random()
            synthetic_sample = danger_sample.copy()
            
            for col in feature_cols:
                synthetic_sample[col] = (alpha * danger_sample[col] + 
                                       (1 - alpha) * neighbor_sample[col])
            
            synthetic_samples.append(synthetic_sample)
            synthetic_labels.append(minority_class)
        
        # 원본 데이터와 합성 데이터 결합
        if synthetic_samples:
            synthetic_df = pd.DataFrame(synthetic_samples)
            synthetic_series = pd.Series(synthetic_labels, index=synthetic_df.index)
            
            X_resampled = pd.concat([X, synthetic_df], ignore_index=True)
            y_resampled = pd.concat([y, synthetic_series], ignore_index=True)
        else:
            X_resampled = X.copy()
            y_resampled = y.copy()
        
        # 결과 검증
        final_class_counts = self._get_class_distribution(y_resampled)
        logger.info(f"Borderline SMOTE 적용 후 클래스 분포: {final_class_counts}")
        logger.info(f"생성된 합성 샘플 수: {len(synthetic_samples)}")
        
        return X_resampled, y_resampled


class ADASYNResampler(BaseResampler):
    """ADASYN (Adaptive Synthetic Sampling) 구현"""
    
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        """
        Args:
            k_neighbors: K-NN에서 사용할 이웃 수
            random_state: 재현성을 위한 시드
        """
        super().__init__(random_state)
        self.k_neighbors = k_neighbors
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, 
                    id_column: str = 'id') -> Tuple[pd.DataFrame, pd.Series]:
        """
        ADASYN을 사용하여 적응적 소수 클래스 오버샘플링을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 시리즈
            id_column: ID 컬럼명
            
        Returns:
            리샘플링된 (X, y) 튜플
        """
        self._validate_input(X, y, id_column)
        
        # 클래스 분포 확인
        class_counts = self._get_class_distribution(y)
        logger.info(f"ADASYN 적용 전 클래스 분포: {class_counts}")
        
        if len(class_counts) != 2:
            raise ValueError("ADASYN은 이진 분류 문제에만 적용 가능합니다")
        
        # 다수 클래스와 소수 클래스 식별
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        # 소수 클래스 샘플들
        minority_samples = X[y == minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        
        if len(minority_samples) <= self.k_neighbors:
            logger.warning(f"소수 클래스 샘플 수({len(minority_samples)})가 k_neighbors({self.k_neighbors})보다 작습니다. k_neighbors를 {len(minority_samples)-1}로 조정합니다.")
            self.k_neighbors = min(self.k_neighbors, len(minority_samples) - 1)
        
        # 필요한 합성 샘플 수 계산
        target_count = class_counts[majority_class]
        synthetic_count = target_count - class_counts[minority_class]
        
        if synthetic_count <= 0:
            logger.info("이미 클래스가 균형잡혀 있습니다. ADASYN을 적용하지 않습니다.")
            return X, y
        
        # K-NN 모델 학습 (전체 데이터에 대해)
        feature_cols = [col for col in X.columns if col != id_column]
        knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='auto')
        knn.fit(X[feature_cols])
        
        # 각 소수 클래스 샘플의 학습 난이도 계산
        learning_difficulties = []
        
        for idx, sample in minority_samples.iterrows():
            # K-NN으로 이웃 찾기
            distances, indices = knn.kneighbors([sample[feature_cols]])
            
            # 이웃들의 클래스 확인
            neighbor_labels = y.iloc[indices[0][1:]]  # 자기 자신 제외
            majority_neighbors = sum(neighbor_labels == majority_class)
            
            # 학습 난이도 = 다수 클래스 이웃 비율
            difficulty = majority_neighbors / self.k_neighbors
            learning_difficulties.append(difficulty)
        
        # 학습 난이도 정규화
        total_difficulty = sum(learning_difficulties)
        if total_difficulty == 0:
            # 모든 샘플이 쉬운 경우 균등 분배
            normalized_difficulties = [1.0 / len(learning_difficulties)] * len(learning_difficulties)
        else:
            normalized_difficulties = [d / total_difficulty for d in learning_difficulties]
        
        # 각 샘플별로 생성할 합성 샘플 수 계산
        samples_to_generate = [int(round(d * synthetic_count)) for d in normalized_difficulties]
        
        # 반올림으로 인한 차이 조정
        while sum(samples_to_generate) < synthetic_count:
            # 가장 높은 난이도를 가진 샘플에 추가
            max_idx = learning_difficulties.index(max(learning_difficulties))
            samples_to_generate[max_idx] += 1
        
        # 합성 샘플 생성
        synthetic_samples = []
        synthetic_labels = []
        
        for i, (idx, sample) in enumerate(minority_samples.iterrows()):
            num_to_generate = samples_to_generate[i]
            
            for _ in range(num_to_generate):
                # K-NN으로 이웃 찾기
                distances, indices = knn.kneighbors([sample[feature_cols]])
                
                # 소수 클래스 이웃들 찾기
                minority_neighbors = []
                for neighbor_idx in indices[0][1:]:  # 자기 자신 제외
                    if y.iloc[neighbor_idx] == minority_class:
                        minority_neighbors.append(neighbor_idx)
                
                if len(minority_neighbors) == 0:
                    continue
                
                # 랜덤 소수 클래스 이웃 선택
                neighbor_idx = self.rs.choice(minority_neighbors)
                neighbor_sample = X.iloc[neighbor_idx]
                
                # 합성 샘플 생성 (선형 보간)
                alpha = self.rs.random()
                synthetic_sample = sample.copy()
                
                for col in feature_cols:
                    synthetic_sample[col] = (alpha * sample[col] + 
                                           (1 - alpha) * neighbor_sample[col])
                
                synthetic_samples.append(synthetic_sample)
                synthetic_labels.append(minority_class)
        
        # 원본 데이터와 합성 데이터 결합
        if synthetic_samples:
            synthetic_df = pd.DataFrame(synthetic_samples)
            synthetic_series = pd.Series(synthetic_labels, index=synthetic_df.index)
            
            X_resampled = pd.concat([X, synthetic_df], ignore_index=True)
            y_resampled = pd.concat([y, synthetic_series], ignore_index=True)
        else:
            X_resampled = X.copy()
            y_resampled = y.copy()
        
        # 결과 검증
        final_class_counts = self._get_class_distribution(y_resampled)
        logger.info(f"ADASYN 적용 후 클래스 분포: {final_class_counts}")
        logger.info(f"생성된 합성 샘플 수: {len(synthetic_samples)}")
        
        return X_resampled, y_resampled


class UnderSampler(BaseResampler):
    """언더샘플링 기법 구현"""
    
    def __init__(self, strategy: str = 'random', random_state: int = 42):
        """
        Args:
            strategy: 언더샘플링 전략 ('random', 'tomek', 'enn')
            random_state: 재현성을 위한 시드
        """
        super().__init__(random_state)
        self.strategy = strategy
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, 
                    id_column: str = 'id') -> Tuple[pd.DataFrame, pd.Series]:
        """
        언더샘플링을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 시리즈
            id_column: ID 컬럼명
            
        Returns:
            리샘플링된 (X, y) 튜플
        """
        self._validate_input(X, y, id_column)
        
        # 클래스 분포 확인
        class_counts = self._get_class_distribution(y)
        logger.info(f"언더샘플링 적용 전 클래스 분포: {class_counts}")
        
        if len(class_counts) != 2:
            raise ValueError("언더샘플링은 이진 분류 문제에만 적용 가능합니다")
        
        # 다수 클래스와 소수 클래스 식별
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        # 소수 클래스 샘플들
        minority_samples = X[y == minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        
        if self.strategy == 'random':
            # 랜덤 언더샘플링
            majority_samples = X[y == majority_class].copy()
            majority_labels = y[y == majority_class].copy()
            
            # 소수 클래스 수만큼 다수 클래스 샘플링
            n_minority = len(minority_samples)
            majority_downsampled = majority_samples.sample(n=n_minority, random_state=self.random_state)
            majority_labels_downsampled = majority_labels.loc[majority_downsampled.index]
            
            # 결과 결합
            X_resampled = pd.concat([minority_samples, majority_downsampled], ignore_index=True)
            y_resampled = pd.concat([minority_labels, majority_labels_downsampled], ignore_index=True)
            
        elif self.strategy == 'tomek':
            # Tomek Links 기반 언더샘플링 (간단한 구현)
            logger.warning("Tomek Links 언더샘플링은 복잡하므로 랜덤 언더샘플링으로 대체합니다.")
            return self.fit_resample(X, y, id_column)
            
        elif self.strategy == 'enn':
            # Edited Nearest Neighbors (간단한 구현)
            logger.warning("ENN 언더샘플링은 복잡하므로 랜덤 언더샘플링으로 대체합니다.")
            return self.fit_resample(X, y, id_column)
            
        else:
            raise ValueError(f"지원하지 않는 언더샘플링 전략: {self.strategy}")
        
        # 결과 검증
        final_class_counts = self._get_class_distribution(y_resampled)
        logger.info(f"언더샘플링 적용 후 클래스 분포: {final_class_counts}")
        
        return X_resampled, y_resampled


class HybridResampler(BaseResampler):
    """하이브리드 샘플링 기법 구현"""
    
    def __init__(self, strategy: str = 'smote_tomek', random_state: int = 42):
        """
        Args:
            strategy: 하이브리드 전략 ('smote_tomek', 'smote_enn')
            random_state: 재현성을 위한 시드
        """
        super().__init__(random_state)
        self.strategy = strategy
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, 
                    id_column: str = 'id') -> Tuple[pd.DataFrame, pd.Series]:
        """
        하이브리드 샘플링을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 시리즈
            id_column: ID 컬럼명
            
        Returns:
            리샘플링된 (X, y) 튜플
        """
        self._validate_input(X, y, id_column)
        
        logger.info(f"하이브리드 샘플링 적용: {self.strategy}")
        
        if self.strategy == 'smote_tomek':
            # SMOTE + Tomek Links
            smote = SMOTEResampler(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y, id_column)
            
            # Tomek Links 제거 (간단한 구현으로 생략)
            logger.info("Tomek Links 제거는 생략하고 SMOTE만 적용합니다.")
            
        elif self.strategy == 'smote_enn':
            # SMOTE + ENN
            smote = SMOTEResampler(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y, id_column)
            
            # ENN 제거 (간단한 구현으로 생략)
            logger.info("ENN 제거는 생략하고 SMOTE만 적용합니다.")
            
        else:
            raise ValueError(f"지원하지 않는 하이브리드 전략: {self.strategy}")
        
        return X_resampled, y_resampled


class ResamplingPipeline:
    """리샘플링 파이프라인 통합 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 리샘플링 설정이 포함된 설정 딕셔너리
        """
        self.config = config
        self.resampler = None
        self._setup_resampler()
    
    def _setup_resampler(self):
        """설정에 따라 리샘플러를 설정합니다"""
        resampling_config = self.config.get('resampling', {})
        
        if not resampling_config.get('enabled', False):
            logger.info("리샘플링이 비활성화되어 있습니다.")
            return
        
        method = resampling_config.get('method', 'smote')
        random_state = resampling_config.get('random_state', 42)
        
        if method == 'smote':
            k_neighbors = resampling_config.get('smote_k_neighbors', 5)
            self.resampler = SMOTEResampler(k_neighbors=k_neighbors, random_state=random_state)
            
        elif method == 'borderline_smote':
            k_neighbors = resampling_config.get('borderline_smote_k_neighbors', 5)
            self.resampler = BorderlineSMOTEResampler(k_neighbors=k_neighbors, random_state=random_state)
            
        elif method == 'adasyn':
            k_neighbors = resampling_config.get('adasyn_k_neighbors', 5)
            self.resampler = ADASYNResampler(k_neighbors=k_neighbors, random_state=random_state)
            
        elif method == 'under_sampling':
            strategy = resampling_config.get('under_sampling_strategy', 'random')
            self.resampler = UnderSampler(strategy=strategy, random_state=random_state)
            
        elif method == 'hybrid':
            strategy = resampling_config.get('hybrid_strategy', 'smote_tomek')
            self.resampler = HybridResampler(strategy=strategy, random_state=random_state)
            
        else:
            raise ValueError(f"지원하지 않는 리샘플링 방법: {method}")
        
        logger.info(f"리샘플러 설정 완료: {method}")
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series, 
                    id_column: str = 'id') -> Tuple[pd.DataFrame, pd.Series]:
        """
        리샘플링을 수행합니다.
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 시리즈
            id_column: ID 컬럼명
            
        Returns:
            리샘플링된 (X, y) 튜플
        """
        if self.resampler is None:
            logger.info("리샘플러가 설정되지 않았습니다. 원본 데이터를 반환합니다.")
            return X, y
        
        return self.resampler.fit_resample(X, y, id_column)


def apply_timeseries_imputation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    ID별로 시계열 보간을 적용합니다.
    
    Args:
        df: 입력 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        시계열 보간이 적용된 데이터프레임
    """
    id_column = config['time_series']['id_column']
    date_column = config['time_series']['date_column']
    
    # 시계열 보간 대상 컬럼
    ts_columns = config['preprocessing']['time_series_imputation']['columns']
    fallback_strategy = config['preprocessing']['time_series_imputation']['fallback_strategy']
    
    logger.info(f"시계열 보간 적용: {ts_columns}")
    logger.info(f"Fallback 전략: {fallback_strategy}")
    
    # 데이터 복사
    df_imputed = df.copy()
    
    # ID별로 그룹화하여 시계열 보간 적용
    for col in ts_columns:
        if col in df.columns:
            # ID별로 정렬 후 ffill, bfill 적용
            df_imputed[col] = df_imputed.groupby(id_column)[col].ffill().bfill()
            
            # 여전히 결측치가 있다면 fallback 전략 적용
            if df_imputed[col].isnull().any():
                if fallback_strategy == "mean":
                    fallback_val = df_imputed[col].mean()
                elif fallback_strategy == "median":
                    fallback_val = df_imputed[col].median()
                elif fallback_strategy == "constant":
                    fallback_val = 0  # 설정에서 constant_value를 추가할 수 있음
                else:
                    fallback_val = df_imputed[col].mean()  # 기본값
                
                df_imputed[col].fillna(fallback_val, inplace=True)
                logger.info(f"  {col}: {fallback_strategy} 값으로 추가 보간 ({fallback_val:.4f})")
    
    return df_imputed


def get_numerical_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    수치형 컬럼 목록을 반환합니다.
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        수치형 컬럼 목록
    """
    # 설정에서 명시된 수치형 컬럼
    numerical_cols = config['preprocessing']['numerical_imputation']['columns']
    
    # 실제 존재하는 컬럼만 필터링
    available_numerical = [col for col in numerical_cols if col in df.columns]
    
    logger.info(f"수치형 컬럼: {available_numerical}")
    return available_numerical


def get_categorical_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    범주형 컬럼 목록을 반환합니다.
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        범주형 컬럼 목록
    """
    # 설정에서 명시된 범주형 컬럼
    categorical_cols = config['preprocessing']['categorical_imputation']['columns']
    
    # 실제 존재하는 컬럼만 필터링
    available_categorical = [col for col in categorical_cols if col in df.columns]
    
    logger.info(f"범주형 컬럼: {available_categorical}")
    return available_categorical


def get_passthrough_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    전처리 파이프라인에서 그대로 통과시킬 컬럼 목록을 반환합니다.
    
    Args:
        df: 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        통과시킬 컬럼 목록
    """
    # 기본적으로 제외할 컬럼들 (ID 컬럼은 리샘플링을 위해 포함)
    exclude_columns = [
        config['time_series']['date_column'],
        config['time_series']['year_column']
    ]
    
    # 타겟 컬럼들은 반드시 포함 (피처 엔지니어링 후 분리하기 위해)
    target_columns = config['features']['target_columns']
    
    # 이미 처리되는 컬럼들도 제외
    numerical_cols = get_numerical_columns(df, config)
    categorical_cols = get_categorical_columns(df, config)
    exclude_columns.extend(numerical_cols)
    exclude_columns.extend(categorical_cols)
    
    # 통과시킬 컬럼들 (존재하는 컬럼만)
    passthrough_cols = [col for col in df.columns if col not in exclude_columns]
    
    # 타겟 컬럼들이 제외되었다면 다시 추가
    for target in target_columns:
        if target in df.columns and target not in passthrough_cols:
            passthrough_cols.append(target)
            logger.info(f"타겟 컬럼 '{target}'을 통과 컬럼에 추가")
    
    # ID 컬럼이 제외되었다면 다시 추가 (리샘플링을 위해 필요)
    id_column = config['time_series']['id_column']
    if id_column in df.columns and id_column not in passthrough_cols:
        passthrough_cols.append(id_column)
        logger.info(f"ID 컬럼 '{id_column}'을 통과 컬럼에 추가 (리샘플링용)")
    
    logger.info(f"통과시킬 컬럼: {passthrough_cols}")
    return passthrough_cols


def create_preprocessing_pipeline(config: Dict[str, Any]) -> ColumnTransformer:
    """
    전처리 파이프라인을 생성합니다.
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        전처리 파이프라인
    """
    # 수치형 결측치 처리 + 스케일링
    numerical_strategy = config['preprocessing']['numerical_imputation']['strategy']
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=numerical_strategy)),
        ('scaler', StandardScaler())
    ])
    
    # 범주형 결측치 처리 + 인코딩
    categorical_strategy = config['preprocessing']['categorical_imputation']['strategy']
    categorical_encoding = config['preprocessing']['categorical_encoding']['method']
    
    if categorical_encoding == 'onehot':
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
    elif categorical_encoding == 'ordinal':
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    elif categorical_encoding == 'none':
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='missing'))
        ])
    else:
        raise ValueError(f"지원하지 않는 범주형 인코딩 방법: {categorical_encoding}. 'onehot', 'ordinal', 'none' 중 선택하세요.")
    
    # 컬럼 변환기 생성 (placeholder로 시작)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, 'placeholder_numerical'),
            ('cat', categorical_transformer, 'placeholder_categorical')
        ],
        remainder='passthrough'  # 처리되지 않은 컬럼은 그대로 유지
    )
    
    logger.info(f"전처리 파이프라인 생성 완료")
    logger.info(f"  - 수치형 전략: {numerical_strategy}")
    logger.info(f"  - 범주형 전략: {categorical_strategy}")
    logger.info(f"  - 범주형 인코딩: {categorical_encoding}")
    
    return preprocessor


def fit_preprocessing_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """
    전처리 파이프라인을 피팅하고 데이터를 변환합니다.
    
    Args:
        df: 입력 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        (피팅된 전처리 파이프라인, 변환된 데이터프레임) 튜플
    """
    logger.info("전처리 파이프라인 피팅 시작")
    
    # 시계열 보간 적용
    df_imputed = apply_timeseries_imputation(df, config)
    
    # 수치형, 범주형, 통과 컬럼 식별
    numerical_cols = get_numerical_columns(df_imputed, config)
    categorical_cols = get_categorical_columns(df_imputed, config)
    passthrough_cols = get_passthrough_columns(df_imputed, config)
    
    # 전처리 파이프라인 생성
    preprocessor = create_preprocessing_pipeline(config)
    
    # 실제 컬럼으로 파이프라인 업데이트
    transformers = []
    
    if numerical_cols:
        # 수치형 전처리기 가져오기 (피팅 전이므로 transformers에서 가져옴)
        num_transformer = preprocessor.transformers[0][1]  # ('num', transformer, columns)
        transformers.append(('num', num_transformer, numerical_cols))
    
    if categorical_cols:
        # 범주형 전처리기 가져오기 (피팅 전이므로 transformers에서 가져옴)
        cat_transformer = preprocessor.transformers[1][1]  # ('cat', transformer, columns)
        transformers.append(('cat', cat_transformer, categorical_cols))
    
    if passthrough_cols:
        transformers.append(('pass', 'passthrough', passthrough_cols))
    
    # 새로운 파이프라인 생성
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'  # 처리되지 않은 컬럼은 그대로 유지
    )
    
    # 파이프라인 피팅
    logger.info("전처리 파이프라인 피팅 중...")
    preprocessor.fit(df_imputed)
    
    # 데이터 변환
    logger.info("데이터 변환 중...")
    transformed_data = transform_data(df_imputed, preprocessor, config)
    
    logger.info("전처리 파이프라인 피팅 완료")
    return preprocessor, transformed_data


def transform_data(df: pd.DataFrame, preprocessor: ColumnTransformer, config: Dict[str, Any]) -> pd.DataFrame:
    """
    전처리 파이프라인을 사용하여 데이터를 변환합니다.
    
    Args:
        df: 입력 데이터프레임
        preprocessor: 피팅된 전처리 파이프라인
        config: 설정 딕셔너리
        
    Returns:
        변환된 데이터프레임
    """
    # 파이프라인을 사용하여 데이터 변환
    transformed_array = preprocessor.transform(df)
    
    # 변환된 배열을 데이터프레임으로 변환
    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed_array, columns=feature_names, index=df.index)
    
    # 'remainder__'로 시작하는 컬럼명을 원래 이름으로 복원
    new_columns = []
    for col in transformed_df.columns:
        if col.startswith('remainder__'):
            new_columns.append(col.replace('remainder__', '', 1))
        else:
            new_columns.append(col)
    transformed_df.columns = new_columns
    
    # ID 컬럼 추가 (나중에 리샘플링에서 사용)
    id_column = config['time_series']['id_column']
    if id_column in df.columns and id_column not in transformed_df.columns:
        transformed_df[id_column] = df[id_column]
    
    logger.info(f"데이터 변환 완료: {transformed_df.shape}")
    return transformed_df


def apply_resampling(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any], 
                    fold_info: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    리샘플링을 적용합니다. 데이터 유출을 방지하기 위해 각 폴드에서 독립적으로 적용됩니다.
    
    Args:
        X: 피처 데이터프레임
        y: 타겟 시리즈
        config: 설정 딕셔너리
        fold_info: 폴드 정보 (MLflow 로깅용)
        
    Returns:
        리샘플링된 (X, y) 튜플
    """
    resampling_config = config.get('resampling', {})
    
    if not resampling_config.get('enabled', False):
        logger.info("리샘플링이 비활성화되어 있습니다.")
        return X, y
    
    # 리샘플링 파이프라인 생성
    resampling_pipeline = ResamplingPipeline(config)
    
    # ID 컬럼 확인
    id_column = config['time_series']['id_column']
    if id_column not in X.columns:
        logger.warning(f"ID 컬럼 '{id_column}'이 X에 없습니다. 리샘플링을 건너뜁니다.")
        return X, y
    
    # 리샘플링 적용
    logger.info("리샘플링 적용 중...")
    X_resampled, y_resampled = resampling_pipeline.fit_resample(X, y, id_column)
    
    # 클래스 분포 변화 로깅
    original_dist = y.value_counts().to_dict()
    resampled_dist = y_resampled.value_counts().to_dict()
    
    logger.info(f"리샘플링 전 클래스 분포: {original_dist}")
    logger.info(f"리샘플링 후 클래스 분포: {resampled_dist}")
    
    # MLflow 로깅 (폴드 정보가 있는 경우)
    if fold_info is not None:
        import mlflow
        fold_num = fold_info.get('fold_num', 'unknown')
        
        # 클래스 분포 변화 로깅
        for class_label, count in original_dist.items():
            mlflow.log_metric(f"fold_{fold_num}_original_class_{class_label}_count", count)
        
        for class_label, count in resampled_dist.items():
            mlflow.log_metric(f"fold_{fold_num}_resampled_class_{class_label}_count", count)
        
        # 리샘플링 파라미터 로깅
        method = resampling_config.get('method', 'unknown')
        mlflow.log_param(f"fold_{fold_num}_resampling_method", method)
        
        if method == 'smote':
            k_neighbors = resampling_config.get('smote_k_neighbors', 5)
            mlflow.log_param(f"fold_{fold_num}_smote_k_neighbors", k_neighbors)
        elif method == 'borderline_smote':
            k_neighbors = resampling_config.get('borderline_smote_k_neighbors', 5)
            mlflow.log_param(f"fold_{fold_num}_borderline_smote_k_neighbors", k_neighbors)
        elif method == 'adasyn':
            k_neighbors = resampling_config.get('adasyn_k_neighbors', 5)
            mlflow.log_param(f"fold_{fold_num}_adasyn_k_neighbors", k_neighbors)
    
    return X_resampled, y_resampled


def validate_preprocessing(df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """
    전처리 결과를 검증합니다.
    
    Args:
        df: 전처리된 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        검증 통과 여부
    """
    logger.info("전처리 결과 검증 중...")
    
    # 기본 검증
    if df.empty:
        logger.error("전처리된 데이터프레임이 비어있습니다.")
        return False
    
    # 결측치 검증
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"전처리 후에도 {missing_count}개의 결측치가 남아있습니다.")
    
    # 수치형 컬럼 검증
    numerical_cols = get_numerical_columns(df, config)
    for col in numerical_cols:
        if col in df.columns:
            if df[col].dtype not in ['int64', 'float64']:
                logger.warning(f"수치형 컬럼 {col}이 예상 타입이 아닙니다: {df[col].dtype}")
    
    # 범주형 컬럼 검증
    categorical_cols = get_categorical_columns(df, config)
    for col in categorical_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                logger.warning(f"범주형 컬럼 {col}이 인코딩되지 않았습니다.")
    
    logger.info("전처리 결과 검증 완료")
    return True


def main(config=None):
    """
    전처리 모듈 테스트
    """
    import yaml
    if config is None:
        print("[WARNING] config 인자가 전달되지 않았습니다. 외부에서 config를 넘겨주세요.")
        return
    # 샘플 데이터 생성 (테스트용)
    np.random.seed(42)
    n_samples = 1000
    ids = np.repeat(range(100), 10)
    dates = np.tile(pd.date_range('2020-01-01', periods=10, freq='M'), 100)
    anxiety_scores = np.random.normal(50, 15, n_samples)
    depress_scores = np.random.normal(45, 12, n_samples)
    sleep_scores = np.random.normal(60, 10, n_samples)
    comp_scores = np.random.normal(70, 8, n_samples)
    targets = np.random.choice([0, 1], size=n_samples, p=[0.999, 0.001])
    df = pd.DataFrame({
        'id': ids,
        'dov': dates,
        'yov': dates.year,
        'anxiety_score': anxiety_scores,
        'depress_score': depress_scores,
        'sleep_score': sleep_scores,
        'comp': comp_scores,
        'age': np.random.normal(45, 15, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'psychia_cate': np.random.choice(['A', 'B', 'C'], n_samples),
        'suicide_t': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'suicide_a': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'suicide_a_next_year': targets
    })
    print("원본 데이터 형태:", df.shape)
    print("클래스 분포:", df['suicide_a_next_year'].value_counts())
    # 리샘플링 테스트
    print("\n=== 리샘플링 테스트 ===")
    config['resampling']['enabled'] = True
    config['resampling']['method'] = 'smote'
    X = df.drop(['suicide_a_next_year'], axis=1)
    y = df['suicide_a_next_year']
    X_resampled, y_resampled = apply_resampling(X, y, config)
    print("SMOTE 적용 후 데이터 형태:", X_resampled.shape)
    print("SMOTE 적용 후 클래스 분포:", y_resampled.value_counts())
    print("\n리샘플링 테스트 완료!")


if __name__ == "__main__":
    main() 