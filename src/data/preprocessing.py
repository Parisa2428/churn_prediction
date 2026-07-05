"""
ماژول پیش‌پردازش داده
شامل: مدیریت missing values، encoding، scaling و handling imbalance
"""
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from src.config import config


class ChurnPreprocessor:
    """کلاس پیش‌پردازش کامل داده ریزش مشتری"""
    
    def __init__(self) -> None:
        self.preprocessor: ColumnTransformer | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list = []
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        مدیریت مقادیر گمشده
        - TotalCharges: تبدیل به عدد و پر کردن با median
        """
        logger.info("[Preprocessing] Handling missing values...")
        
        # تبدیل TotalCharges به عددی
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
        # بررسی missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        logger.info(f"[Missing] Percentage:\n{missing_pct[missing_pct > 0]}")
        
        # پر کردن TotalCharges با median
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        
        logger.success("[Missing] Handling completed")
        return df
    
    def identify_columns(self, df: pd.DataFrame) -> Tuple[list, list, list]:
        """شناسایی ستون‌های عددی، دسته‌ای دوتایی و چندمقداری"""
        target = config.TARGET_COLUMN
        id_col = config.ID_COLUMN
        
        # حذف ستون‌های غیرضروری
        drop_cols = [id_col, target]
        df_features = df.drop(columns=drop_cols)
        
        # دسته‌بندی ستون‌ها
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_features.select_dtypes(include=["object"]).columns.tolist()
        
        # ستون‌های دوتایی
        binary_cols = [
            col for col in categorical_cols 
            if df_features[col].nunique() == 2
        ]
        
        # ستون‌های چندمقداری
        multi_cols = [
            col for col in categorical_cols 
            if df_features[col].nunique() > 2
        ]
        
        logger.info(f"[Columns] Numeric ({len(numeric_cols)}): {numeric_cols}")
        logger.info(f"[Columns] Binary ({len(binary_cols)}): {binary_cols}")
        logger.info(f"[Columns] Multi-value ({len(multi_cols)}): {multi_cols}")
        
        return numeric_cols, binary_cols, multi_cols
    
    def build_preprocessor(
        self, 
        numeric_cols: list, 
        binary_cols: list, 
        multi_cols: list
    ) -> ColumnTransformer:
        """ساخت pipeline پیش‌پردازش"""
        
        # pipeline برای ستون‌های عددی
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        # pipeline برای ستون‌های دوتایی
        binary_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False))
        ])
        
        # pipeline برای ستون‌های چندمقداری
        multi_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        # ساخت preprocessor نهایی
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("bin", binary_transformer, binary_cols),
            ("cat", multi_transformer, multi_cols)
        ])
        
        return preprocessor
    
    def fit_transform(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """اجرای کامل پیش‌پردازش روی داده"""
        logger.info("=" * 60)
        logger.info("[Preprocessing] Starting data preprocessing")
        logger.info("=" * 60)
        
        # ۱. مدیریت missing values
        df = self.handle_missing_values(df)
        
        # ۲. جداسازی features و target
        X = df.drop(columns=[config.ID_COLUMN, config.TARGET_COLUMN])
        y = (df[config.TARGET_COLUMN] == config.POSITIVE_CLASS).astype(int)
        
        logger.info(f"[Class Distribution] Before SMOTE:")
        logger.info(f"  No Churn (0): {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
        logger.info(f"  Churn (1): {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")
        
        # ۳. شناسایی ستون‌ها
        numeric_cols, binary_cols, multi_cols = self.identify_columns(df)
        
        # ۴. ساخت preprocessor
        self.preprocessor = self.build_preprocessor(
            numeric_cols, binary_cols, multi_cols
        )
        
        # ۵. train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        logger.info(f"[Split] Train: {X_train.shape}, Test: {X_test.shape}")
        
        # ۶. fit و transform روی train
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # ۷. استخراج نام ویژگی‌ها با استفاده از روش مطمئن sklearn
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        logger.info(f"[Features] Total features after encoding: {len(self.feature_names)}")
        
        # ۸. SMOTE فقط روی train
        logger.info("[SMOTE] Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(
            X_train_processed, y_train
        )
        
        logger.success(f"[SMOTE] After SMOTE - Train: {X_train_balanced.shape}")
        logger.info(f"[Class Distribution] After SMOTE:")
        logger.info(f"  No Churn (0): {(y_train_balanced == 0).sum()}")
        logger.info(f"  Churn (1): {(y_train_balanced == 1).sum()}")
        
        return (
            X_train_balanced, 
            X_test_processed, 
            y_train_balanced, 
            y_test
        )


def run_preprocessing() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """تابع اصلی اجرای پیش‌پردازش"""
    from src.data.loader import load_dataset
    
    df = load_dataset()
    preprocessor = ChurnPreprocessor()
    
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    return X_train, X_test, y_train, y_test, preprocessor.feature_names


if __name__ == "__main__":
    # Test module
    logger.info("[Test] Testing preprocessing")
    X_train, X_test, y_train, y_test, feature_names = run_preprocessing()
    
    logger.info(f"\n[Dimensions]:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  y_test: {y_test.shape}")
    logger.info(f"  Number of features: {len(feature_names)}")
    logger.info(f"\n[First 10 Features]:")
    for i, name in enumerate(feature_names[:10], 1):
        logger.info(f"  {i}. {name}")