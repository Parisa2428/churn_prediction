"""
تنظیمات سراسری پروژه پیش‌بینی ریزش مشتری
این فایل مسیرها و پارامترهای ثابت پروژه رو تعریف می‌کنه
"""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """کلاس تنظیمات سراسری پروژه"""
    
    # ==================== مسیرها ====================
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports" / "figures"
    
    # ==================== دیتاست ====================
    DATASET_URL: str = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )
    TARGET_COLUMN: str = "Churn"
    ID_COLUMN: str = "customerID"
    
    # ==================== آموزش مدل ====================
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_SPLITS: int = 5
    
    # ==================== کلاس‌ها ====================
    POSITIVE_CLASS: str = "Yes"
    NEGATIVE_CLASS: str = "No"


# ساخت یک نمونه از کلاس تنظیمات
config = Config()


# ساخت خودکار دایرکتوری‌ها (اگه وجود ندارن)
for directory in [
    config.DATA_RAW_DIR,
    config.DATA_PROCESSED_DIR,
    config.MODELS_DIR,
    config.REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)