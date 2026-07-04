"""
ماژول بارگذاری دیتاست
این فایل مسئول دانلود و بارگذاری دیتاست Telco Customer Churn هست
"""
import pandas as pd
from loguru import logger

from src.config import config


def load_dataset() -> pd.DataFrame:
    """
    بارگذاری دیتاست از فایل محلی یا URL
    
    اول چک می‌کنه اگه فایل توی data/raw/ وجود داره، از همون بخونه.
    اگه نه، از اینترنت دانلود می‌کنه و ذخیره می‌کنه.
    
    Returns:
        pd.DataFrame: دیتاست بارگذاری‌شده
    """
    local_path = config.DATA_RAW_DIR / "Telco-Customer-Churn.csv"
    
    # چک کردن وجود فایل محلی
    if local_path.exists():
        logger.info(f"📂 بارگذاری دیتاست از مسیر محلی: {local_path}")
        df = pd.read_csv(local_path)
    else:
        # دانلود از اینترنت
        logger.info(f"🌐 دانلود دیتاست از: {config.DATASET_URL}")
        df = pd.read_csv(config.DATASET_URL)
        
        # ذخیره در مسیر محلی برای استفاده‌های بعدی
        df.to_csv(local_path, index=False)
        logger.success(f"💾 دیتاست ذخیره شد در: {local_path}")
    
    # نمایش اطلاعات اولیه
    logger.info(f"📊 ابعاد دیتاست: {df.shape[0]} سطر، {df.shape[1]} ستون")
    logger.info(f"📋 ستون‌ها: {list(df.columns)}")
    
    return df


def display_dataset_info(df: pd.DataFrame) -> None:
    """
    نمایش اطلاعات کامل دیتاست
    
    Args:
        df: دیتاست
    """
    logger.info("=" * 60)
    logger.info("📊 اطلاعات دیتاست")
    logger.info("=" * 60)
    
    # اطلاعات پایه
    logger.info(f"تعداد نمونه‌ها: {len(df)}")
    logger.info(f"تعداد ویژگی‌ها: {df.shape[1]}")
    
    # انواع داده
    logger.info("\n🔢 انواع داده:")
    logger.info(f"{df.dtypes.value_counts()}")
    
    # مقادیر گمشده
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    logger.info("\n❌ مقادیر گمشده:")
    if missing.sum() == 0:
        logger.info("هیچ مقدار گمشده‌ای وجود نداره ✅")
    else:
        for col in df.columns:
            if missing[col] > 0:
                logger.info(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
    
    # توزیع متغیر هدف
    logger.info("\n🎯 توزیع متغیر هدف (Churn):")
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    
    for value, count in churn_counts.items():
        logger.info(f"  {value}: {count} ({churn_pct[value]:.2f}%)")
    
    # ۵ نمونه اول
    logger.info("\n👀 ۵ نمونه اول دیتاست:")
    logger.info(f"\n{df.head()}")


if __name__ == "__main__":
    # تست ماژول
    logger.info("🚀 تست بارگذاری دیتاست")
    df = load_dataset()
    display_dataset_info(df)