"""
Dataset loader module
Responsible for downloading and loading the Telco Customer Churn dataset.
"""
import pandas as pd
from loguru import logger

from src.config import config


def load_dataset() -> pd.DataFrame:
    """
    Load dataset from local file or URL.
    First checks if file exists in data/raw/, otherwise downloads from internet.
    """
    local_path = config.DATA_RAW_DIR / "Telco-Customer-Churn.csv"
    
    if local_path.exists():
        logger.info(f"[Loading] Dataset from local path: {local_path}")
        df = pd.read_csv(local_path)
    else:
        logger.info(f"[Download] Dataset from: {config.DATASET_URL}")
        df = pd.read_csv(config.DATASET_URL)
        df.to_csv(local_path, index=False)
        logger.success(f"[Saved] Dataset saved to: {local_path}")
    
    logger.info(f"[Info] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"[Info] Columns: {list(df.columns)}")
    
    return df


def display_dataset_info(df: pd.DataFrame) -> None:
    """Display complete dataset information."""
    logger.info("=" * 60)
    logger.info("[Dataset Info]")
    logger.info("=" * 60)
    
    logger.info(f"Number of samples: {len(df)}")
    logger.info(f"Number of features: {df.shape[1]}")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    logger.info("\n[Missing Values]:")
    if missing.sum() == 0:
        logger.info("No missing values found")
    else:
        for col in df.columns:
            if missing[col] > 0:
                logger.info(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
    
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    
    logger.info("\n[Target Distribution] (Churn):")
    for value, count in churn_counts.items():
        logger.info(f"  {value}: {count} ({churn_pct[value]:.2f}%)")
    
    logger.info("\n[First 5 Samples]:")
    logger.info(f"\n{df.head()}")


if __name__ == "__main__":
    logger.info("[Test] Loading dataset")
    df = load_dataset()
    display_dataset_info(df)
    