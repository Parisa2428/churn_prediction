"""
اسکریپت اصلی اجرای پروژه پیش‌بینی ریزش مشتری
"""
import sys
import io
from loguru import logger
import joblib

# تنظیم encoding برای نمایش درست در ویندوز
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.config import config
from src.data.preprocessing import run_preprocessing, ChurnPreprocessor
from src.data.loader import load_dataset
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.explainability.shap_explainer import SHAPExplainer


def main() -> None:
    """تابع اصلی اجرای پروژه"""
    logger.info("=" * 70)
    logger.info("[START] Customer Churn Prediction Project")
    logger.info("=" * 70)
    
    # ==================== مرحله ۱: پیش‌پردازش ====================
    logger.info("\n" + "=" * 70)
    logger.info("[Step 1] Data Preprocessing")
    logger.info("=" * 70)
    
    X_train, X_test, y_train, y_test, feature_names = run_preprocessing()
    
    logger.success("[Step 1] Preprocessing completed successfully")
    logger.info(f"   X_train shape: {X_train.shape}")
    logger.info(f"   X_test shape: {X_test.shape}")
    logger.info(f"   Number of features: {len(feature_names)}")
    
    # ==================== مرحله ۲: آموزش مدل‌ها ====================
    logger.info("\n" + "=" * 70)
    logger.info("[Step 2] Training Machine Learning Models")
    logger.info("=" * 70)
    
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train, y_train)
    
    # ذخیره مدل‌ها
    trainer.save_models()
    
    # ذخیره preprocessor
    logger.info("[Saving] Saving preprocessor for API usage...")
    preprocessor = ChurnPreprocessor()
    df = load_dataset()
    preprocessor.fit_transform(df)
    joblib.dump(preprocessor.preprocessor, config.MODELS_DIR / "preprocessor.pkl")
    logger.success(f"[Saved] Preprocessor saved: {config.MODELS_DIR / 'preprocessor.pkl'}")
    
    # ==================== مرحله ۳: ارزیابی ====================
    logger.info("\n" + "=" * 70)
    logger.info("[Step 3] Model Evaluation")
    logger.info("=" * 70)
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test, y_test)
    
    # رسم نمودارها
    logger.info("\n[Plotting] Generating evaluation plots...")
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_metrics_comparison()
    
    # چاپ خلاصه نتایج
    logger.info("\n" + "=" * 70)
    logger.info("[Summary] Model Results")
    logger.info("=" * 70)
    
    for model_name, data in results.items():
        logger.info(f"\n{model_name}:")
        for metric, value in data["metrics"].items():
            logger.info(f"   - {metric}: {value:.4f}")
    
    # پیدا کردن بهترین مدل
    best_model_name = max(results, key=lambda k: results[k]["metrics"]["ROC-AUC"])
    best_auc = results[best_model_name]["metrics"]["ROC-AUC"]
    logger.success(f"\n[Best] Best model: {best_model_name}")
    logger.success(f"[Best] ROC-AUC: {best_auc:.4f}")
    
    # ==================== مرحله ۴: تحلیل SHAP ====================
    logger.info("\n" + "=" * 70)
    logger.info("[Step 4] SHAP Analysis (Model Interpretability)")
    logger.info("=" * 70)
    
    logger.info(f"\n[SHAP] Analyzing model: {best_model_name}...")
    explainer = SHAPExplainer(
        models[best_model_name], 
        feature_names, 
        X_train
    )
    
    explainer.plot_summary(X_test, best_model_name)
    explainer.plot_bar(X_test, best_model_name)
    explainer.get_top_features(X_test, top_n=10)
    
    # ==================== پایان ====================
    logger.info("\n" + "=" * 70)
    logger.success("[DONE] Project completed successfully!")
    logger.info("=" * 70)
    logger.info(f"[Output] Plots saved in: {config.REPORTS_DIR}")
    logger.info(f"[Output] Models saved in: {config.MODELS_DIR}")
    logger.info("\n[Next] To run API:")
    logger.info("   uvicorn src.api.main:app --reload")
    logger.info("\n[Next] To run UI:")
    logger.info("   streamlit run src/ui/app.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()