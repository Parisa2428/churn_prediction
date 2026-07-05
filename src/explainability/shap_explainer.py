"""
ماژول تفسیر مدل با SHAP
برای فهمیدن اهمیت ویژگی‌ها و توضیح پیش‌بینی‌ها
"""
import numpy as np
import matplotlib.pyplot as plt
import shap
from loguru import logger

from src.config import config


class SHAPExplainer:
    """کلاس تفسیر مدل با SHAP"""
    
    def __init__(self, model, feature_names: list, X_background: np.ndarray):
        self.model = model
        self.feature_names = feature_names
        self.X_background = X_background
        self.explainer = None
        self._init_explainer()
    
    def _init_explainer(self) -> None:
        """انتخاب explainer مناسب بر اساس نوع مدل"""
        model_name = self.model.__class__.__name__
        
        if model_name in ["XGBClassifier", "XGBRFClassifier"]:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("[SHAP] Using TreeExplainer for XGBoost")
        elif model_name in ["RandomForestClassifier", "GradientBoostingClassifier"]:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("[SHAP] Using TreeExplainer for Tree-based model")
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.kmeans(self.X_background, 50)
            )
            logger.info("[SHAP] Using KernelExplainer for linear model")
    
    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """محاسبه SHAP values"""
        logger.info(f"[SHAP] Computing SHAP values for {X.shape[0]} samples...")
        
        model_name = self.model.__class__.__name__
        
        if model_name in ["XGBClassifier", "RandomForestClassifier"]:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values_all = self.explainer.shap_values(X)
            shap_values = shap_values_all[..., 1]
        
        return shap_values
    
    def plot_summary(self, X: np.ndarray, model_name: str) -> None:
        """رسم Summary Plot (Beeswarm)"""
        shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.feature_names,
            show=False,
            max_display=15
        )
        plt.title(
            f"SHAP Summary Plot - {model_name}", 
            fontsize=14, fontweight="bold"
        )
        
        path = config.REPORTS_DIR / f"shap_summary_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.success(f"[Saved] SHAP summary plot: {path}")
        plt.show()
    
    def plot_bar(self, X: np.ndarray, model_name: str) -> None:
        """رسم Bar Plot (میانگین |SHAP|)"""
        shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
            max_display=15
        )
        plt.title(
            f"Feature Importance (Mean |SHAP|) - {model_name}", 
            fontsize=14, fontweight="bold"
        )
        
        path = config.REPORTS_DIR / f"shap_bar_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.success(f"[Saved] SHAP bar plot: {path}")
        plt.show()
    
    def get_top_features(self, X: np.ndarray, top_n: int = 10) -> list:
        """دریافت مهم‌ترین ویژگی‌ها"""
        shap_values = self.compute_shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        top_features = [
            (self.feature_names[i], mean_abs_shap[i]) 
            for i in top_indices
        ]
        
        logger.info(f"\n[SHAP] Top {top_n} features:")
        for i, (feat, score) in enumerate(top_features, 1):
            logger.info(f"  {i}. {feat}: {score:.4f}")
        
        return top_features