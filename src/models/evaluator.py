"""
ماژول ارزیابی مدل‌ها
شامل محاسبه متریک‌ها و رسم نمودارها
"""
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    roc_curve, classification_report
)

from src.config import config


class ModelEvaluator:
    """کلاس ارزیابی مدل‌ها"""
    
    def __init__(self) -> None:
        self.results: Dict[str, Dict] = {}
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """ارزیابی یک مدل"""
        logger.info(f"\n{'='*60}")
        logger.info(f"[Evaluation] Model: {model_name}")
        logger.info(f"{'='*60}")
        
        # پیش‌بینی
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # محاسبه متریک‌ها
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
        }
        
        # چاپ گزارش
        logger.info("\n" + classification_report(
            y_test, y_pred, 
            target_names=["No Churn", "Churn"]
        ))
        
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        self.results[model_name] = {
            "metrics": metrics,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_test": y_test,
        }
        
        return metrics
    
    def evaluate_all_models(
        self,
        models: Dict,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """ارزیابی تمام مدل‌ها"""
        for name, model in models.items():
            self.evaluate_model(model, X_test, y_test, name)
        return self.results
    
    def plot_confusion_matrices(self) -> None:
        """رسم Confusion Matrix برای تمام مدل‌ها"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, data) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = confusion_matrix(data["y_test"], data["y_pred"])
            
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"]
            )
            ax.set_title(f"{model_name}\nConfusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        
        plt.tight_layout()
        path = config.REPORTS_DIR / "confusion_matrices.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.success(f"[Saved] Confusion matrices: {path}")
        plt.show()
    
    def plot_roc_curves(self) -> None:
        """رسم ROC Curve برای تمام مدل‌ها"""
        plt.figure(figsize=(10, 7))
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        
        for idx, (model_name, data) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(data["y_test"], data["y_prob"])
            auc = data["metrics"]["ROC-AUC"]
            
            plt.plot(
                fpr, tpr, 
                color=colors[idx % len(colors)],
                lw=2.5,
                label=f"{model_name} (AUC = {auc:.4f})"
            )
        
        plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve - Model Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        path = config.REPORTS_DIR / "roc_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.success(f"[Saved] ROC curves: {path}")
        plt.show()
    
    def plot_metrics_comparison(self) -> None:
        """رسم مقایسه متریک‌ها بین مدل‌ها"""
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        model_names = list(self.results.keys())
        
        data = {
            metric: [self.results[m]["metrics"][metric] for m in model_names]
            for metric in metrics_names
        }
        
        x = np.arange(len(model_names))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics_names):
            ax.bar(
                x + i * width, data[metric], width, 
                label=metric, alpha=0.85
            )
        
        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        path = config.REPORTS_DIR / "metrics_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.success(f"[Saved] Metrics comparison: {path}")
        plt.show()