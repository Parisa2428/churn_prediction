"""
ماژول آموزش مدل‌ها با Hyperparameter Tuning
"""
from typing import Dict, Any
import joblib
from loguru import logger
import optuna
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import config


optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """کلاس آموزش و بهینه‌سازی مدل‌ها"""
    
    def __init__(self) -> None:
        self.best_models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict] = {}
    
    def train_logistic_regression(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> LogisticRegression:
        """آموزش Logistic Regression به عنوان Baseline"""
        logger.info("=" * 60)
        logger.info("[Training] Logistic Regression (Baseline)")
        logger.info("=" * 60)
        
        def objective(trial: optuna.Trial) -> float:
            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            max_iter = trial.suggest_int("max_iter", 500, 2000)
            
            if l1_ratio == 0.0:
                penalty = "l2"
            elif l1_ratio == 1.0:
                penalty = "l1"
            else:
                penalty = "elasticnet"
            
            solver = "saga" if penalty == "elasticnet" else "liblinear"
            
            model = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                l1_ratio=l1_ratio if penalty == "elasticnet" else None,
                max_iter=max_iter,
                random_state=config.RANDOM_STATE,
                class_weight="balanced"
            )
            
            cv = StratifiedKFold(
                n_splits=config.N_SPLITS, 
                shuffle=True, 
                random_state=config.RANDOM_STATE
            )
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv, scoring="roc_auc"
            )
            return scores.mean()
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        logger.success(f"[LogisticRegression] Best ROC-AUC: {study.best_value:.4f}")
        logger.info(f"[LogisticRegression] Best params: {study.best_params}")
        
        l1_ratio = study.best_params["l1_ratio"]
        if l1_ratio == 0.0:
            penalty = "l2"
        elif l1_ratio == 1.0:
            penalty = "l1"
        else:
            penalty = "elasticnet"
        
        solver = "saga" if penalty == "elasticnet" else "liblinear"
        
        best_model = LogisticRegression(
            C=study.best_params["C"],
            penalty=penalty,
            solver=solver,
            l1_ratio=l1_ratio if penalty == "elasticnet" else None,
            max_iter=study.best_params["max_iter"],
            random_state=config.RANDOM_STATE,
            class_weight="balanced"
        )
        best_model.fit(X_train, y_train)
        
        self.best_models["LogisticRegression"] = best_model
        self.best_params["LogisticRegression"] = study.best_params
        
        return best_model
    
    def train_random_forest(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> RandomForestClassifier:
        """آموزش Random Forest"""
        logger.info("=" * 60)
        logger.info("[Training] Random Forest")
        logger.info("=" * 60)
        
        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 100, 300)
            max_depth = trial.suggest_int("max_depth", 5, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=config.RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            )
            
            cv = StratifiedKFold(
                n_splits=config.N_SPLITS, 
                shuffle=True, 
                random_state=config.RANDOM_STATE
            )
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv, scoring="roc_auc"
            )
            return scores.mean()
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        logger.success(f"[RandomForest] Best ROC-AUC: {study.best_value:.4f}")
        logger.info(f"[RandomForest] Best params: {study.best_params}")
        
        best_model = RandomForestClassifier(
            **study.best_params,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        )
        best_model.fit(X_train, y_train)
        
        self.best_models["RandomForest"] = best_model
        self.best_params["RandomForest"] = study.best_params
        
        return best_model
    
    def train_xgboost(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> XGBClassifier:
        """آموزش XGBoost"""
        logger.info("=" * 60)
        logger.info("[Training] XGBoost")
        logger.info("=" * 60)
        
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos
        logger.info(f"[XGBoost] scale_pos_weight: {scale_pos_weight:.2f}")
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": scale_pos_weight,
                "random_state": config.RANDOM_STATE,
                "eval_metric": "logloss",
            }
            
            model = XGBClassifier(**params)
            
            cv = StratifiedKFold(
                n_splits=config.N_SPLITS, 
                shuffle=True, 
                random_state=config.RANDOM_STATE
            )
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv, scoring="roc_auc"
            )
            return scores.mean()
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        
        logger.success(f"[XGBoost] Best ROC-AUC: {study.best_value:.4f}")
        logger.info(f"[XGBoost] Best params: {study.best_params}")
        
        best_params = study.best_params.copy()
        best_params["scale_pos_weight"] = scale_pos_weight
        best_params["random_state"] = config.RANDOM_STATE
        best_params["eval_metric"] = "logloss"
        
        best_model = XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        self.best_models["XGBoost"] = best_model
        self.best_params["XGBoost"] = study.best_params
        
        return best_model
    
    def train_all_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """آموزش تمام مدل‌ها"""
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        return self.best_models
    
    def save_models(self) -> None:
        """ذخیره مدل‌های آموزش‌دیده"""
        for name, model in self.best_models.items():
            path = config.MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, path)
            logger.success(f"[Saved] Model saved: {path}")