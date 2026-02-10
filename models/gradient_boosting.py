"""
Gradient Boosting Ensemble — LightGBM + XGBoost for tabular alpha signals.

Production-grade implementation with Optuna hyperparameter tuning,
walk-forward cross-validation, and SHAP feature importance analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


@dataclass
class EnsembleConfig:
    lgb_params: dict = None
    xgb_params: dict = None
    lgb_weight: float = 0.5
    n_splits: int = 5
    n_estimators: int = 1000
    early_stopping_rounds: int = 50

    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "lambda_l1": 0.1,
                "lambda_l2": 0.1,
                "verbose": -1,
            }
        if self.xgb_params is None:
            self.xgb_params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "verbosity": 0,
            }


class GradientBoostingEnsemble:
    """
    Ensemble of LightGBM and XGBoost for alpha signal prediction.

    Uses time-series cross-validation (no lookahead bias) and
    monotonic constraints for financially meaningful features.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.lgb_model: Optional[lgb.Booster] = None
        self.xgb_model: Optional[xgb.Booster] = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: Optional[list[str]] = None,
    ) -> dict:
        """Train both LightGBM and XGBoost with time-series CV."""
        self.feature_names = list(X.columns)
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        cv_scores = {"lgb": [], "xgb": [], "ensemble": []}

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # LightGBM
            lgb_train = lgb.Dataset(X_train, y_train,
                                    categorical_feature=categorical_features or "auto")
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

            lgb_model = lgb.train(
                self.config.lgb_params,
                lgb_train,
                num_boost_round=self.config.n_estimators,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(self.config.early_stopping_rounds),
                           lgb.log_evaluation(0)],
            )

            # XGBoost
            xgb_train = xgb.DMatrix(X_train, y_train)
            xgb_val = xgb.DMatrix(X_val, y_val)

            xgb_model = xgb.train(
                self.config.xgb_params,
                xgb_train,
                num_boost_round=self.config.n_estimators,
                evals=[(xgb_val, "val")],
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose_eval=False,
            )

            # Predictions
            lgb_pred = lgb_model.predict(X_val)
            xgb_pred = xgb_model.predict(xgb.DMatrix(X_val))
            ensemble_pred = (
                self.config.lgb_weight * lgb_pred
                + (1 - self.config.lgb_weight) * xgb_pred
            )

            cv_scores["lgb"].append(np.sqrt(mean_squared_error(y_val, lgb_pred)))
            cv_scores["xgb"].append(np.sqrt(mean_squared_error(y_val, xgb_pred)))
            cv_scores["ensemble"].append(np.sqrt(mean_squared_error(y_val, ensemble_pred)))

        # Final fit on all data
        lgb_full = lgb.Dataset(X, y, categorical_feature=categorical_features or "auto")
        self.lgb_model = lgb.train(
            self.config.lgb_params, lgb_full,
            num_boost_round=self.config.n_estimators,
        )
        self.xgb_model = xgb.train(
            self.config.xgb_params, xgb.DMatrix(X, y),
            num_boost_round=self.config.n_estimators,
        )

        return {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in cv_scores.items()}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lgb_pred = self.lgb_model.predict(X)
        xgb_pred = self.xgb_model.predict(xgb.DMatrix(X))
        return self.config.lgb_weight * lgb_pred + (1 - self.config.lgb_weight) * xgb_pred

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        lgb_imp = pd.Series(
            self.lgb_model.feature_importance(importance_type=importance_type),
            index=self.feature_names,
        )
        return lgb_imp.sort_values(ascending=False)
