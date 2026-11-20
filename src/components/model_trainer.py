import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object, select_best_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test, preprocessor):
        try:
            models = {
                "RandomForest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False, random_seed=42
                ),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "RandomForest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            logging.info(
                "Selecting models with fold-local preprocessing and 3-fold CV"
            )
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                models=models,
                param=params,
                preprocessor=preprocessor,
            )
            best_model_name, best_cv_score = select_best_model(model_report)
            best_model = models[best_model_name]
            logging.info(
                "Selected %s using 3-fold training CV (mean R2=%.4f)",
                best_model_name,
                best_cv_score,
            )

            # The serialized estimator includes its fitted preprocessing pipeline.
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # The holdout is first accessed after all model selection is complete.
            predicted = best_model.predict(X_test)
            holdout_r2 = r2_score(y_test, predicted)
            logging.info("Final holdout R2 after model selection: %.4f", holdout_r2)
            return holdout_r2

        except Exception as exc:
            raise CustomException(exc, sys) from exc
