import os
import sys

import dill
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as exc:
        raise CustomException(exc, sys) from exc


def evaluate_models(X_train, y_train, models, param, preprocessor):
    """Compare estimators with fold-local preprocessing on training data only.

    Each GridSearchCV candidate is a Pipeline, so the preprocessor is cloned and
    fitted only on that fold's training rows. The returned best estimators are
    refitted on the complete training partition. No holdout data is accepted.
    """
    try:
        report = {}
        best_estimators = {}

        for model_name, model in models.items():
            estimator = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", model),
                ]
            )
            prefixed_params = {
                f"model__{parameter}": values
                for parameter, values in param[model_name].items()
            }
            search = GridSearchCV(
                estimator=estimator,
                param_grid=prefixed_params,
                cv=3,
                scoring="r2",
                refit=True,
            )
            search.fit(X_train, y_train)

            report[model_name] = float(search.best_score_)
            best_estimators[model_name] = search.best_estimator_

        models.update(best_estimators)
        return report

    except Exception as exc:
        raise CustomException(exc, sys) from exc


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as exc:
        raise CustomException(exc, sys) from exc
