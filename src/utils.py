import os
import sys

import dill
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, models, param):
    """Select model configurations using cross-validation on training data only.

    The returned scores are mean cross-validation scores. The supplied model
    mapping is updated with each GridSearchCV best estimator, refitted on the
    complete training partition. A separate holdout set is intentionally not
    accepted here so it cannot influence model or hyperparameter selection.
    """
    try:
        report = {}
        best_estimators = {}

        for model_name, model in models.items():
            search = GridSearchCV(
                estimator=model,
                param_grid=param[model_name],
                cv=3,
                scoring="r2",
            )
            search.fit(X_train, y_train)

            report[model_name] = float(search.best_score_)
            best_estimators[model_name] = search.best_estimator_

        models.update(best_estimators)
        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
