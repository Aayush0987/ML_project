import os
import pandas as pd
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving object: {e}", sys) from e


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        fitted_models = {}  # store fitted best models

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            base_model = list(models.values())[i]
            para = params.get(model_name, {})  # safer

            # GridSearchCV handles fitting
            gs = GridSearchCV(base_model, para, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_  # already fitted
            fitted_models[model_name] = best_model

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
        
        return report, fitted_models  # return both

    except Exception as e: 
        raise CustomException(e, sys)


def load_object(file_path):                 
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)