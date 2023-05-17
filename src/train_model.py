"""Python script to train the model"""
from typing import TypeVar

import joblib
import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.base import BaseEstimator

from config import Location, ModelParams
from finetune import finetune_search
from model_utils import initialise_model


@task
def get_processed_data(data_location: str):
    """Get processed data from a specified location

    Parameters
    ----------
    data_location : str
        Location to get the data
    """
    with open(data_location, "rb") as file:
        processed_data = joblib.load(file)
    return processed_data


@flow
def train_model(
    model_name: str,
    model_params: dict,
    X_train: TypeVar("pandas.core.frame.DataFrame"),  # noqa: F821
    y_train: TypeVar("pandas.core.series.Series"),  # noqa: F821
):
    """Train the model

    Parameters
    ----------
    model_name: str
        name of the model to train
    model_params : dict
        Parameters for the model
    X_train : pd.DataFrame
        Features for training
    y_train : pd.Series
        Label for training
    """
    model = initialise_model(model_name, model_params)
    model.fit(X_train, y_train)

    return model


@task
def predict(model: BaseEstimator, X_test: pd.DataFrame):
    """_summary_

    Parameters
    ----------
    model : estimator
    X_test : pd.DataFrame
        Features for testing
    """
    return model.predict(X_test)


@task
def evaluate_model(
    model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series
):
    """
    Evaluate the model accuracy performance
    Parameters
    ----------
    model: BaseEstimtor
        fitted model
    X_test : pd.DataFrame
        Features for testing
    y_test: pd.Series
        Label for testing
    """
    return model.score(X_test, y_test)
    # return metrics.accuracy_score(y_test, model.predict(X_test), normalize=True)


@task
def save_model(model: BaseEstimator, save_path: str):
    """Save model to a specified location

    Parameters
    ----------
    model : BaseEstimator
    save_path : str
    """
    joblib.dump(model, save_path)


@task
def save_predictions(predictions: np.array, save_path: str):
    """Save predictions to a specified location

    Parameters
    ----------
    predictions : np.array
    save_path : str
    """
    joblib.dump(predictions, save_path)


@flow
def train(
    location: Location = Location(),
    model_params: ModelParams = ModelParams(),
):
    """Flow to train the model

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    model_params : ModelParams, optional
        Configurations for training the model, by default ModelParams()
    """
    train_data = get_processed_data(location.train_data)
    X_train, X_test, y_train, y_test = (
        train_data["X_train"],
        train_data["X_test"],
        train_data["y_train"],
        train_data["y_test"],
    )

    if model_params.finetune.enable:
        best_model_name, best_model_params = finetune_search(
            X_train, y_train, model_params
        )
        model = train_model(
            best_model_name, best_model_params, X_train, y_train
        )
    else:
        model = train_model(
            model_params.model_name,
            model_params.model_params,
            X_train,
            y_train,
        )

    predictions = predict(model, X_test)
    performance = evaluate_model(model, X_test, y_test)
    print(performance)
    save_model(model, save_path=location.model)
    save_predictions(predictions, save_path=location.data_final)


if __name__ == "__main__":
    train()
