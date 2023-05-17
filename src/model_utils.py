"""Python script to initiate an estimator object"""
from prefect import task
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@task
def initialise_model(model_name: str, model_params: dict):
    """
    Initialize a model based on the selected type
    Parameters
    ----------
    model_name: name of the model to instantiate
    model_params: Configurations for the model
    Returns
    -------
    scikit-learn estimator object
    """

    available_models = {
        # Logistic regression
        "LR": LogisticRegression(random_state=42),
        # Random Forest classifier
        "RFC": RandomForestClassifier(random_state=42, n_jobs=-1),
        # XGBoost classifier
        "XGC": XGBClassifier(
            random_state=42, eval_metric="logloss", n_jobs=-1
        ),
    }

    return available_models.get(model_name).set_params(**model_params)
