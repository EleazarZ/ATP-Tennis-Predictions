"""Python script to finetune the model"""
from functools import partial
from typing import TypeVar, Union

import optuna
from prefect import flow, task
from sklearn.model_selection import cross_val_score

from config import ModelParams
from model_utils import initialise_model


@task
def init_sampler(
    sampler_name: str, sampler_params: dict, grids: dict
) -> optuna.samplers:
    """
    Initialise a sampler object for the fine tuning
    Parameters
    ----------
    sampler_name: name of sampler. Available list: 'bayesian_search', 'grid_search', 'random_search', 'genetic_search']
    sampler_params: parameters of the sampler object
    grids: grid of model hyperparameters to fine tune
    returns
    -------
    sampler: A sampler object
    """
    available_samplers = {
        "bayesian_search": optuna.samplers.TPESampler,
        "grid_search": optuna.samplers.GridSampler,
        "random_search": optuna.samplers.RandomSampler,
        "genetic_search": optuna.samplers.NSGAIISampler,
    }
    if sampler_name == "grid_search":
        sampler_params["search_space"] = grids

    sampler = available_samplers.get(sampler_name)(**sampler_params)
    return sampler


@flow
def objective(
    trial: TypeVar("optuna.Trial"),
    X_train: TypeVar("pandas.core.frame.DataFrame"),  # noqa: F821
    y_train: TypeVar("pandas.core.series.Series"),  # noqa: F821
    grids: dict,
    n_folds: int,
) -> float:
    """
    Objective function to optimize

    Parameters
    ----------
    trial: optuna trial
    X_train: dataframe with the feaures
    y_train: dataframe with the true labels
    grids: grid of the model hyperparameters
    n_fols: number of folds to use in the ccross-validation
    Returns
    -------
    cv_score: the mean score of the cross folds
    """

    model_names = list(grids.keys())

    model_name = trial.suggest_categorical("model_name", model_names)
    model_params_grid = grids.get(model_name)

    params = define_parameter_search_space(
        trial, model_name, model_params_grid
    )

    model = initialise_model(model_name=model_name, model_params=params)

    cv_score = cross_val_score(model, X_train, y_train, cv=n_folds).mean()
    return cv_score


@task
def define_parameter_search_space(
    trial: optuna.Trial, model_name: str, model_params_grid: dict
) -> dict:
    """
    Create an hyperparameter space based on the provided grid
    Parameters
    ----------
    trial: optuna trial
    model_name: name of the model to instantiate
    model_params: Configurations for training the model, by default ModelParams()
    Returns
    -------
    params: dictionary of hyperparameters to evaluate
    """
    params = {}
    for param_name, param_values in model_params_grid.items():
        # Categorical parameter
        if max([isinstance(elt, (str, bool)) for elt in param_values]):
            parameter = trial.suggest_categorical(
                f"{model_name}-{param_name}", param_values
            )
        else:
            # Floating point parameter
            if max([isinstance(elt, float) for elt in param_values]) and sum(
                [isinstance(elt, (float, int)) for elt in param_values]
            ) == len(param_values):
                parameter = trial.suggest_float(
                    f"{model_name}-{param_name}",
                    min(param_values),
                    max(param_values),
                )
            # Integer parameter
            elif sum([isinstance(elt, int) for elt in param_values]) == len(
                param_values
            ):
                parameter = trial.suggest_int(
                    f"{model_name}-{param_name}",
                    min(param_values),
                    max(param_values),
                )
            else:
                parameter = trial.suggest_categorical(
                    f"{model_name}-{param_name}", param_values
                )

        params[param_name] = parameter
    return params


@flow
def finetune_search(
    X_train: TypeVar("pandas.core.frame.DataFrame"),  # noqa: F821
    y_train: TypeVar("pandas.core.series.Series"),  # noqa: F821
    model_params: ModelParams = ModelParams(),
) -> Union[str, dict]:
    """
    Fine tune

    Parameters
    ----------
    X_train: dataframe with the feaures
    y_train: dataframe with the true labels
    model_params: ModelParams
    Returns
    -------
    best_model_name, best_model_params: name of best model and params fine tuned
    """

    # Define finetune grid

    if model_params.finetune.sampler.name == "grid_search":
        gridsearch_grids = {
            "model_name": [
                elt["model_name"] for elt in model_params.finetune.grids
            ]
        }
        for elt in model_params.finetune.grids:
            model_name = elt["model_name"]
            for k, v in elt["grid"].items():
                gridsearch_grids[f"{model_name}-{k}"] = v
    else:
        gridsearch_grids = None

    grids = {
        elt["model_name"]: elt["grid"] for elt in model_params.finetune.grids
    }

    # Initialise optuna sampler
    sampler = init_sampler(
        sampler_name=model_params.finetune.sampler.name,
        sampler_params=model_params.finetune.sampler.params,
        grids=gridsearch_grids,
    )

    # Optimize
    study = optuna.create_study(
        direction=model_params.finetune.direction, sampler=sampler
    )

    study.optimize(
        partial(
            objective,
            X_train=X_train,
            y_train=y_train,
            grids=grids,
            n_folds=model_params.finetune.n_folds,
        ),
        n_trials=model_params.finetune.n_trials,
        show_progress_bar=True,
    )

    # Finding best performance and params
    # trials_dataframe = study.trials_dataframe()
    # best_perf = study.best_value
    best_params = study.best_params

    best_model_name = best_params.get("model_name")
    best_params.pop("model_name", None)
    best_model_params = {
        k.replace(f"{best_model_name}-", ""): v for k, v in best_params.items()
    }
    return best_model_name, best_model_params
