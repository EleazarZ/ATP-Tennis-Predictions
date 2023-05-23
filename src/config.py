"""
create Pydantic models
"""
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, validator


class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    data_raw: str = "data/raw/atp_matches_till_2022.csv"
    data_process: str = "data/processed/processed_data.csv"
    processor: str = "models/processor.pkl"
    config: str = "models/config.pkl"
    featurizer: str = "models/featurizer.pkl"
    train_data: str = "data/final/train_data.pkl"
    model: str = "models/model.pkl"
    data_final: str = "data/final/prediction.pkl"


def float_value_validation(value: float) -> float:
    """Check if the value is between zero and one

    Parameters
    ----------
    value : float
        value

    Returns
    -------
    float
        value

    Raises
    ------
    ValueError
        Raises error when value is negative or above one
    """
    if (value is not None) and (value > 1 or value < 0):
        raise ValueError(f"{value} must be between zero and one")
    return value


class ProcessConfig(BaseModel):
    """Specify the parameters of the `process` flow"""

    select_condition_expression: str = "df['tourney_date'].dt.year>=1991"
    missing_data_threshold_by_column: float = 0.4
    missing_data_threshold_by_row: float = 0.4
    drop_duplicates_subset_columns: list = None

    sample_size: float = 0.001
    random_state: int = 42
    date_columns: List[dict] = [{"name": "tourney_date", "format": "%Y%m%d"}]
    n_hist_matches: int = 5
    n_hist_confront: int = 3

    test_size: float = 0.2

    _validated_sample_size = validator("sample_size", allow_reuse=True)(
        float_value_validation
    )

    _validated_missing_data_threshold_by_column = validator(
        "missing_data_threshold_by_column", allow_reuse=True
    )(float_value_validation)

    _validated_missing_data_threshold_by_row = validator(
        "missing_data_threshold_by_row", allow_reuse=True
    )(float_value_validation)


class FeaturizeConfig(BaseModel):
    """Specify the parameters of the `featurize` flow"""

    target_column: str = "target"

    # Columns that will not be used in the model
    columns_to_remove: list = ["player1_id", "player2_id"]

    correlation_threshold: float = 0.85

    # Define categorical columns to encode
    categorical_columns: list = [
        "tourney_level",
        "surface",
        "best_of",
        "player1_hand",
        "player2_hand",
        "hist_confront",
    ]


class SamplerEnum(str, Enum):
    """Enumerate the optuna sampler options"""

    bayesian_search = "bayesian_search"
    grid_search = "grid_search"
    random_search = "random_search"
    genetic_search = "genetic_search"


class DirectionEnum(str, Enum):
    """Enumerate the optuna optimization direction"""

    minimize = "minimize"
    maximize = "maximize"


class FineTuneSampler(BaseModel):
    """Specify the parameters of the fine tuning sampler"""

    name: SamplerEnum = SamplerEnum.grid_search.value
    params: dict = {"seed": 42}


class FineTuneParams(BaseModel):
    """Specify the parameters of the `fine tuning` flow"""

    enable: bool = True
    direction: DirectionEnum = DirectionEnum.maximize.value
    sampler: FineTuneSampler = FineTuneSampler()
    n_trials: int = 2
    n_folds: int = 3
    grids: List[Dict] = [
        {"model_name": "RFC", "grid": {"n_estimators": [10, 100]}},
        {"model_name": "XGC", "grid": {"n_estimators": [10, 100]}},
    ]


class ModelParams(BaseModel):
    """Specify the parameters of the `train` flow"""

    model_name: str = "LR"
    model_params: dict = {}
    finetune: FineTuneParams = FineTuneParams()
