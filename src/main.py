"""
Create a flow
"""
from prefect import flow

from config import FeaturizeConfig, Location, ModelParams, ProcessConfig
from featurize import featurize
from process import process
from train_model import train


@flow
def main_flow(
    location: Location = Location(),
    process_config: ProcessConfig = ProcessConfig(),
    featurize_config: FeaturizeConfig = FeaturizeConfig(),
    model_params: ModelParams = ModelParams(),
):
    """Flow to run the process, featurize and train flows

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    process_config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    featurize_config: FeaturizeConfig, optional
        Configurations for processing data, by default FeaturizeConfig()
    model_params : ModelParams, optional
        Configurations for training models, by default ModelParams()
    """
    process(location, process_config, save=True)
    featurize(location, process_config, featurize_config, save=True)
    train(location, model_params)


if __name__ == "__main__":
    main_flow()
