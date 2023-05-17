import pickle
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from config import FeaturizeConfig, Location, ProcessConfig


@task
def remove_cor_features(
    df: pd.DataFrame,
    TARGET: str = None,
    exclude: list = None,
    threshold: float = 0.9,
    remove: bool = True,
) -> Union[pd.DataFrame, None]:
    """
    Retrieve highly correlated columns and remove one of them
    Parameters
    ----------
        df: dataframe
        TARGET(str): indicate the target column to be predicted
        exclude(list): define a list of columns to keep whatever are their correlation
        threshold(float): maximum proportion of correlation to tolerate
        remove(bool): removes or not the correlated columns from the dataframe
    Returns:
    -------
        pandas DataFrame
    """
    if TARGET:
        df_tmp = df.loc[:, ~df.columns.isin([TARGET])]

    # Create correlation matrix
    corr_matrix = (
        df_tmp.apply(lambda x: pd.factorize(x)[0])
        .corr(method="pearson", min_periods=1)
        .abs()
    )
    # corr_matrix = df_tmp.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation greater than threshold
    to_drop = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]

    # Remove the target column or other selected exog features if present in the list
    if exclude is not None:
        for elt in exclude:
            if elt in to_drop:
                to_drop.remove(elt)

    # Drop features
    if remove is True:
        df = df.drop(to_drop, axis=1)
        # print('%d columns were dropped from the dataframe.' % (len(to_drop)))
        # print(to_drop)
        return df
    else:
        print("There are %d columns to remove." % (len(to_drop)))
        print("\n", to_drop)


@task
def fillna(
    df: pd.DataFrame, categorical: list = None
) -> (pd.DataFrame, pd.DataFrame):
    """
    Fill missing data. The categorical columns are filled with mode value and
    the others with the median.
    Parameters
    ----------
        df(pandas.DataFrame)
        cateforical(list): list of categorical columns
    Returns
    -------
        pandas.DataFrame, pandas.Series
    """
    if categorical is None:
        categ = df.select_dtypes(exclude="float")
        number = df.select_dtypes(include="float")
    else:
        categ = df.loc[:, df.columns.isin(categorical)]
        number = df.loc[:, ~df.columns.isin(categorical)]
    #
    fill_value_categorical = categ.mode().iloc[0]
    fill_value_number = number.median()
    fill_value = pd.concat([fill_value_categorical, fill_value_number], axis=0)

    categ = categ.fillna(categ.mode().iloc[0])
    number = number.fillna(number.median())

    df = pd.concat([number, categ], axis=1)
    return df, fill_value


# Flow
@flow
def featurize(
    location: Location = Location(),
    process_config: ProcessConfig = ProcessConfig(),
    featurize_config: FeaturizeConfig = FeaturizeConfig(),
    save: bool = False,
) -> Union[None, Tuple[dict, dict]]:
    """Flow to featurize the processed data

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    process_config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    featurize_config: FeaturizeConfig, optional
        Configurations for processing data, by default FeaturizeConfig()
    save: bool
        flag that indicates whether or not to save the output
    """

    # Load the processed dataset
    processed_df = pd.read_csv(location.data_process)

    # Exclude columns based on correlation
    # Remove highly correlated columns
    processed_df = remove_cor_features(
        processed_df,
        TARGET=featurize_config.target_column,
        exclude=featurize_config.columns_to_remove,
        threshold=featurize_config.correlation_threshold,
        remove=True,
    )

    # Define X (features) and y (target)
    X = processed_df.loc[
        :,
        ~processed_df.columns.isin(
            featurize_config.columns_to_remove
            + [featurize_config.target_column]
        ),
    ]
    y = processed_df[featurize_config.target_column]

    # Fill NAs in X
    X, fill_value = fillna(X, featurize_config.categorical_columns)

    # Encode categorical columns
    if featurize_config.categorical_columns:
        # X = pd.get_dummies(data=X, columns=featurize_config.categorical_columns, drop_first=True)

        encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
        encoder.fit(X[featurize_config.categorical_columns])
        X_encoded = encoder.transform(
            X[featurize_config.categorical_columns]
        ).toarray()
        X_encoded = pd.DataFrame(
            X_encoded, columns=encoder.get_feature_names_out()
        )

        X = pd.concat(
            [
                X.loc[
                    :, ~X.columns.isin(featurize_config.categorical_columns)
                ],
                X_encoded,
            ],
            axis=1,
        )

    # Séparer jeu test et jeu d'entrainement, 20% des données dans le jeu test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=process_config.test_size,
        random_state=process_config.random_state,
        shuffle=False,
    )

    # Scale the data (fit to a scaler and transform)
    scaler = preprocessing.StandardScaler()
    X_train.loc[:, :] = scaler.fit_transform(X_train)
    X_test.loc[:, :] = scaler.transform(X_test)

    # Dump and save the outputs
    featurizer = {
        "scaler": scaler,
        "fill_value": fill_value,
        "encoder": encoder,
        "categorical_columns": featurize_config.categorical_columns,
        "X_columns": X.iloc[0:0, :],
    }

    train_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    # Save the new dataframe to a csv file
    if save:
        with open(location.train_data, "wb") as handle:
            joblib.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(location.featurizer, "wb") as handle:
            joblib.dump(featurizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return train_data, featurizer


if __name__ == "__main__":
    featurize(save=True)
