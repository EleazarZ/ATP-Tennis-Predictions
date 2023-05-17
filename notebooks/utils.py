import numpy as np
import pandas as pd
import plotly.express as px
import shap
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Retrieve the information (column type, missing value, unique values) in a dataframe
def recap_dataframe(dataframe: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """Exploratory data analysis of a given dataframe.
    Retrieve information are column data type, missing and filling ratios, number of unique values, etc.
    Parameters
    ----------
        dataframe(pandas.DataFrame): the data to analyze
        axis(int): 0 for columns and 1 for rows
    Returns
    --------
        pandas.DataFrame."""
    if axis == 0:
        tab_recap = pd.DataFrame({"column_dtype": dataframe.dtypes})
        tab_recap["value_count"] = pd.Series(
            dataframe.apply(lambda x: x.size, axis=0)
        )
        tab_recap["n_unique"] = dataframe.apply(
            lambda x: len(x.unique()), axis=0
        )
    else:
        tab_recap = pd.DataFrame(
            {"value_count": dataframe.apply(lambda x: x.size, axis=1)}
        )
    tab_recap["filling_factor"] = pd.Series(dataframe.count(axis=axis))
    tab_recap["null_value"] = pd.Series(
        dataframe.apply(lambda x: x.isnull().sum(), axis=axis)
    )
    tab_recap["null_value_ratio"] = pd.Series(
        dataframe.apply(
            lambda x: round((((x.isnull().sum()) * 100) / x.size), 3),
            axis=axis,
        )
    )
    return tab_recap


# Function to plot a confusion matrix
def plot_confusion_matrix(
    y_true, y_pred, classes=None, normalize=True, cmap="Bluered"
):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`."""

    # Compute the confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(3)
        title_text = "Normalized confusion matrix"
    else:
        title_text = "Confusion matrix, without normalization"

    if classes is None:
        classes = [f" {x} " for x in y_true.unique()]
    fig = fig = px.imshow(
        cm, x=classes, y=classes, text_auto=True, color_continuous_scale=cmap
    )
    fig.update_layout(
        title=title_text,
        yaxis_title="True labels",
        xaxis_title="Predicted labels",
        width=600,
        height=400,
    )
    return fig


def features_importance(model, X_train, nb_samples_used_for_shap=None):
    """Select a given number of features based on their importance. The importance are retrieved from a trained model using shap
    Parameters:
        model: trained model object
        X_train: pandas dataframe
        nb_samples_used_for_shap: 'all' or an int number
    Returns:
        ordered features based on their importance
        mean of the shapelet values
        shapelet values of the data
    """

    if nb_samples_used_for_shap is None:
        X_train_sample = X_train
    else:
        X_train_sample = X_train.sample(nb_samples_used_for_shap)

    if isinstance(model, (RandomForestClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, XGBClassifier):
        # Fix "UnicodeDecodeError" of shap on xgboost
        mybooster = model.get_booster()
        model_bytearray = mybooster.save_raw()[4:]

        def myfun(self=None):
            return model_bytearray

        mybooster.save_raw = myfun
        explainer = shap.TreeExplainer(mybooster)
    else:
        raise TypeError("Model type not yet supported by TreeExplainer")
        # explainer = shap.KernelExplainer(model.predict, X_train_sample)

    shap_values_train = explainer.shap_values(np.asarray(X_train_sample))
    shap_mean = np.mean(np.abs(shap_values_train), axis=0)
    ordered_feats = np.argsort(shap_mean)

    return ordered_feats, shap_mean, shap_values_train
