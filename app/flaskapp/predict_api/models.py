"""class Content to process and return a prediction from a request"""
import marshal
import os
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class Content(object):
    """Class Content that take a dictionary in input. The class contains a predict class method"""

    def __init__(self, content_value):
        self.content_value = content_value
        self.models_path = (
            Path(__file__).resolve().parent.parent.parent.parent.absolute()
        )

    @classmethod
    def predict(cls, content_value):

        new_obj = cls(content_value)

        new_obj.load()

        X = new_obj.preprocess()

        prob = new_obj.model.predict_proba(X)[0]

        pred = new_obj.model.predict(X)[0]

        # 1 if player1 is winner, else 0 (if player2)
        winner = "player1" if pred == 1 else "player2"

        result = {
            "Probability player 1 will be winner": round(prob[-1], 2),
            "Probability player 2 will be winner": round(prob[0], 2),
            "Predicted winner": winner,
        }

        # Make sure values are float and not numpy float for jsonify
        for k, v in result.items():
            if isinstance(v, np.floating):
                result[k] = float(v)

        return result

    def load(self):
        # Load configs
        with open(
            os.path.join(self.models_path, "models", "config.pkl"), "rb"
        ) as handle:
            config = joblib.load(handle)

        # Load processor functions
        processor = {}
        with open(
            os.path.join(
                self.models_path, config.get("location").get("processor")
            ),
            "rb",
        ) as funcfile:
            while True:
                try:
                    code = marshal.load(funcfile)
                except EOFError:
                    break
                processor[code.co_name] = types.FunctionType(
                    code, processor, code.co_name
                )

        # Load the model
        # model_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models", 'model.pkl')
        with open(
            os.path.join(
                self.models_path, config.get("location").get("model")
            ),
            "rb",
        ) as handle:
            self.model = joblib.load(handle)

        # Load the featurizer
        with open(
            os.path.join(
                self.models_path, config.get("location").get("featurizer")
            ),
            "rb",
        ) as handle:
            featurizer = joblib.load(handle)

        self.scaler = featurizer["scaler"]
        self.fill_value = featurizer["fill_value"]
        self.encoder = featurizer["encoder"]
        self.categ_columns = featurizer["categorical_columns"]
        self.X_columns = featurizer["X_columns"]

        self.process_config = config.get("processor")
        self.df = config["df"]
        self.processor = processor

    def preprocess(self):
        """Apply some preprocessing functions such imputing null values and standardizing the data before predicting the content"""

        self.content_value["tourney_info"]["tourney_date"] = pd.to_datetime(
            self.content_value["tourney_info"]["tourney_date"]
        )

        content = self.processor["process_single_match"](
            row=self.content_value,
            df=self.df,
            n_hist_matches=self.process_config.get("n_hist_matches"),
            n_hist_confront=self.process_config.get("n_hist_confront"),
            predict_mode=True,
        )
        content = pd.DataFrame(content, columns=content.keys(), index=[0])

        # Fill NA if present
        for c in content.columns:
            if (content[c].isnull().sum()) and (c in self.fill_value.index):
                content.loc[:, c] = self.fill_value[c]

        # Encode categorical values
        encoded_data = self.encoder.transform(
            content[self.categ_columns]
        ).toarray()
        encoded_data = pd.DataFrame(
            encoded_data, columns=self.encoder.get_feature_names_out()
        )
        content = pd.concat([content, encoded_data], axis=1)

        # Align the data to the ones used for the fit
        content, _ = content.align(self.X_columns, join="right", axis=1)

        X = pd.DataFrame(
            self.scaler.transform(content), columns=content.columns
        )

        return X
