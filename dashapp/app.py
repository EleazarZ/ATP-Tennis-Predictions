#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python script to run a dash application"""
from pathlib import Path

import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import plotly.express as px
import requests
from dash import Dash, Input, Output, State, dcc, html
from views import input_player_attributes, input_tourney_infos

# The api-endpoint
API_ENDPOINT = "http://127.0.0.1:5000/predict/"

# Load the processed data to get the players infos
models_path = Path(__file__).parent.parent.parent.absolute()
with open(models_path / "models" / "config.pkl", "rb") as handle:
    df = joblib.load(handle)
    df = df["df"]

player_id_options = (
    pd.concat(
        [
            df[["winner_id", "winner_name"]].rename(
                columns={
                    "winner_id": "player_id",
                    "winner_name": "player_name",
                }
            ),
            df[["loser_id", "loser_name"]].rename(
                columns={"loser_id": "player_id", "loser_name": "player_name"}
            ),
        ],
        axis=0,
    )
    .drop_duplicates()
    .to_dict(orient="records")
)

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.H1(
            children="ATP Predict Dash App", style={"textAlign": "center"}
        ),
        html.Br(),
        html.H2("Input data"),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [html.H2("Tournament info"), input_tourney_infos()]
                    )
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H2("Player 1 info"),
                            input_player_attributes(1, player_id_options),
                        ]
                    )
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H2("Player 2 info"),
                            input_player_attributes(2, player_id_options),
                        ]
                    )
                ),
            ]
        ),
        html.Button("Submit", id="predict-submit-button", n_clicks=0),
        html.Div(id="predict-output"),
    ],
    style={
        "width": "95%",
        "margin-left": "15px",
        "margin-right": "10px",
    },
)


# Callbacks
@app.callback(
    Output(component_id="predict-output", component_property="children"),
    inputs=dict(
        n_clicks_predict=Input("predict-submit-button", "n_clicks"),
        # tournament infos
        tourney_info=dict(
            surface=State(component_id="surface", component_property="value"),
            match_num=State(
                component_id="match-num", component_property="value"
            ),
            tourney_date=State(
                component_id="tourney-date", component_property="date"
            ),
            best_of=State(component_id="best-of", component_property="value"),
            tourney_level=State(
                component_id="tourney-level", component_property="value"
            ),
        ),
        # player 1 infos
        player1=dict(
            id=State(component_id="player-1-id", component_property="value"),
            age=State(component_id="player-1-age", component_property="value"),
            hand=State(
                component_id="player-1-hand", component_property="value"
            ),
            ht=State(component_id="player-1-ht", component_property="value"),
            rank=State(
                component_id="player-1-rank", component_property="value"
            ),
            rank_points=State(
                component_id="player-1-rank-points", component_property="value"
            ),
        ),
        # player 2 infos
        player2=dict(
            id=State(component_id="player-2-id", component_property="value"),
            age=State(component_id="player-1-age", component_property="value"),
            hand=State(
                component_id="player-2-hand", component_property="value"
            ),
            ht=State(component_id="player-2-ht", component_property="value"),
            rank=State(
                component_id="player-2-rank", component_property="value"
            ),
            rank_points=State(
                component_id="player-2-rank-points", component_property="value"
            ),
        ),
    ),
)
def update_output(n_clicks_predict, tourney_info, player1, player2):
    if n_clicks_predict is not None:
        if n_clicks_predict > 0:

            # Data to be sent to the API
            json_data = {
                "tourney_info": tourney_info,
                "player1": player1,
                "player2": player2,
            }

            for key, value in json_data.items():
                for k, v in value.items():
                    if v == "None":
                        json_data[key][k] = None

            # sending post request and saving response as response object
            response = requests.post(url=API_ENDPOINT, json=json_data)
            if response.ok or (response.status_code == 200):
                response = response.json()
                prob = pd.Series(
                    {
                        "Player 1": response["result"][
                            "Probability player 1 will be winner"
                        ],
                        "Player 2": response["result"][
                            "Probability player 2 will be winner"
                        ],
                    }
                )
                subtitle = (
                    "Predicted winner is "
                    + response["result"]["Predicted winner"]
                )
                fig = px.bar(
                    x=prob.index,
                    y=prob,
                    title=f"Probability to win the match: {subtitle}",
                )
                fig.update_layout(
                    xaxis_title="Players", yaxis_title="probability"
                )
                children = [
                    html.H2("prediction result"),
                    dcc.Graph(figure=fig),
                ]
            else:
                children = [html.H2("prediction result"), response.text]
            return children
        return html.H4("")


@app.callback(
    Output("player-1-id", "options"),
    Output("player-1-id", "value"),
    Input("id-submit-button-1", "n_clicks"),
    State("player-1-input-id", "value"),
    State("player-1-input-name", "value"),
    State("player-1-id", "options"),
    prevent_initial_call=True,
)
def add_dropdown_option_1(n_clicks, input_id, input_name, options):
    return (
        options
        + [
            {
                "label": str(input_id) + " - " + str(input_name),
                "value": input_id,
            }
        ],
        input_id,
    )


@app.callback(
    Output("player-2-id", "options"),
    Output("player-2-id", "value"),
    Input("id-submit-button-2", "n_clicks"),
    State("player-2-input-id", "value"),
    State("player-2-input-name", "value"),
    State("player-2-id", "options"),
    prevent_initial_call=True,
)
def add_dropdown_option_2(n_clicks, input_id, input_name, options):
    return (
        options
        + [
            {
                "label": str(input_id) + " - " + str(input_name),
                "value": input_id,
            }
        ],
        input_id,
    )


@app.callback(
    Output("player-1-input-id-name", "children"),
    Output("player-2-input-id-name", "children"),
    Input("player-1-id", "value"),
    Input("player-2-id", "value"),
    prevent_initial_call=True,
)
def input_palyers_infos(input_value_1, input_value_2):
    def _input_player_infos(input_value, id_suffix):
        if input_value == "None":
            children = [
                dcc.Input(
                    id=f"player-{id_suffix}-input-id",
                    value="",
                    placeholder="input player id...",
                ),
                dcc.Input(
                    id=f"player-{id_suffix}-input-name",
                    value="",
                    placeholder="input player name...",
                ),
                html.Button(
                    "Add Option",
                    id=f"id-submit-button-{id_suffix}",
                    n_clicks=0,
                ),
            ]
        else:
            children = html.H4("")
        return children

    return _input_player_infos(input_value_1, 1), _input_player_infos(
        input_value_2, 2
    )


if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)
