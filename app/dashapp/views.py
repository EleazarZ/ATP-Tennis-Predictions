"""Python functions that returns components of forms to fill of data"""
from datetime import date

import dash_bootstrap_components as dbc
from dash import dcc, html


def input_player_attributes(player, player_id_options):

    children = [
        dbc.InputGroup(
            [
                dbc.InputGroupText("id and name"),
                dbc.Select(
                    id=f"player-{player}-id",
                    options=[
                        {
                            "label": "Not in the list, input the values",
                            "value": "None",
                        }
                    ]
                    + [
                        {
                            "label": str(i["player_id"])
                            + " - "
                            + i["player_name"],
                            "value": i["player_id"],
                        }
                        for i in player_id_options
                    ],
                    # value=player_id_options[0]["player_id"],
                    placeholder="Select the player",
                ),
            ]
        ),
        html.Div(id=f"player-{player}-input-id-name"),
        html.Br(),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Age"),
                dbc.Input(
                    id=f"player-{player}-age",
                    type="number",
                    min=10,
                    max=100,
                    step=0.1,
                    placeholder="Age of the player",
                ),
            ]
        ),
        html.Br(),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Height"),
                dbc.Input(
                    id=f"player-{player}-ht",
                    type="number",
                    min=10,
                    max=500,
                    step=0.1,
                    placeholder="Height of the player",
                ),
            ]
        ),
        html.Br(),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Rank"),
                dbc.Input(
                    id=f"player-{player}-rank",
                    type="number",
                    min=0,
                    max=1000000,
                    step=1,
                    placeholder="Rank of the player",
                ),
            ]
        ),
        html.Br(),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Rank points"),
                dbc.Input(
                    id=f"player-{player}-rank-points",
                    type="number",
                    min=10,
                    max=100000,
                    step=0.1,
                    placeholder="Rank points of the player",
                ),
            ]
        ),
        html.Br(),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Hand"),
                dbc.Select(
                    id=f"player-{player}-hand",
                    options=[
                        {"label": "Left", "value": "L"},
                        {"label": "Right", "value": "R"},
                        {"label": "Both", "value": "U"},
                        {"label": "Unknown", "value": "None"},
                    ],
                    # value = np.nan,
                    placeholder="hand of the player",
                ),
            ]
        ),
    ]
    return html.Div(children)


def input_tourney_infos():
    tourney_levels = {
        "G": "Grand Slams",
        "M": "Masters 1000s",
        "A": "other tour-level events",
        "C": "Challengers",
        "S": "Satellites/ITFs",
        "F": "Tour finals and other season-ending events",
        "D": "Davis Cup",
    }

    children = [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Starting date of the tournament"),
                dcc.DatePickerSingle(
                    id="tourney-date",
                    min_date_allowed=date.today(),
                    max_date_allowed=date.today().replace(
                        year=date.today().year + 1
                    ),
                    initial_visible_month=date(2023, 1, 1),
                    date=date.today(),
                    display_format="YYYY-MM-DD",
                ),
            ]
        ),
        html.Br(),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Tournament level"),
                dbc.Select(
                    id="tourney-level",
                    options=[
                        {"label": v, "value": k}
                        for k, v in tourney_levels.items()
                    ],
                    placeholder="tournament level",
                ),
            ]
        ),
        html.Br(),
        dbc.InputGroup(
            [
                dbc.InputGroupText("surface"),
                dbc.Select(
                    id="surface",
                    options=[
                        {"label": i, "value": i}
                        for i in ["Grass", "Clay", "Carpet", "Hard", "None"]
                    ],
                ),
            ]
        ),
        html.Br(),
        dbc.InputGroup(
            [
                dbc.InputGroupText("The maximum number of sets allowed"),
                dbc.Select(
                    id="best-of",
                    options=[{"label": i, "value": i} for i in [1, 3, 5]],
                    value=1,
                    placeholder="maximum number of sets",
                ),
            ]
        ),
        html.Br(),
        dbc.InputGroup(
            [
                dbc.InputGroupText("match number in the tournament"),
                dbc.Input(
                    id="match-num",
                    type="number",
                    min=1,
                    max=1000,
                    step=1,
                    placeholder="match number",
                ),
            ]
        ),
    ]
    return html.Div(children)
