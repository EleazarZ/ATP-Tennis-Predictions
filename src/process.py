"""Python script to process the data"""
import numpy as np
import pandas as pd
from prefect import flow, task
from tqdm import tqdm

from config import Location, ProcessConfig

tqdm.pandas()


@flow
def process_row(row, df, n_hist_matches=5, n_hist_confront=3):
    """
    Apply different operations on the row of a dataframe
    - extract the tournament information
    - extrat the past performance of the players (winner and loser)
    - extract result of past confrontations
    - extract available information on each player (height, age, ha, etc.)
    Parameters
    ----------
        row: row of a pandas Dataframe
        df: pandas Dataframe containing historical data
        n_hist_matches: int, number of precedent matches to consider
        n_hist_confront: int, number of precedent confrontation to consider
    Returns
    -------
        A new row as pandas Series
    """

    new_row = row[["tourney_level", "surface", "best_of"]].to_dict()

    # Treat potential imbalance classes
    # for odd line indexes, define winner as player1 and loser as player2
    if row.name % 2 == 0:
        player1, player2 = "winner", "loser"
        target = 1
    else:  # for even line index, do the inverse
        player1, player2 = "loser", "winner"
        target = 0

    new_row["target"] = target  # 1 if player1 win else 0
    #
    i = 1
    for player in [player1, player2]:
        for c in ["age", "hand", "ht", "rank", "rank_points"]:
            new_row[f"player{i}_{c}"] = row[f"{player}_{c}"]
        i += 1

    player1_id = row[f"{player1}_id"]
    player2_id = row[f"{player2}_id"]

    new_row["player1_id"] = player1_id
    new_row["player2_id"] = player2_id

    # conditions and date of the match
    tourney_date = row["tourney_date"]
    match_num = row["match_num"]

    i = 1
    for player_id in [player1_id, player2_id]:
        player_stats = retrieve_player_stats(
            df, player_id, n_hist_matches, tourney_date, match_num
        )
        if player_stats is not None:
            for k in player_stats.keys():
                new_row[f"player{i}_{k}"] = player_stats[k]
        i += 1

    new_row["hist_confront"] = hist_confrontation(
        df, player1_id, player2_id, tourney_date, match_num, n_hist_confront
    )
    new_row = pd.Series(new_row)
    return new_row


@task
def hist_confrontation(
    df, player1_id, player2_id, tourney_date, match_num, n_hist
):
    """Retrieve the result of the precedent confrontations between two given players
    Parameters
    ----------
        df(pandas.DataFrame): historical data
        player1_id(int): id of the first player
        player2_id(int): id of the 2nd player
        tourney_date(str): starting date of the competition
        match_num(int): match number in a certain tournament
        n_hist(int): number of precedent confrontations to consider
    Returns
    ------
    result(int): 1 if player1 won most of the confrontations, -1 if player2, and 0 if equality
    """

    mask_player = (
        (df["winner_id"] == player1_id) & (df["loser_id"] == player2_id)
    ) | ((df["winner_id"] == player2_id) & (df["loser_id"] == player1_id))
    tmp = df[mask_player]

    tourney_date = pd.to_datetime(tourney_date)
    mask_tourney_date = (tmp["tourney_date"] < tourney_date) | (
        (tmp["tourney_date"] < tourney_date) & (tmp["match_num"] < match_num)
    )
    tmp = tmp[mask_tourney_date]

    tmp = tmp.sort_values(
        by=["tourney_date", "match_num"], ascending=[False, False]
    ).iloc[:n_hist]

    if tmp.empty:
        # print("Non available historical confrontation for players {} and {}".format(player1_id, player2_id))
        return 0

    if len(tmp[tmp["winner_id"] == player1_id]) > len(
        tmp[tmp["winner_id"] == player2_id]
    ):
        result = 1
    elif len(tmp[tmp["winner_id"] == player2_id]) > len(
        tmp[tmp["winner_id"] == player1_id]
    ):
        result = -1
    else:
        result = 0
    return result


@task
def retrieve_player_stats(
    df: pd.DataFrame,
    player_id: int,
    n_hist: int,
    tourney_date: str,
    match_num: int,
) -> dict:
    """Retrieve the precedent match statistics of a given player
    Parameters
    ----------
        df(pandas.DataFrame): historical data
        player_id(int): id of the player
        n_hist(int): number of precent matches to consider
        tourney_date(str): starting date of the competition
        match_num(int): match number in a certain tournament
    Returns
    ------
    stats_dict(dict)
    """
    # The matches that involved the player
    tmp = df[(df["winner_id"] == player_id) | (df["winner_id"] == player_id)]

    # Sorted precedent matches by date
    tourney_date = pd.to_datetime(tourney_date)
    mask = (tmp["tourney_date"] < tourney_date) | (
        (tmp["tourney_date"] < tourney_date) & (tmp["match_num"] < match_num)
    )
    tmp = tmp[mask]

    tmp = tmp.sort_values(
        by=["tourney_date", "match_num"], ascending=[False, False]
    ).iloc[:n_hist]

    if tmp.empty:  # Return NaN if there is no data
        # print("Non available historical data for player {}".format(player_id))
        return {
            "1stIn": np.nan,
            "1stWon": np.nan,
            "2ndWon": np.nan,
            "SvGms": np.nan,
            "ace": np.nan,
            "bpFaced": np.nan,
            "bpSaved": np.nan,
            "df": np.nan,
            "svpt": np.nan,
        }

    # Split the data when the player lose or won
    tmp_w = tmp[tmp["winner_id"] == player_id]
    tmp_l = tmp[tmp["loser_id"] == player_id]

    w_columns = []
    l_columns = []
    for c in tmp_w.columns:
        if ("w_" in c) or ("winner_" in c):
            w_columns.append(c)
    #
    for c in tmp_l.columns:
        if ("l_" in c) or ("loser_" in c):
            l_columns.append(c)
    #
    tmp_w = tmp_w[w_columns].rename(
        columns=lambda x: "".join(x.split("_")[1:])
    )
    tmp_l = tmp_l[w_columns].rename(
        columns=lambda x: "".join(x.split("_")[1:])
    )

    tmp = pd.concat([tmp_w, tmp_l], axis=0)

    # Mean of the match statistics
    stats_columns = [
        "1stIn",
        "1stWon",
        "2ndWon",
        "SvGms",
        "ace",
        "bpFaced",
        "bpSaved",
        "df",
        "svpt",
    ]
    stats_dict = tmp[stats_columns].mean().to_dict()
    return stats_dict


@task
def filter_on_conditions(
    df: pd.DataFrame,
    select_condition_expression: str = None,
    missing_data_threshold_by_column: float = None,
    missing_data_threshold_by_row: float = None,
):
    """
    Filter a dataframe based on given conditions
    Parameters
    ----------
    df: dataframe
    select_condition_expression: str condition
    missing_data_threshold_by_column: maximum missing data ratio allowed in a column
    missing_data_threshold_by_row: maximum missing data ratio allowed in a row
    Returns
    -------
    df: dataframe with the applied conditions
    """

    # filter on select condtion expression using pandas eval
    if select_condition_expression is not None:
        df = df[pd.eval(select_condition_expression)]

    # Filter on missing data ratio by column
    if missing_data_threshold_by_column is not None:
        missing_data_ratio_by_column = df.apply(
            lambda x: x.isnull().sum() / x.size, axis=0
        )
        selected_columns = missing_data_ratio_by_column[
            missing_data_ratio_by_column < missing_data_threshold_by_column
        ].index.tolist()
        df = df.loc[:, selected_columns]

    # Filter on missing data ratio by row
    if missing_data_threshold_by_row is not None:
        missing_data_ratio_by_row = df.apply(
            lambda x: x.isnull().sum() / x.size, axis=1
        )
        selected_rows = missing_data_ratio_by_row[
            missing_data_ratio_by_row < missing_data_threshold_by_row
        ].index.tolist()
        df = df.loc[selected_rows, :]
    return df


# Flow
@flow
def process(
    location: Location = Location(),
    config: ProcessConfig = ProcessConfig(),
    save: bool = False,
):
    """Flow to process the ata

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    save: bool
        flag that indicates whether or not to save the output
    """

    raw_df = pd.read_csv(location.data_raw)

    # Drop duplicated data
    raw_df.drop_duplicates(subset=config.drop_duplicates_subset_columns)

    for d in config.date_columns:
        column_name, column_format = d["name"], d["format"]
        raw_df[column_name] = pd.to_datetime(
            raw_df[column_name], format=column_format
        )

    raw_df = filter_on_conditions(
        df=raw_df,
        select_condition_expression=config.select_condition_expression,
        missing_data_threshold_by_column=config.missing_data_threshold_by_column,
        missing_data_threshold_by_row=config.missing_data_threshold_by_row,
    )

    sample_size = int(len(raw_df) * config.sample_size)

    processed_df = raw_df.sample(
        sample_size, ignore_index=True, random_state=config.random_state
    ).progress_apply(
        lambda row: process_row(
            row, raw_df, config.n_hist_matches, config.n_hist_confront
        ),
        axis=1,
    )

    # Save the new dataframe to a csv file
    if save:
        processed_df.to_csv(location.data_process, index=False)
    else:
        return processed_df
