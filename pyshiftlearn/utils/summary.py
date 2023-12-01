import pandas as pd


def average_rank(df: pd.DataFrame, _columns: list[str], mode: str = "descending") -> pd.DataFrame:
    """
    Calculate the average rank for each column in the given DataFrame.

    Parameters
    ---------
    df: pd.DataFrame
        The DataFrame containing the columns to calculate average rank for.
    _columns: list[str]
        The list of column names to calculate average rank for.
    mode : str, optional
        The mode for ranking the columns. Either "ascending" or "descending". The default is "descending".
    Returns
    ------
    pd.DataFrame
        The DataFrame with an additional column "Average_rank" containing the average rank for each column.
    """
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()
    # If the data frame doesn't have the "Average Rank" column, add it
    if "Average Rank" not in df_copy.columns:
        df_copy["Average Rank"] = 0
    # Sum the ranks of the selected columns, divide by the number of columns, and add it to the "Average Rank" column
    if mode == "descending":
        df_copy["Average Rank"] += df_copy[_columns].rank(ascending=False).sum(axis=1) / len(_columns)
    else:
        df_copy["Average Rank"] += df_copy[_columns].rank(ascending=True).sum(axis=1) / len(_columns)
    return df_copy
