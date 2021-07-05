import pandas as pd

def extract_date_features(df, date_col):
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week"] = df[date_col].dt.week
    df["day"] = df[date_col].dt.day
    df["hour"] = df[date_col].dt.hour
    df["minute"] = df[date_col].dt.minute
    df["dayofweek"] = df[date_col].dt.dayofweek

    return df