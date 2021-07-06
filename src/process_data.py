from os.path import join

import pandas as pd
from imblearn.over_sampling import SMOTE

import config
from utilities.categorical_encoders import Encoder
from utilities.feature_engineering import extract_date_features


def process(train_df):
    # Encode categorical features using label encoder
    categorical_features = ['category', 'gender','merchant', 'city', 'state']
    encoder = Encoder(encoder="label_encoder", columns_names=categorical_features)
    train_df = encoder.fit_transform(train_df)
    
    # Extact date features from the transaction date
    train_df["trans_date_trans_time"] = pd.to_datetime(train_df["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S")
    train_df = extract_date_features(train_df, "trans_date_trans_time")

    # Drop unuseful columns
    col_to_drop = config.COL_TO_DROP
    print(f"__COL_TO_DROP__: {col_to_drop}")
    train_df = train_df.drop(col_to_drop, axis=1)

    # Oversamping
    sm = SMOTE()
    train_df[[col for col in train_df.columns if col not in [config.TARGET]]], train_df["is_fraud"] = sm.fit_resample(train_df.drop("is_fraud", axis=1), train_df["is_fraud"])
    return train_df


def create_agg_account_features(train_df, windows=["24H", "7D", "30D","90D"]):
    train_df.sort_values(by="trans_date_trans_time", inplace=True)
    for window in windows:
        try:
            train_df.set_index("trans_date_trans_time", inplace=True)
        except Exception as exc:
            print(exc)
        agg = train_df.groupby(["cc_num"])["amt"].rolling(window).agg({f"mean_{window}":"mean",f"count_{window}":"count"}).reset_index()

        train_df = train_df.merge(agg, on=["cc_num","trans_date_trans_time"], how="left")
    return train_df

def extract_features(train_df):
    # Calculate age of the user
    train_df["dob"] = pd.to_datetime(train_df["dob"], format="%Y-%m-%d")
    train_df["age"]  = train_df["trans_date_trans_time"]-train_df["dob"]
    train_df["age"] = round(train_df["age"].dt.days/365)

    # create aggregated features for the different accounts
    # Generally the behavior of the client in termes of number of transaction can help identify fraudulent transactions.
    # Past transaction of the last 24h, week, month are examples of the account aggregated features
    train_df = create_agg_account_features(train_df)

    return train_df

if __name__ == "__main__": 

    # Read training data
    df = pd.read_csv(join(config.DATA_PATH, "train_{}_folds.csv".format(config.TRAIN_VERSION)))

    # pre-process the dataframe
    df = process(df)
    # Engineer features if any
    df = extract_date_features(df)
       
    # save the new csv with kfold column
    df.to_csv(join(config.DATA_PATH, "train_{}_process.csv".format(config.TRAIN_VERSION)), index=False)
    # df.to_pickle(join(config.DATA_PATH, "train_{}_folds.pkl".format(config.VAL_STRATEGY)))
