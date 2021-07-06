import argparse
import os
from os.path import join

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import model_dispatcher
from utilities.metrics import calculate_error


def run(df, fold, model, error):
    """
    Train the model for the specific fold and evaluate it
    :input df: Dataframe to use for training
    :input fold: Int of the fold number
    :input model: Name of the model used from the model_dispatcher
    :input error: String with the name of the metric to be used in the evaluation
    input save: If true save the model in the path defined in the config file
    """
    print("**********", "Train for fold number: ", fold, "**********")
    df_temp = df.copy()
    if config.SCALE:
        scaler = StandardScaler()
        cols_to_scale = [col for col in df_temp.columns if col not in ["kfold",config.TARGET]]
        df_temp[cols_to_scale] = scaler.fit_transform(df_temp[cols_to_scale])
    
    # training data is where kfold is not equal to provided fold     
    # also, note that we reset the index    
    df_train = df_temp[(df_temp.kfold != fold)].reset_index(drop=True) 

    # validation data is where kfold is equal to provided fold  
   
    df_valid = df_temp[df_temp.kfold == fold].reset_index(drop=True)

    print("Shape of the training data: {}, Shape of testing data: {}".format(df_train.shape[0], df_valid.shape[0]))
    #  drop the label column from dataframe
    x_train, y_train = df_train.drop(["kfold",config.TARGET], axis=1), df_train[config.TARGET].values
    x_valid, y_valid = df_valid.drop(["kfold",config.TARGET], axis=1), df_valid[config.TARGET].values
        
    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]
     # init model, fit and predict
    if clf == 'xgb':
        import xgboost as xgb

        # convert to Dmatric to gain efficiency and performance
        x_train = xgb.DMatrix(data = x_train, label=y_train)
        x_valid  = xgb.DMatrix(data = X_valid, label=y_valid)
        clf.fit(
            x_train, 
            y_train, 
            eval_metric=config.EVAL_METRIC, 
            eval_set=[(x_train, y_train), (x_valid, y_valid)], 
            verbose=True, 
            early_stopping_rounds = 200
        )
    else:
        clf.fit(x_train, y_train) 
    # predict
    preds = clf.predict(x_valid)
    # calculate & print metric
    calculate_error(error, x_train, y_train, y_valid, preds, clf, fold)
    # save the model
    if config.SAVE_MODELS:
        model_output_path = config.MODEL_OUTPUT
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)
        joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.bin"))
    return clf


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse     
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str)
    parser.add_argument("--fold", type=int)

    # load data

    df = pd.read_csv(join(config.DATA_PATH, "train_{}_folds.csv".format(config.VAL_STRATEGY)))
    # df = pd.read_pickle(join(config.DATA_PATH, "train_{}_folds.pkl".format(config.VAL_STRATEGY)))

    args = parser.parse_args()
    if args.fold:
        run(df= df, fold= args.fold, model=args.model, error=config.EVAL_METRIC)
    else:
        [run(df = df, fold=fold, model=args.model, error=config.EVAL_METRIC) for fold in range(config.NFOLDS)]
