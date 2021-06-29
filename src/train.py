import argparse
import os
from math import sqrt
from os.path import join

import joblib
import pandas as pd
# from model_selection.selection import Selection
from sklearn import tree
from sklearn.metrics import mean_squared_error

import config
import model_dispatcher


def run(fold, model):
    # load data

    # df = pd.read_csv(join(config.DATA_PATH, "train_{}_folds.csv".format(config.VAL_STRATEGY)))
    df = pd.read_pickle(join(config.DATA_PATH, "train_{}_folds.pkl".format(config.VAL_STRATEGY)))

    # training data is where kfold is not equal to provided fold     
    # also, note that we reset the index    

    df_train = df[(df.kfold != fold)].reset_index(drop=True) 

    # validation data is where kfold is equal to provided fold  
   
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #  drop the label column from dataframe
    col_to_drop = config.COL_TO_DROP
    print(f"__COL_TO_DROP__: {col_to_drop}")
    x_train, y_train = df_train.drop(col_to_drop, axis=1), df_train[config.TARGET].values
    x_valid, y_valid = df_valid.drop(col_to_drop, axis=1), df_valid[config.TARGET].values
    
    # selector = Selection(models_config=model_dispatcher.toolkit_models, problem_type="regression", judgment_metric="RMSE")
    # selector.BestModelK(selection_type='btb', btb_n_iter=5, eval_set=[(x_train,y_train) , ( x_valid, y_valid)] )

    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]
     # init model, fit and predict
    if clf == 'xgb':
        import xgboost as xgb
        # convert to Dmatric to gain efficiency and performance
        x_train = xgb.DMatrix(data = x_train, label=y_train)
        X_valid  = xgb.DMatrix(data = X_valid, label=y_valid)
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
    train_error = round(sqrt(mean_squared_error(clf.predict(x_train), y_train)),4)
    val_error = round(sqrt(mean_squared_error(y_valid, preds)),4)
    print("Fold={}, train_error={}, Error={}".format(fold, train_error, val_error))
    # save the model     
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.bin"))

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse     
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model", type=str)
    parser.add_argument("--fold", type=int)


    args = parser.parse_args()
    if args.fold:
        run(fold= args.fold, args.model)
    else:
        [run(fold=fold, model=args.model) for fold in range(config.NFOLDS)]
