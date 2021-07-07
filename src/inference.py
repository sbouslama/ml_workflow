import argparse
import joblib
import os
import config
from os.path import join
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)

    args = parser.parse_args()

    test_data = pd.read_pickle(join(config.DATA_PATH, "test_{}.pkl".format(config.TRAIN_VERSION)))
    # drop columns that was ignored in training phase
    test_data = test_data.drop(config.COL_TO_DROP[2:], axis=1)
    # read sample submission 
    submission = pd.read_csv(join(config.DATA_PATH, "sample_submission.csv"))
    # create out_of_fold test data
    oof_test = np.zeros((len(test_data), config.NUM_CLASS)) 
    
    for fold in range(config.NFOLDS):
        model = joblib.load(os.path.join(config.MODEL_OUTPUT, f"{args.model}_{fold}.bin"))

        test_pred = model.predict(test_data)
        oof_test += np.reshape(test_pred,(-1,config.NUM_CLASS))
    
    submission[config.TARGET] = oof_test/config.NFOLDS
    # create submission folder if it does not exist
    if not os.path.exists(os.path.join(config.DATA_PATH, "submission")):
        os.mkdir(os.path.join(config.DATA_PATH, "submission"))
    
    submission[config.TARGET] = round(submission[config.TARGET]).clip(0,20)
    submission.to_csv(os.path.join(config.DATA_PATH, f"submission/submission_{config.TRAIN_VERSION}_{args.model}.csv"), index=False)