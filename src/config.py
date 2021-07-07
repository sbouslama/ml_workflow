DATA_PATH = "../data"

MODEL_OUTPUT = "../models/"

TARGET = "is_fraud" # String, name of the target column from your dataset
VAL_STRATEGY = "0"
SCALE = False
SAVE_MODELS = True
# NUM_CLASS = 1

FOLD_VERSION  = "V0"
TRAIN_VERSION = "V0.2"

STRAITIFIED_KFOLD = True
NFOLDS = 5 # Int Number of folder used while using the kfold strategy
COL_TO_DROP = ["Unnamed: 0", "cc_num","first","last","street","zip", "trans_date_trans_time","dob","trans_num","unix_time","job","merch_long","merch_lat","trans_date"] # List, columns to drop while training. by default, the columns: kflod and target will be dropped before the training starts.

EVAL_METRIC =  ["f1_score", "auc"]# list of names of the evaluation metric to use ["rmse", "mse"]