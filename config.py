DATA_PATH = "../data"

MODEL_OUTPUT = "../models/"

TARGET = "" # String, name of the target column from your dataset
NUM_CLASS = 1

TRAIN_VERSION = "V0"

STRAITIFIED_KFOLD = False
NFOLDS = 1 # Int Number of folder used while using the kfold strategy
COL_TO_DROP = [] # List, columns to drop while training

EVAL_METRIC =  # String, name of the evaluation metric to use ("rmse", "mse")