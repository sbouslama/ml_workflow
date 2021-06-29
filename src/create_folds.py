from os.path import join

import pandas as pd
from sklearn import model_selection

import config

# *********************************** # 
# cross-validation is a step in the process of building a machine learning model which helps us ensure that our models fit the data accurately and also ensures that we do not overfit. 

# *********************************** # 
def create_folds(df, straitifiedKFOLD, **args):
    
    # we create a new column called kfold and fill it with -1     
    df["kfold"] = -1   
    
    # fetch labels     
    y = df[config.TARGET].values
    # initiate the kfold class from model_selection module
    if not straitifiedKFOLD:
        kf = model_selection.StratifiedKFold(n_splits=config.NFOLDS, **args)
    else:
        kf = model_selection.KFold(n_splits=config.NFOLDS, **args)          
    
    # fill the new kfold column     
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):         
        df.loc[v_, 'kfold'] = f   

    return df
 
if __name__ == "__main__": 

    # Read training data
    df = pd.read_pickle(join(config.DATA_PATH, "train_{}.pkl".format(config.TRAIN_VERSION)))

    # create folds
    df = create_folds(df, config.STRAITIFIED_KFOLD)
       
    # save the new csv with kfold column
    # df.to_csv(, index=False)
    df.to_pickle(join(config.DATA_PATH, "train_{}_folds.pkl".format(config.VAL_STRATEGY)))