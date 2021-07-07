import sys
from scipy.sparse.construct import rand

from sklearn import tree, ensemble
from sklearn.svm import SVC
import xgboost as xgb


models = {     
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini", random_state=777),
    "decision_tree_entropy": tree.DecisionTreeClassifier(     criterion="entropy", random_state=777),
    "random_forest_classifier": ensemble.RandomForestClassifier(random_state=42),
    "random_forest_regressor": ensemble.RandomForestRegressor(n_estimators=25), 
    "xgb_regressor": xgb.XGBRegressor(
        objective ='reg:squarederror', 
        max_depth=8,
        n_estimators=1000,
        min_child_weight=300, 
        colsample_bytree=0.8, 
        subsample=0.9, 
        eta=0.3,    
        seed=42
    ),
    "svm": SVC()

}


