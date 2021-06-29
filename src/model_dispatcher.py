import sys
sys.path.append("C:/Users/uber/Desktop/repos/data-toolkit/src")


from sklearn import tree, ensemble
import xgboost as xgb


models = {     
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(     criterion="entropy"),
    "random_forest": ensemble.RandomForestRegressor(n_estimators=25), 
    "xgb": xgb.XGBRegressor(
        objective ='reg:squarederror', 
        max_depth=8,
        n_estimators=1000,
        min_child_weight=300, 
        colsample_bytree=0.8, 
        subsample=0.9, 
        eta=0.3,    
        seed=42
    )

}


