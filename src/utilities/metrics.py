from math import sqrt

from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score


def calculate_error(error, x_train, y_train, y_valid, preds, clf, fold):
    """
    :input error: String with the name of the metric to be used in the evaluation
    :input x_train: Dataframe of the training data
    :input y_train: Dataframe of the training target
    :input y_valid: Dataframe of the validation data
    :input preds: Dataframe of the predictions
    :input clf: Name of the model used from the model_dispatcher
    :input fold: Int of the fold number
    """
    assert set(error).difference(set(["mse","rmse","f1_score","auc"])) == set()
    assert isinstance(error,list), "metric/error should have the list type"
    def calculate_error(err):
        if err=="mse":
            train_error = mean_squared_error(clf.predict(x_train), y_train)
            val_error = mean_squared_error(y_valid, preds)
        if err=="rmse":
            train_error = sqrt(mean_squared_error(clf.predict(x_train), y_train))
            val_error = sqrt(mean_squared_error(y_valid, preds))
        if err=="f1_score":
            train_error = f1_score(clf.predict(x_train), y_train)
            val_error = f1_score(y_valid, preds)
        if err=="auc":
            train_error = roc_auc_score(clf.predict(x_train), y_train)
            val_error = roc_auc_score(y_valid, preds)

        return train_error, val_error

    for err in error:
        train_error, val_error = calculate_error(err)
        print("Error/metric: {} Fold={}, train_error={}, valid_error={}".format(err, fold, round(train_error,3), round(val_error,3)))
