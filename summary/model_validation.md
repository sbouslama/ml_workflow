# Model validation 

There are many different ways one can do cross-validation, and it is the most critical step when it comes to building a good machine learning model which is generalizable when it comes to unseen data. Choosing the right cross-validation depends on the dataset you are dealing with, and one’s choice of cross-validation on one dataset may or may not apply to other datasets. However, there are a few types of cross-validation techniques which are the most popular and widely used.  

These include: 
 
## 1-  k-fold cross-validation: ngroups = k

The process is the following:

- Split train data into K folds.

- Iterate though each fold: retrain the model on all folds except current fold, predict for the current fold.

- Use the predictions to calculate quality on each fold. Find such hyper-parameters, that quality on each fold is maximized. You can also estimate mean and variance of the loss. This is very helpful in order to understand the significance of improvement.

## 2-  stratified k-fold cross-validation 

It is used to avoid having random folds and preserve the distribution of the target variable over different folds.
Stratification is useful for: 
- Small data

- Unbalanced datasets

- Binary/Multi-class classification

Stratified k-fold could be used for regression problems. We just have to divide the target into bins. 


To change the code and create stratified k-folds, you can indicate that in the config file. You should set `straitifiedKFOLD = True`


## 3- hold-out based validation: ngroups = 2

Hold-out is when you split up your dataset into a ‘train’ and ‘test’ set 

The hold-out method is good to use when you have a very large dataset, you’re on a time crunch, or you are starting to build an initial model in your data science project. 

Hold-out is also used very frequently with time-series data.

## 4- leave-one-out cross-validation: ngroups = len(train_df)

LOOCV, is a configuration of k-fold cross-validation where k is set to the number of examples in the dataset.

It is used  for Small datasets or when estimated model performance is critical

# Takeaways

Cross-validation is the first and most essential step when it comes to building machine learning models. If you want to do feature engineering, split your data first. If you're going to build models, split your data first. If you have a good crossvalidation scheme in which validation data is representative of training and realworld data, you will be able to build a good machine learning model which is highly generalizable.