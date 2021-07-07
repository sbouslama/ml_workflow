# Evaluation metrics

## I. Classification


### **Accuracy**
***


Accuracy is the proportion of true results among the total number of cases examined. 

Accuracy = (TP+TN)/(TP+FP+FN+TN)


**When to use?**
Accuracy is a valid choice of evaluation for classification problems which are well balanced and not skewed or No class imbalance.

### **Precision**
***

It answers the following question: what proportion of predicted Positives is truly Positive?

Precision = (TP)/(TP+FP)

Precision is a valid choice of evaluation metric when we want to be very sure of our prediction. For example: If we are building a system to predict if we should decrease the credit limit on a particular account, we want to be very sure about our prediction or it may result in customer dissatisfaction.

### **Recall**
***
It answers the following question: what proportion of actual Positives is correctly classified?

Recall = (TP)/(TP+FN)

**When to use?**

Recall is a valid choice of evaluation metric when we want to capture as many positives as possible. For example: If we are building a system to predict if a person has cancer or not, we want to capture the disease even if we are not very sure.

### **F1 score**
***

F1 = 2*(precision*recall)/(precision+recall)

**When to use?**

We want to have a model with both good precision and recall.

### **Log Loss / Binary Crossentropy**
***

Binary Log loss for an example is given by the below formula where p is the probability of predicting 1.

-(ylog(p)+(1-y)log(1-p))

**When to Use?**

When the output of a classifier is prediction probabilities. Log Loss takes into account the uncertainty of your prediction based on how much it varies from the actual label. This gives us a more nuanced view of the performance of our model. In general, minimizing Log Loss gives greater accuracy for the classifier.

### **AUC**
***
AUC ROC indicates how well the probabilities from the positive classes are separated from the negative classes





## II. Regression 

### **R Square/Adjusted R Square**
***
R Square measures how much variability in dependent variable can be explained by the model. It is the square of the Correlation Coefficient(R) and that is why it is called R Square.

R Square is calculated by the sum of squared of prediction error divided by the total sum of the square which replaces the calculated prediction with mean. R Square value is between 0 to 1 and a bigger value indicates a better fit between prediction and actual value.
R Square is a good measure to determine how well the model fits the dependent variables. However, it does not take into consideration of overfitting problem.


### **MSE/RMSE**
***
While R Square is a relative measure of how well the model fits dependent variables, Mean Square Error is an absolute measure of the goodness for the fit.

MSE is calculated by the sum of square of prediction error which is real output minus predicted output and then divide by the number of data points. It gives you an absolute number on how much your predicted results deviate from the actual number.

Root Mean Square Error(RMSE) is the square root of MSE. It is used more commonly than MSE because firstly sometimes MSE value can be too big to compare easily

### **MAE**
***
Mean Absolute Error(MAE) is similar to Mean Square Error(MSE). However, instead of the sum of square of error in MSE, MAE is taking the sum of the absolute value of error.

Compare to MSE or RMSE, MAE is a more direct representation of sum of error terms. **MSE gives larger penalization to big prediction error** by square it while MAE treats all errors the same.