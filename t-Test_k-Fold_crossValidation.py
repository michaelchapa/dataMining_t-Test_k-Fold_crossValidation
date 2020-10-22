import math
import numpy as np
import pandas as pd
import operator as op
import matplotlib.pyplot as plt

from itertools import accumulate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

######################### generate_TPR_FPR ##################################
# Purpose:
#   Calculates the numbers for True Positive (TP), False Positive (FP), 
#   True Negatives (TN) and False Negatives (FN). Also True Positive Rate (TPR)
#   and False Positive Rate (FPR). Prints them to terminal.
# Parameters:
#   I String Array      actualClass - Symmetric Binary Class label
#   I Int Array         probability - Probability of predicting (+) Class label
# Returns:
#   DataFrame
# Notes:
#   TP, FP -> Inclusive up (+); TN, FN -> Exclusive down (-);
def generate_TPR_FPR(actualClass, probability):
    encodedClass = pd.get_dummies(actualClass)
    P = encodedClass['P'].to_numpy()
    N = encodedClass['N'].to_numpy()
    
    pCount = sum(P)
    nCount = sum(N)
    
    TP = list(accumulate(P))
    FP = list(accumulate(N))
    
    TN = list(accumulate(N, op.sub, initial = nCount))
    TN = TN[1:]
    FN = list(accumulate(P, op.sub, initial = pCount))
    FN = FN[1:]

    TPR = [x / pCount for x in TP]
    FPR = [x / nCount for x in FP]
    
    labels = ['Actual Class', 'Prob', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR']
    columns = zip(labels, [actualClass.to_numpy(), probability.to_numpy(), \
                           TP, FP, TN, FN, TPR, FPR])
    data = pd.DataFrame()
    for column in columns:
        data[column[0]] = column[1]  
        
    print(data)
    return data
    

######################### generate_ROC_curve ################################
# Purpose:
#   Creates a Reciever Operating Characteristic (ROC) [curve].
#   A graphical plot which illustrates the performance of a
#   Binary Classifier System as its discrimination threshold is varied.
# Parameters:
#   I   String Array   actualClass - Symmetric Binary Class Label
#   I   Int Array      probability - Probability of predicting (+) Class Label
# Returns:
#   None
# Notes:
#   The roc_curve() is restricted to the Binary Classification task
def generate_ROC_curve(actualClass, probability):
    # convert from Binary Nominal to Binary Numerical
    encodedClass = pd.get_dummies(actualClass) # generates 2 columns (N & P)
    actualClass = encodedClass['P']
    actualClass = [int(value) for value in actualClass]
    probability = [float(value) for value in probability]
    
    # False-Pos Rate (FPR), True-Pos Rate (TPR)
    fpr, tpr, thresholds = roc_curve(actualClass, probability, pos_label = 1)
    aucScore = roc_auc_score(actualClass, probability)
    xY_lim = np.arange(0, 1.1, .1)
    
    # Plot ROC Curve
    plt.figure()
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Probabilistic classifier')
    plt.plot(fpr, tpr, 'r-', label = 'ROC curve (area = %.2lf)' % (aucScore))
    plt.plot(xY_lim, xY_lim, 'b--')
    plt.legend(loc = "lower right")
    plt.show()


######################### t_Test_pairwise ###############################
# Purpose:
#   Does t-test on two models trained on the same data.
#   Uses Pairwise comparison to calculate variance.
# Parameters:
#   I   2-D Array   errorRates      shape(2, 10) consists of error-rates
# Returns:
#   None
# Notes:
#   T-table PDF located in project folder (or can be found in most stat books)
#   Variance calculation is a Summation from 1 to k, where the column values
#   are x & y.
def t_Test_pairwise(errorRates):
    _, k = np.shape(errorRates)
    dof = k - 1 # degrees of freedom (dof)
    M1 = np.mean(errorRates[0, :])
    M2 = np.mean(errorRates[1, :])
    
    variance = sum([((1/ k) * (((x - y) - (M1 - M2)) ** 2)) \
                    for x, y in errorRates.T])
    
    t = (M1 - M2) / math.sqrt(variance / k)
    
    print("Pairwise Comparison t-Test")
    print("var: %.2lf" % (variance))
    print("  t: %.2lf" % (t))
    print("dof: %d" % (dof))
    print("\nt-distribution critical value: 4.297")
    
    
######################### t_Test_twoSample ##############################
# Purpose: 
#   Does t-test on two models trained on independent datasets.
#   Uses Two-Sample comparison to calculate variance.
# Parameters:
#   I   2-D Array   errorRates      shape(2, 10) consists of error-rates
# Returns:
#   None
# Notes:
#   None
def t_Test_twoSample(errorRates):
    _, k = np.shape(errorRates)
    dof = k - 1 # degrees of freedom (dof)
    M1_mean = np.mean(errorRates[0, :])
    M2_mean = np.mean(errorRates[1, :])
    M1_var = np.var(errorRates[0, :])
    M2_var = np.var(errorRates[1, :])

    variance = math.sqrt((M1_var / k) + (M2_var / k))
    t = (M1_mean - M2_mean) / math.sqrt(variance / k)
    
    print("\n\nTwo-Sample t-Test")
    print("var: %.2lf" % (variance))
    print("t: %.2lf" % (t))
    print("dof: %d" % (dof))
    print("\nt-distribution critical value: 4.297")


######################## k_fold_crossValidation #############################
# Purpose:
#   
# Parameters:
#   I   String      file        CSV file made of 
#   
# Returns:
#   None
# Notes:
#   None
def k_fold_crossValidation(file, k):
    data = pd.read_csv(file).to_numpy()
    X = data[:, :-1]
    y = data[:, -1]

    rkf = RepeatedKFold(n_repeats = 10, n_splits = k)
    
    M1 = []
    M2 = []
    M1_accuracies = []
    M2_accuracies = []
    count = 0
    
    for trainIndex, testIndex in rkf.split(data):
        X_train, X_test = X[trainIndex], X[testIndex]
        y_train, y_test = y[trainIndex], y[testIndex]
        
        # Naive Bayes Classifier
        gnb = GaussianNB()
        y_predict = gnb.fit(X_train, y_train).predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_predict)

        # Adaboost Classifier
        clf = AdaBoostClassifier(n_estimators = 100)
        y_predict = clf.fit(X_train, y_train).predict(X_test)
        ab_accuracy = accuracy_score(y_test, y_predict)
        
        M1_accuracies.append(nb_accuracy)
        M2_accuracies.append(ab_accuracy)
        
        if(count == 9):
            M1.append(np.mean(M1_accuracies))
            M2.append(np.mean(M2_accuracies))
            M1_accuracies = []
            M2_accuracies = []
            count = 0
            
        count += 1
    
    print(M1)
    print(M2)

    
    
def main():
    results = np.array([[1, 'P', .95], [2, 'N', .85], [3, 'P', .78], 
                        [4, 'P', .66], [5, 'N', .60], [6, 'P', .55], 
                        [7, 'N', .53], [8, 'N', .52], [9, 'N', .51], 
                        [10, 'P', .40]])
    
    model_errorRates = np.array([[30.5, 32.2, 20.7, 20.6, 31.0, 
                                 41.0, 27.7, 26.0, 21.5, 26.0], 
                                [22.4, 14.5, 22.4, 19.6, 20.7, 
                                 20.4, 22.1, 19.4, 16.2, 35.0]])
    
    data = pd.DataFrame(results[:, 1:], index = results[:, 0], \
                        columns = ['actualClass', 'probability'])
    
    # generate_TPR_FPR(data['actualClass'], data['probability'])
    # generate_ROC_curve(data['actualClass'], data['probability'])
    # t_Test_pairwise(model_errorRates)
    # t_Test_twoSample(model_errorRates)
    k_fold_crossValidation("hwk07.csv", 10)
    
# Context the file is running in is __main__ 
if __name__ == "__main__":
    main()