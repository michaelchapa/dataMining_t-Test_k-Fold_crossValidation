import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator as op

from itertools import accumulate
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder

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
#   None
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

def main():
    results = np.array([[1, 'P', .95], [2, 'N', .85], [3, 'P', .78], 
                        [4, 'P', .66], [5, 'N', .60], [6, 'P', .55], 
                        [7, 'N', .53], [8, 'N', .52], [9, 'N', .51], 
                        [10, 'P', .40]])
    
    data = pd.DataFrame(results[:, 1:], index = results[:, 0], \
                        columns = ['actualClass', 'probability'])
    
    generate_TPR_FPR(data['actualClass'], data['probability'])
    generate_ROC_curve(data['actualClass'], data['probability'])
    
# Context the file is running in is __main__ 
if __name__ == "__main__":
    main()