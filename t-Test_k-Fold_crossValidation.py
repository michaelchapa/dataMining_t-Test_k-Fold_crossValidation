import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder

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
    
    # False-Pos Rate, True-Pos Rate
    fpr, tpr, thresholds = roc_curve(actualClass, probability, pos_label = 1)
    xY = np.arange(0, 1.1, .1)
    
    plt.figure()
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Probabilistic classifier')
    plt.plot(fpr, tpr, 'b-', xY, xY, 'g--')
    plt.show()

def main():
    features = ['actualClass', 'probability']
    results = np.array([[1, 'P', .95], [2, 'N', .85], [3, 'P', .78], 
                                  [4, 'P', .66], [5, 'N', .60], [6, 'P', .55], 
                                  [7, 'N', .53], [8, 'N', .52], [9, 'N', .51], 
                                  [10, 'P', .40]])
    
    data = pd.DataFrame(results[:, 1:], \
                        index = results[:, 0], columns = features)
    
    generate_ROC_curve(data['actualClass'], data['probability'])
    
# Context the file is running in is __main__ 
if __name__ == "__main__":
    main()