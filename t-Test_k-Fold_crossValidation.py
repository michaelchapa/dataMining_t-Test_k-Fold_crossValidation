import numpy as np
import pandas as pd

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
#   O   ROC_curve
# Notes:
#   The roc_curve() is restricted to the Binary Classification task
def generate_ROC_curve(actualClass, probability):
    # convert from Binary Nominal to Binary Numerical
    encodedClass = pd.get_dummies(actualClass) 
    actualClass = encodedClass['P']
    actualClass = actualClass.to_numpy()
    probability = probability.to_numpy()
    
    fpr, tpr, thresholds = roc_curve(actualClass, probability)
    

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