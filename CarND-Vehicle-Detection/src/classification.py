import time
from sklearn.svm import LinearSVC
import os, pdb
import numpy as np

def get_classifer(scaled_x_train, scaled_x_test, y_train, y_test):
    # Parameter Tuning
    C_list = [10, 1, 0.1]
    for C in C_list:
        svc = LinearSVC(C=C) # Use a linear SVC 
        t=time.time()
        svc.fit(scaled_x_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC with C=%f...'%C)
        print('Test Accuracy of SVC = ', round(svc.score(scaled_x_test, y_test), 4))

    # Train model using all data
    svc = LinearSVC(C=10)
    svc.fit(np.concatenate((scaled_x_train, scaled_x_test)), np.concatenate((y_train, y_test)))
    return svc