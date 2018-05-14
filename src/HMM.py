import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
from sklearn.metrics import confusion_matrix
# from hmm_class import HMM
from hmm_class_elan import HMM

# SAMPLE CODE: EXECUTION WILL INITIALLY FAIL

# Define signals
signal1 = np.array([[1.,  1.1,  0.9, 1.0, 0.0,  0.2,  0.1,  0.3,  3.4,  3.6,  3.5]])
signal2 = np.array([[0.8, 1.2, 0.4, 0.2, 0.15, 2.8, 3.6]])

# Collect training data together
toy_data = np.hstack([signal1, signal2])
toy_lengths = [11, 7]

# Create and fit HMM model to data
toy_hmm = HMM()
toy_hmm.fit(toy_data, toy_lengths, 3)

toy_means = [d.get_mean() for d in toy_hmm.dists]
toy_covs = [d.get_cov() for d in toy_hmm.dists]
print('Transition probabilities: ')
print(toy_hmm.trans)
print('Means: ')
print(toy_means)
print('Covariances: ')
print(toy_covs)

# d# EXPECTED OUTPUT OF CELL ABOVE WHEN CODE IS FUNCTIONING CORRECTLY
#
# Transition probabilities:
# [[ 0.66666667  0.33333333  0.          0.        ]
#  [ 0.          0.71428571  0.28571429  0.        ]
#  [ 0.          0.          0.6         0.4       ]
#  [ 1.          0.          0.          0.        ]]
# Means:
# [array([ 1.]), array([ 0.19285714]), array([ 3.38])]
# Covariances:
# [array([[ 0.02]]), array([[ 0.01702381]]), array([[ 0.112]])]