import numpy as np
import util
from train_env import TrainSimEnv
from naebt import NAEKF
from snaekf import SNAEKF
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import random
import pickle

# filename_data = 'test_data_500_RAIM.csv'
# filename_label = 'test_label_500_RAIM.csv'
# meas_residuals_test, SNR_test, true_labels_test = util.read_from_file_in_SVM_format(filename_data, filename_label)

# #########################################
# #########################################
# #########################################

# fn = 'model_521_linear.sav'
# loaded_model = pickle.load(open(fn,'rb'))
# im_classifier1 = SNAEKF()
# im_classifier1.clf = loaded_model

# filename = 'results/RAIM/data_500_SVM_linear.txt'
# predicted_labels_test = im_classifier1.predict(meas_residuals_test, SNR_test)
# acc, cc = im_classifier1.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)

# #########################################
# #########################################
# #########################################

fn = 'model_521_rbf.sav'
loaded_model = pickle.load(open(fn,'rb'))

print(loaded_model.get_params())
# im_classifier2 = SNAEKF()
# im_classifier2.clf = loaded_model

# filename = 'results/RAIM/data_500_SVM_rbf.txt'
# predicted_labels_test = im_classifier2.predict(meas_residuals_test, SNR_test)
# acc, cc = im_classifier2.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)
