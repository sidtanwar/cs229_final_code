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

filename_data = 'train_data_521.csv'
filename_label = 'train_label_521.csv'

meas_residuals_train, SNR_train, true_labels_train = util.read_from_file_in_SVM_format(filename_data, filename_label)

print(len(meas_residuals_train))
print(meas_residuals_train[0].shape)

# #########################################################################
# #########################################################################
# #########################################################################
# #########################################################################
# #########################################################################
# #########################################################################
# #########################################################################
# #########################################################################

# kernel = 'linear'
# fn = 'model_521_' + kernel + '.sav'

# im_classifier1 = SNAEKF()

# print('Started training linear kernel')
# im_classifier1.fit(meas_residuals_train, SNR_train, true_labels_train, kernel)
# pickle.dump(im_classifier1.clf, open(fn,'wb'))

# # test_loaded_model = pickle.load(open(fn,'rb'))

# print('Done training linear kernel')

# ###########################################################################################################

# print('Testing on train data')

# filename = 'results/linear/data_521_train.txt'

# predicted_labels_train = im_classifier1.predict(meas_residuals_train, SNR_train)
# acc, cc = im_classifier1.evaluate_classifier(predicted_labels_train, true_labels_train, verbose=True, filename=filename)

# print('Done Testing on train data')


# ###########################################################################################################

# filename_data = 'test_data_521.csv'
# filename_label = 'test_label_521.csv'
# filename = 'results/linear/data_521.txt'

# meas_residuals_test, SNR_test, true_labels_test = util.read_from_file_in_SVM_format(filename_data, filename_label)

# predicted_labels_test = im_classifier1.predict(meas_residuals_test, SNR_test)
# acc, cc = im_classifier1.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)

# ###########################################################################################################

# filename_data = 'test_data_500.csv'
# filename_label = 'test_label_500.csv'
# filename = 'results/linear/data_500.txt'

# meas_residuals_test, SNR_test, true_labels_test = util.read_from_file_in_SVM_format(filename_data, filename_label)

# predicted_labels_test = im_classifier1.predict(meas_residuals_test, SNR_test)
# acc, cc = im_classifier1.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)

# ###########################################################################################################

# filename_data = 'test_data_050.csv'
# filename_label = 'test_label_050.csv'
# filename = 'results/linear/data_050.txt'

# meas_residuals_test, SNR_test, true_labels_test = util.read_from_file_in_SVM_format(filename_data, filename_label)

# predicted_labels_test = im_classifier1.predict(meas_residuals_test, SNR_test)
# acc, cc = im_classifier1.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)

# ###########################################################################################################

# filename_data = 'test_data_005.csv'
# filename_label = 'test_label_005.csv'
# filename = 'results/linear/data_005.txt'

# meas_residuals_test, SNR_test, true_labels_test = util.read_from_file_in_SVM_format(filename_data, filename_label)

# predicted_labels_test = im_classifier1.predict(meas_residuals_test, SNR_test)
# acc, cc = im_classifier1.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)

# print('      ')
# print('      ')
# print('      ')
# print('      ')
# print('      ')
# print('    DONE TESTING LINEAR !  ')
# print('      ')
# print('      ')
# print('      ')
# print('      ')
# print('      ')


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################

kernel = 'rbf'
# gamma_list = [10,1,0.1,0.01]
gamma_list = [0.001]

for gamma in gamma_list:
    fn = 'model_521_' + kernel + str(gamma) + '.sav'

    im_classifier2 = SNAEKF()

    print('Started training rbf kernel')
    im_classifier2.fit(meas_residuals_train, SNR_train, true_labels_train, kernel, gamma)
    pickle.dump(im_classifier2.clf, open(fn,'wb'))
    print('Done training rbf kernel with gamma:', gamma)

    ###########################################################################################################

    print('Testing on train data')

    filename = 'results/rbf_varying_gamma/data_521_train_001.txt'

    predicted_labels_train = im_classifier2.predict(meas_residuals_train, SNR_train)
    acc, cc = im_classifier2.evaluate_classifier(predicted_labels_train, true_labels_train, verbose=True, filename=filename)

    print('Done Testing on train data')

    ###########################################################################################################

    filename_data = 'test_data_521.csv'
    filename_label = 'test_label_521.csv'
    filename = 'results/rbf_varying_gamma/data_521_001.txt'

    meas_residuals_test, SNR_test, true_labels_test = util.read_from_file_in_SVM_format(filename_data, filename_label)

    predicted_labels_test = im_classifier2.predict(meas_residuals_test, SNR_test)
    acc, cc = im_classifier2.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)

    ###########################################################################################################

    filename_data = 'test_data_500.csv'
    filename_label = 'test_label_500.csv'
    filename = 'results/rbf_varying_gamma/data_500_001.txt'

    meas_residuals_test, SNR_test, true_labels_test = util.read_from_file_in_SVM_format(filename_data, filename_label)

    predicted_labels_test = im_classifier2.predict(meas_residuals_test, SNR_test)
    acc, cc = im_classifier2.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)

    ###########################################################################################################

    filename_data = 'test_data_050.csv'
    filename_label = 'test_label_050.csv'
    filename = 'results/rbf_varying_gamma/data_050_001.txt'

    meas_residuals_test, SNR_test, true_labels_test = util.read_from_file_in_SVM_format(filename_data, filename_label)

    predicted_labels_test = im_classifier2.predict(meas_residuals_test, SNR_test)
    acc, cc = im_classifier2.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)

    ###########################################################################################################

    filename_data = 'test_data_005.csv'
    filename_label = 'test_label_005.csv'
    filename = 'results/rbf_varying_gamma/data_005_001.txt'

    meas_residuals_test, SNR_test, true_labels_test = util.read_from_file_in_SVM_format(filename_data, filename_label)

    predicted_labels_test = im_classifier2.predict(meas_residuals_test, SNR_test)
    acc, cc = im_classifier2.evaluate_classifier(predicted_labels_test, true_labels_test, verbose=True, filename=filename)

    print('      ')
    print('      ')
    print('      ')
    print('      ')
    print('      ')
    print('    DONE TESTING RBF with gamma:  ', gamma)
    print('      ')
    print('      ')
    print('      ')
    print('      ')
    print('      ')
