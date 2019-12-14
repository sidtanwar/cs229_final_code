import numpy as np
import util
from train_env import TrainSimEnv
from naebt import NAEKF
from snaekf import SNAEKF
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import random
from scipy.stats import chi2

############################################################ 
# GET NEW DATASET
# WILL TEST NN and SVM on this
############################################################ 

### initialize environment variables

pos_init_filename = 'pos_init.csv'
vel_filename = 'vel.csv'
sat_pos_init_filename = 'sat_data1.csv'
sigma_init_filename = 'sigma_init3.csv'
R_ekf_filename = 'r_ekf.csv'

pos_init = util.read_file(pos_init_filename, 'p')
vel = util.read_file(vel_filename, 'v')
sat_pos_init = util.read_file(sat_pos_init_filename, 's')
sigma_init = util.read_file(sigma_init_filename, 'c')   # initial pos covar
R_ekf = util.read_file(R_ekf_filename,'r')   # process noise

# constants
Ndt = 10   # num predict steps (over a measurement trajectory set)
Ntraj = 100   # num of (Ndt sets/steps) that form a complete trajectory
            # each Ndt set achieves a binary classification
delta_t = 0.1   # time between predicts
Nsat = 6 
Qa = 25
Qb = 45000
v_sigma = 0.1

### Parameters for the algorithm
Nsam_train = 10000
Nsam_test = 100
tm = 1   # time between measurements
Nm = int(Ndt * delta_t / tm)   # num meas for 1 classification
offset = 0
kernel = 'linear'

# For SNAEKF
k_e = 1.5 # k = 1.1875 for NAEKF
k_n = 2.0
k_u = 5.0
PL = np.array([k_e, k_n, k_u])  # decrease it for more 1s
epsilon = 0.00001 # decrease it for more 1s

faulty_range_biases = 50*np.ones(Nsat)
healthy_range_biases = 0*np.ones(Nsat)

sigma_range = 5

faulty_SNR_LB = 35*np.ones(Nsat)
faulty_SNR_UB = 40*np.ones(Nsat)
healthy_SNR_LB = 35*np.ones(Nsat)
healthy_SNR_UB = 45*np.ones(Nsat)

train_env = TrainSimEnv(pos_init = pos_init, vel = vel,
                        sat_pos_init = sat_pos_init,
                        Ndt = Ndt, Ntraj = Ntraj, sigma_init = sigma_init, 
                        delta_t = delta_t, Nsat= Nsat, R_ekf = R_ekf, 
                        Qa = Qa, Qb = Qb, v_sigma = v_sigma)

train_env.gen_agent_traj()
train_env.gen_sat_traj()

im_classifier = SNAEKF()

print('Type of classifier:', type(im_classifier))

fault_probs = np.array([0.5, 0, 0, 0, 0, 0])
# Get training data set and train
print('Start of training')
if(type(im_classifier)==NAEKF):
    # residuals_train, SNR_train = im_classifier.sample(Nsam_train, LB_pseudo, UB_pseudo, LB_snr, UB_snr, Nsat, Nm, tm)
    print('oh o you dumb')
elif(type(im_classifier)==SNAEKF):
    residuals_train, SNR_train = im_classifier.sample(Nsam_train, Nsat, Nm, \
               faulty_range_biases, healthy_range_biases, sigma_range, \
               faulty_SNR_LB, faulty_SNR_UB, healthy_SNR_LB, healthy_SNR_UB,\
               fault_probs)

print('  Completed sampling')

true_labels_train, meas_residuals_train, ps, sigmas, p_Nms = im_classifier.get_true_labels(residuals_train, SNR_train, train_env, tm, offset, PL, epsilon)
print('  Computed true labels for training set')
# print('    True labels:', true_labels_train)
print('Prevelence in training data', np.sum(true_labels_train)/Nsam_train)

filename_data = 'test_data_500_RAIM.csv'
filename_label = 'test_label_500_RAIM.csv'

util.save_to_file_in_NN_format(filename_data, filename_label, Nsat, Nsam_train, meas_residuals_train, SNR_train, true_labels_train)

############################################################ 
# DO RAIM
############################################################ 

df = Nsat - 4
if df < 0:
    print('You just plain messed up bro. Nsats ahve to be greater than 4 for RAIM to work as expected')

threshold = chi2.ppf(1 - epsilon, df)

print('  Doing RAIM')
RAIM_true_labels_train = util.simple_RAIM(meas_residuals_train, threshold, sigma_range, train_env, p_Nms, tm, offset)
print('  Computed true labels for training set')
# print('    RAIM labels:', RAIM_true_labels_train)

############################################################ 
# RAIM STATISTICS
############################################################ 

print('Computing RAIM Statistics')

filename = 'results/RAIM/data_500.txt'
acc, cc = im_classifier.evaluate_classifier(RAIM_true_labels_train, true_labels_train, verbose=True, filename=filename)
