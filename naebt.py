import numpy as np
import util
from gps_im_classifier import GPS_IM_Classifier 
import sklearn.svm
import sklearn.metrics
import scipy

import sys
import random

# random.seed(3000)
np.set_printoptions(threshold=sys.maxsize)
#import sklearn.svm import SVC
#from sklearn.metrics import classification_report, confusion_matrix

class NAEKF(GPS_IM_Classifier):

    def __init__(self):
        """
        Constructor
        """

        self.clf = None     # Classifier object



    def predict(self, residuals, SNR):
        """
        Uses trained SVM to make predictions on inputs
        Input: residuals, SNR -- lists of numpy arrays from sample()
        Output: binary labels for each example (n_examples, )
        """
        # Convert residuals and SNR to matrix of dimension (Nsam, dim)
        # where dim = 2 * (Nsat x Nm)
        Nsam = len(residuals)
        Nsat = residuals[0].shape[0]
        Nm = residuals[0].shape[1]
        dim = 2 * Nsat * Nm # dimension of theta
        X_test = np.nan * np.ones([Nsam, dim]) # numpy array of test examples

        for i in range(Nsam):
            curr_res = residuals[i].flatten(); curr_snr = SNR[i].flatten()
            concatenated = np.array([curr_res, curr_snr])
            X_test[i,:] = concatenated.flatten()

        if self.clf is None:
            print('Throw an error (classifier clf not yet fitted)')
            # For now, return empty labels (None)
            return None

        labels = self.clf.predict(X_test)
        return labels

        

    def sample(self, Nsam, LB_pseudo, UB_pseudo, LB_snr, UB_snr, Nsat, Nm, tm):
        """
        Generates samples based on some models or something
        Input: Nsam -- number of samples,
                LB UB -- lower / upper bd on each pseudorange (m) and SNR value (dBHz),
                Nsat, Nm, tm
        (previous input had listed distribution params instead of LB, UB)
        Output: 2 lists of length Nsam, residuals - Nsat x Nm, SNR - Nsat x Nm
        """
        
        #LB_snr_arr = LB_snr * np.ones(Nsat * Nm)
        #UB_snr_arr = UB_snr * np.ones(Nsat * Nm)

        # Initialize lists for residuals and SNR (length of Nsam)
        residuals = [None] * Nsam
        SNR = [None] * Nsam
        
        # For each sample:
        for i in range(Nsam):
            residuals[i] = np.random.uniform(LB_pseudo, UB_pseudo, size=[Nsat,Nm])
            SNR[i] = np.random.uniform(LB_snr, UB_snr, size=[Nsat,Nm])
        
        # ## plotting debug
        # # show random residual
        # rand_num = np.random.randint(Nsam)
        # rand_sat = np.random.randint(Nsat)
        # sample_residuals = np.array([(residuals[i])[rand_sat,:] for i in range(Nsam)])
        # sample_SNRs = np.array([(SNR[i])[rand_sat,:] for i in range(Nsam)])
        # util.myplot(range(Nsam),sample_residuals,0,Nsam, LB_pseudo, UB_pseudo)

        # # show random SNR
        # util.myplot(range(Nsam),sample_SNRs,0,Nsam, LB_snr, UB_snr)

        return residuals, SNR 
        
    
    def estimate(self, t, meas_pseudo, meas_snr, tm, delta_t, simEnv, offset):
        """
        runs an EKF
        Input: t = Nm x tm, 
                meas_pseudo -- numpy array of measurement (NSat x Nm),
                meas_SNR -- numpy array of measurement SNR (Nsat x Nm), 
                tm -- time between each individual measurement, 
                delta_t -- time between predict step (can be smaller than tm), 
                simEnv -- TrainSimEnv object
                offset -- num measurement sets (each corresponds to Ndt measurements) to skip 
                            (if this is too tedious just put it to zero)
        Output: 2 arrays of length Ndt = t/delta_t 
                p : (3 x Ndt)
                sigma : (3 x 3 x Ndt)  
                meas_residual : Nsat x Nm
                p_NM : 3 x Nm
        """
        # simEnv = simEnv_orig
        sat_pos = simEnv.sat_pos # Get satellite positions ( list of Nsat, each (3 x (Ndt*Ntraj)) )
        Nsat = simEnv.Nsat
        vel = simEnv.vel # Get velocity values (3 x (Ndt*Ntraj))
        Ndt = int(t / delta_t) # number of predict steps
        Nm = int(t / tm) # number of measurements
        R_ekf = simEnv.R_ekf # process noise matrix
        Qa = simEnv.Qa; Qb = simEnv.Qb
        p = np.nan * np.ones([3, Ndt])
        sigma = np.nan * np.ones([3, 3, Ndt])

        curr_pos = (simEnv.pos_init).copy() # Get initial position (3,)
        curr_sigma = (simEnv.sigma_init).copy()
        im = 0 # measurement index

        meas_residuals = np.zeros((Nsat,Nm))
        p_Nm = np.zeros((3,Nm))
        for i in range(Ndt):
            # Prediction step 
            curr_pos += (vel[:, Ndt*offset + i] + np.random.normal(0,simEnv.v_sigma, size = (3,))) * delta_t
            curr_sigma += R_ekf * (delta_t**2)
            
            if ((i+1)*delta_t)%tm == 0:
                # Update step
                for j in range(Nsat): # for each satellite
                    psat = sat_pos[j][:,Ndt*offset + i]
                    rho = meas_pseudo[j,im] 
                    snr = meas_snr[j,im]
                    dist = np.linalg.norm(curr_pos - psat)
                    H = (curr_pos - psat) / dist
                    Q = Qa + Qb * (10**(-snr/10))
                    # print(Q)
                    K = (curr_sigma @ H) / ((H @ curr_sigma @ H) + Q)
                    p_Nm[:,im] = curr_pos
                    curr_pos += K*(rho - dist)

                    meas_residuals[j,im] = rho - dist
                    
                    # print('Pre', curr_sigma)
                    curr_sigma = (np.eye(3) - np.outer(K, H)) @ curr_sigma
                    # print('Post', curr_sigma)
                im += 1 # increment measurement index
            p[:,i] = curr_pos
            sigma[:,:,i] = curr_sigma 

        return p, sigma, meas_residuals, p_Nm
        
    def im(self, PL, epsilon, p, sigma, simEnv, offset):
        """
        runs an IM algorithm
        Input: PL -- 3-dimensional numpy array for protection level (in m) (3,), 
                epsilon -- related to integrity incident, i.e. when P(|X - Xtrue|> PL)>epsilon
                p, sigma -- output of EKF (3 x Ndt) and (3 x 3 x Ndt), 
                simEnv -- TrainSimEnv object, 
                offset -- num measurement sets (each corresponds to Ndt measurements) to skip 
                            (if this is too tedious just put it to zero) 
        Output: True / False
        TODO: Convert from ECEF positions to ENU positions before applying protection
        TODO: Convert from ENU frame to a frame which is rotated toward vel vector dir
        """
        if p.shape[1] != sigma.shape[2]:
            print('Throw an error (length of pos (p) and covariance (sigma) not the same)')
            return None
        
        Ndt = p.shape[1]
        for i in range(Ndt):
            # print(i)

            # true position in ecef
            tp = simEnv.pos[:, offset*Ndt + i]

            # convert estimate position mean to ENU at tp 

            tp_re = np.reshape(tp, [-1,1])
            p_re = np.reshape(p[:,i], [-1,1])
            p_enu, R_enu = util.ECEF_to_ENU(tp_re, p_re)

            mu_enu = R_enu @ (p_re - tp_re)
            sigma_enu = R_enu @ sigma[:,:,i] @ R_enu.transpose()

            low = -1 * PL
            high = PL
            prob, imvn = scipy.stats.mvn.mvnun(low,high,mu_enu,sigma_enu)

            # mu = p[:, i]
            # low = simEnv.pos[:, offset*Ndt + i] - PL
            # upp = simEnv.pos[:, offset*Ndt + i] + PL
            # prob, imvn = scipy.stats.mvn.mvnun(low,upp,mu,sigma[:, :, i]) 
            if (1 - prob) > epsilon:
                # print('exited in i =', i)
                return True # if fault exists, return true
        return False # if exited without returning, no fault occurred

    def fit(self, residuals, SNR, y, kernel, gamma = 1.0):
        """
        Fit an SVM to the data
        Input: residuals, SNR -- lists of numpy arrays from sample()
                y -- numpy array of corresponding true labels (Nsam,),
                kernel -- kernel type (str)
        Output: None, however it updates Model (classifier object)
        """
        # Convert residuals and SNR to matrix of dimension (Nsam, dim)
        # where dim = 2 * (Nsat x Nm)
        Nsam = len(residuals)
        Nsat = residuals[0].shape[0]
        Nm = residuals[0].shape[1]
        dim = 2 * Nsat * Nm
        X = np.nan * np.ones([Nsam, dim]) # numpy array of sampled theta/meas

        for i in range(Nsam):
            curr_res = residuals[i].flatten(); curr_snr = SNR[i].flatten()
            concatenated = np.array([curr_res, curr_snr])
            X[i,:] = concatenated.flatten()

        # print('X', X[0,:])
        if kernel == 'rbf':
            self.clf = sklearn.svm.SVC(kernel=kernel, gamma = gamma)
        else:
            self.clf = sklearn.svm.SVC(kernel=kernel)
        #fitted_model = self.clf.fit(X, y) # does update self.clf as well
        self.clf.fit(X, y) # does update self.clf as well

        return

    def evaluate_classifier(self, y_pred, y_test, verbose=True, filename = None):
        """
        Does post processing like evaluate accuracy etc. on the model predictions
        Input: y_pred -- predictions from fit
                y_test -- true values
                verbose -- prints results
        Output: Accuracy and other statistics
        """

        if len(y_test) != len(y_pred):
            print('Throw an error (length of y_test and y_pred not equal)')
            return None

        # # compute accuracy
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        class_conf_norm = sklearn.metrics.confusion_matrix(y_test,y_pred)/len(y_test)

        if verbose:
            tp = 0
            tn = 0
            fn = 0
            fp = 0

            for i in range(len(y_pred)):
                if y_pred[i] == True: 
                    if y_test[i] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if y_test[i] == 0:
                        tn += 1
                    else:
                        fn += 1

            total_examples = len(y_pred)
            print('  ')
            print('  ')
            print('  ')
            print('  ')
            print('Total Examples', total_examples)
            print('Confusion Matrix:')
            print('  ')
            print(tp,'   ',fp)
            print(fn,'   ',tn)
            print('  ')
            print('  ')
            print('Accuracy: ', (tp+tn)/total_examples)
            if (tp + fn ) != 0:
                print('Recall: ', (tp)/(tp+fn))
            else:
                print('Recall: Not well defined')
            if (tp + fp) != 0:
                print('Precision: ', tp/(tp+fp))
            else:
                print('Precision: Not well defined')    
            print('  ')
            print('  ')
            print('Prevelence: ', np.sum(np.array(y_test))/total_examples)

            if filename is not None:
                file_o = open(filename,'w')
                print('Total Examples', total_examples, file = file_o)
                print('Confusion Matrix:', file = file_o)
                print('  ', file = file_o)
                print(tp,'   ',fp, file = file_o)
                print(fn,'   ',tn, file = file_o)
                print('  ', file = file_o)
                print('  ', file = file_o)
                print('Accuracy: ', (tp+tn)/total_examples, file = file_o)
                if (tp + fn ) != 0:
                    print('Recall: ', (tp)/(tp+fn), file = file_o)
                else:
                    print('Recall: Not well defined', file = file_o)
                if (tp + fp) != 0:
                    print('Precision: ', tp/(tp+fp), file = file_o)
                else:
                    print('Precision: Not well defined', file = file_o)   
                print('  ', file = file_o)
                print('  ', file = file_o)
                print('Prevelence: ', np.sum(np.array(y_test))/total_examples, file = file_o)
                file_o.close()

        # # show confusion matrix and classification report
        # if verbose:
        #     print('Evaluation of classifier:')
        #     print(' ')
        #     print('Accuracy:', accuracy)
        #     print('Class confusion matrix (normalized):')
        #     print(class_conf_norm)
        #     print('sklearn classification report:')
        #     print(sklearn.metrics.classification_report(y_test,y_pred))

        #     if filename is not None:
        #         file_o = open(filename,'w')
        #         print('Evaluation of classifier:', file = file_o)
        #         print(' ', file = file_o)
        #         print('Accuracy:', accuracy, file = file_o)
        #         print('Class confusion matrix (normalized):', file = file_o)
        #         print(class_conf_norm, file = file_o)
        #         print('sklearn classification report:', file = file_o)
        #         print(sklearn.metrics.classification_report(y_test,y_pred), file = file_o)
        #         file_o.close()
    

        return accuracy, class_conf_norm

    def get_true_labels(self, residuals, SNR, simEnv, tm, offset, PL, epsilon):
        """ 
        Input: residuals -- samples of residuals, list of length Nsam each of (Nsat x Nm),
                SNR -- samples of SNR values, each numpy array of meas SNR (Nsat x Nm),
                simEnv -- TrainSimEnv object,
                tm -- time between measurements,
                offset -- number of measurement sets to skip (Ndt measurements to skip),
                PL -- protection level (m),
                epsilon -- fault probability threshold
        Output: labels -- true labels for each of the sample residuals
                meas_residuals -- list (Nsam) of measured residuals for each true set of residuals each element (Nsat x Nm) 
        """

        # Get values from simEnv
        delta_t = simEnv.delta_t

        # preprocess to get pseudoranges
        pseudoranges = util.preprocessor(simEnv, residuals, tm, offset)
        
        Nsam = len(pseudoranges)
        Nm = pseudoranges[0].shape[1]
        t = Nm * tm
        labels = np.nan * np.ones(Nsam)

        meas_residuals = [None] * Nsam
        ps = [None] * Nsam
        sigmas = [None] * Nsam
        p_Nms = [None] * Nsam
        # go through each sample i
        for i_sam in range(Nsam):
            # EKF
            meas_pseudo = pseudoranges[i_sam]
            meas_snr = SNR[i_sam]
            p, sigma, meas_residual, p_Nm = self.estimate(t, meas_pseudo, meas_snr, tm, delta_t, simEnv, offset)
            meas_residuals[i_sam] = meas_residual
            ps[i_sam] = p
            sigmas[i_sam] = sigma
            p_Nms[i_sam] = p_Nm
            # get label by IM
            labels[i_sam] = self.im(PL, epsilon, p, sigma, simEnv, offset)

        # ## TODO delete later
        # meas_pseudo_old = pseudoranges[0]
        # meas_snr_old = SNR[0]
        # p_old, sigma_old = self.estimate(t, meas_pseudo_old, meas_snr_old, tm, delta_t, simEnv, offset)
        # epsilon_del = 0.01
        # # go through each sample i
        # for i_sam in range(Nsam):
        #     # EKF
        #     # meas_pseudo = pseudoranges[i_sam]
        #     # meas_snr = SNR[i_sam]
        #     meas_pseudo = pseudoranges[0]
        #     meas_snr = SNR[0]
        #     # if ( (np.linalg.norm(meas_pseudo-meas_pseudo_old) > epsilon_del) or (np.linalg.norm(meas_snr-meas_snr_old) > epsilon_del)):
        #     #     print('WTF ', i_sam)
        #     #     break 
        #     p, sigma = self.estimate(t, meas_pseudo, meas_snr, tm, delta_t, simEnv, offset)
        #     if ( (np.linalg.norm(p-p_old) > epsilon_del) or (np.linalg.norm(sigma-sigma_old) > epsilon_del)):
        #         print('EKF done messed up bro ', i_sam)
        #         break 
        #     # get label by IM
        #     labels[i_sam] = self.im(PL, epsilon, p, sigma, simEnv, offset)
        
        return labels, meas_residuals, ps, sigmas, p_Nms