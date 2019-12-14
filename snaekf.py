import numpy as np
import util
from naebt import NAEKF
import scipy

# Simulated Non-active EKF
class SNAEKF(NAEKF):

    # def sample(self, Nsam, LB_pseudo, UB_pseudo, LB_snr, UB_snr, Nsat, Nm, tm, \
    #            faulty_range_biases, faulty_SNR_biases, \
    #            healthy_range_biases, healthy_SNR_biases, \
    #            fault_probs, sigma_range, sigma_SNR):
    def sample(self, Nsam, Nsat, Nm, \
               faulty_range_biases, healthy_range_biases, sigma_range, \
               faulty_SNR_LB, faulty_SNR_UB, healthy_SNR_LB, healthy_SNR_UB,\
               fault_probs):
    
        """
        Generates samples based on some models or something
        Input: Nsam -- number of samples,
                Nsat, Nm,
                faulty_range_biases -- vector (length Nsat) of range residual (m) biases in faulty satellite case, 
                healthy_range_biases -- vector (length Nsat) of range residual (m) biases in healthy satellite case,
                sigma_range, sigma_SNR -- standard deviation of range (m),
                faulty_SNR_LB, faulty_SNR_UB -- lower/upper bound of SNR measurements in faulty case (dBHz),
                healthy_SNR_LB, healthy_SNR_UB -- lower/upper bound of SNR measurements in healthy case (dBHz),
                fault_probs -- vector of probability faults
        Output: 2 lists of length Nsam, residuals - Nsat x Nm, SNR - Nsat x Nm
        """
        
        # Throw informal error if length of biases / fault probabilities is not Nsat
        if len(faulty_range_biases)!=Nsat or len(healthy_range_biases)!=Nsat or len(fault_probs)!=Nsat:
            print('Informal error. Length of bias vectors and/or fault probability vector not same as Nsat.')
            return None
        if len(faulty_SNR_LB)!=Nsat or len(faulty_SNR_UB)!=Nsat \
            or len(healthy_SNR_LB)!=Nsat or len(healthy_SNR_UB)!=Nsat:
            print('Informal error. Length of SNR lower/upper bound vector(s) not same as Nsat.')
            return None
        if any(faulty_SNR_LB<0) or any(faulty_SNR_UB<0) or any(healthy_SNR_LB<0) or any(healthy_SNR_UB<0):
            print('Informal error. SNR lower and upper bounds must be non-negative.')
            return None
        if any(faulty_SNR_UB - faulty_SNR_LB < 0) or any(healthy_SNR_UB - healthy_SNR_LB < 0):
            print('Informal error. SNR upper bounds must be >= lower bounds.')
            return None
        if sigma_range < 0:
            print('Informal error. Standard deviation of range must be non-negative') 
            return None
        
        # Initialize lists for residuals and SNR (length of Nsam)
        residuals = [None] * Nsam
        SNR = [None] * Nsam
        
        # For each sample:
        for i in range(Nsam):
            
            # Get sample of faulty satellites
            faulty_sats = scipy.stats.bernoulli.rvs(fault_probs, size=[1,Nsat])[0]
            healthy_sats = 1-faulty_sats
            
            # Get mean vector and covariance matrix 
            mean_res = faulty_range_biases*faulty_sats + healthy_range_biases*healthy_sats
            cov_res = sigma_range*sigma_range*np.eye(Nsat)
            #print('mean_res', mean_res)
                      
            # Assuming a faulty/healthy satellite stays faulty/healthy for the sequence of Nm measurements
            residuals[i] = np.random.multivariate_normal(mean_res, cov_res, Nm).transpose()
            
            # Get lower and upper bounds of SNR value per satellite
            LB_SNR = faulty_SNR_LB*faulty_sats + healthy_SNR_LB*healthy_sats
            UB_SNR = faulty_SNR_UB*faulty_sats + healthy_SNR_UB*healthy_sats
            # print('LB_SNR', LB_SNR)
            # print('UB_SNR', UB_SNR)
            SNR[i] = scipy.stats.uniform.rvs(LB_SNR, UB_SNR-LB_SNR, size=[Nm, Nsat]).transpose()
            # print('SNR[i]', SNR[i])
            # print('SNR[i].shape', SNR[i].shape)
            
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
