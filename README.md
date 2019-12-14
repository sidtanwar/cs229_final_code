## Scripts:

# constants.py

This script contains some GPS constants used by the util script

# gen_data.py

This is the script that is run to generate the train and test data and contains all of the simulation parameters used. Uncomment regions of the script based on the need.

# gps_im_classifier.py

This is the parent class which provides the structure for the data generation and fitting. It needs a state estimator, integrity monitor, and classifier, along with a sampler.

# naebt.py

This script is the main script which consists the class that we used for this project. This has EKF, IM, Sampler and SVM. 

# RAIM.py

Run this script to generate RAIM results.

# snaekf.py

This script consists of the sampler that assumes a gaussian prior on the received measurements

# svm.py

Run this script to train and test the svm on a bunch of data generated from the gen_data.oy script.

# SVMtest.py

 Run this script to check the parameters of the SVM

# SVM_to_RAIM.py

Run this script to load a trained SVM model and check it on a dataset with a single faulty satellite to compare it with the results of RAIM.

# train_env.py

This uses the simulation parameters and computes satellite and agent trajectory to be used for data generation. 

# util.py

Consists of utility functions as well as the main RAIM algorithm.


################################################

Remember to always copy the data in the right folder before running the scripts.


