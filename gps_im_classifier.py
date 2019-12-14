from abc import ABC, abstractmethod
import numpy as np
import util

class GPS_IM_Classifier(ABC):
    
    # def __init__(self):
    #     self.value = 1
    #     super().__init__()

    @abstractmethod
    def predict(self):
        """
        gives 1 or 0 given feature vectors with current model fit
        """
        pass

    @abstractmethod
    def sample(self):
        pass
    
    @abstractmethod
    def estimate(self):
        pass
        
    @abstractmethod
    def im(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate_classifier(self):
        """
        Does post processing like evaluate accuracy etc. on the model predictions
        """
        pass