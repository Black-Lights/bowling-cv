import cv2
import numpy as np
from config.config import Config

class BowlingKalmanFilter:
    def __init__(self):
        # 4 State variables: [x, y, vx, vy]
        # 2 Measurement variables: [x, y]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition Matrix (F)
        # x_new = x_old + vx
        # y_new = y_old + vy
        self.kf.transitionMatrix = np.array(,
            ,
            ,
            , np.float32)
        
        # Measurement Matrix (H)
        # We define that we are only measuring x and y
        self.kf.measurementMatrix = np.array(,
            , np.float32)
        
        # Process Noise Covariance (Q)
        # Defines how much the system can deviate from the model
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * Config.KF_PROCESS_NOISE
        
        # Measurement Noise Covariance (R)
        # Defines how noisy our inputs are
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * Config.KF_MEASUREMENT_NOISE
        
        # Initial Error Covariance (P)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def initialize(self, x, y):
        """ Reset the filter to a specific location with 0 velocity """
        self.kf.statePost = np.array([[x], [y], , ], np.float32)
        
    def predict(self):
        """ Return the predicted next state (x, y) """
        pred = self.kf.predict()
        return int(pred), int(pred)
    
    def correct(self, x, y):
        """ Incorporate a new real measurement """
        meas = np.array([[x], [y]], np.float32)
        self.kf.correct(meas)
        # Return the corrected state (best estimate)
        return int(self.kf.statePost), int(self.kf.statePost)