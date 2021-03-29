import cv2
import numpy as np

"""
Code for utilizing Kalman filters on data streams
"""


class kalman():
    """For 2d shapes"""

    first = True # first time only!

    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],
                                             [0,1,0,0]],np.float32)

        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                            [0,1,0,1],
                                            [0,0,1,0],
                                            [0,0,0,1]],np.float32)

        self.kalman.processNoiseCov = np.array([[1,0,0,0],
                                           [0,1,0,0],
                                           [0,0,1,0],
                                           [0,0,0,1]],np.float32) * 0.03

    def filter_pos(self,x,y):

        if self.first:
            A = self.kalman.statePost
            A[0:4] = np.array([[np.float32(x)], [np.float32(y)],[0],[0]])
            
            self.kalman.statePost = A
            self.kalman.statePre = A
            self.first = False


        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(current_measurement)
        prediction = self.kalman.predict()
        #print
        #self.box = [int(prediction[0]), int(prediction[1]), 0,0]

        return (int(prediction[0]), int(prediction[1]))

    def filter_box(self,box):

        if self.first:
                A = self.kalman.statePost
                A[0:4] = np.array([[np.float32(box[0])], [np.float32(box[1])],[0],[0]])
                # A[4:8] = 0.0
                self.kalman.statePost = A
                self.kalman.statePre = A
                self.first = False

        current_measurement = np.array([[np.float32(box[0])], [np.float32(box[1])]])
        self.kalman.correct(current_measurement)
        prediction = self.kalman.predict()
        #print(int(prediction[0]), int(prediction[1]))
        box = [int(prediction[0]), int(prediction[1]), box[2], box[3]]
        
        return box