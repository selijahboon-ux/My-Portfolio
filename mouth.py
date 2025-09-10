''' ---------------------------------------------------------------------------------------------------------------------------------------------------------
Author: Elijah Boon
Created: 2025-08-12
Description: This class contains all the mouth-related features for the AI-driven drowsiness detection system.
             - Mouth-Aspect-Ratio (MAR) measures openness using vertical and horizontal mouth distances via Euclidean distance.
             - Yawn detection based on MAR threshold and time duration (typically 4-7 seconds).
--------------------------------------------------------------------------------------------------------------------------------------------------------------'''
import time
from utils import EuclideanDistance as eu
from utils import MouthLandmarks as ml


class MouthFeatures:
    def __init__(self, mouth_threshold, mouth_landmarks=None, yawn_min=0, yawn_max=0):
        self.mouth_threshold = mouth_threshold#Mouth openness threshold.
        self.mouth_landmarks = mouth_landmarks or ml.mouthLandmarks()#Mouth Landmarks.
        self.yawn_start = None#Yawn timer.
        self.yawning = False#Yawn states(close or open).
        self.mar_value = 0.0#Value of the mar
        self.yawn_min = yawn_min#Minimum threshold of yawn duration.
        self.yawn_max = yawn_max#Maximum threshold for yawn duration.

    def reset(self):#Resets man.
        self.mar_value = 0.0
        self.yawn_start = None
        self.yawning = False
        
    def calc_mar(self, landmarks):#Computes Mouth Aspect Ratio Value.
        mouth = [landmarks[i] for i in self.mouth_landmarks]
        A = eu.euclidean(mouth[2], mouth[3])
        B = eu.euclidean(mouth[4], mouth[5])
        C = eu.euclidean(mouth[6], mouth[7])
        D = eu.euclidean(mouth[0], mouth[1])
        self.mar_value = (A + B + C) / (2.0 * D)
        return self.mar_value

    def check_yawn(self, mar, curr=None):
        if curr is None:
            curr = time.time()

        if mar > self.mouth_threshold:
            if self.yawn_start is None:
                self.yawn_start = curr
            else:
                yawn_duration = curr - self.yawn_start
                if yawn_duration >= self.yawn_min:
                    self.yawning = True
                elif yawn_duration > self.yawn_max:
                    self.yawning = False
                    self.yawn_start = None
        else:
            self.yawning = False
            self.yawn_start = None

        return self.yawning


