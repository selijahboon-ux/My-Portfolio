''' ---------------------------------------------------------------------------------------------------------------------------------------------------------
Author: Elijah Boon
Created: 2025-08-12
Description: dont know what to say
--------------------------------------------------------------------------------------------------------------------------------------------------------------'''
import numpy as np
import cv2 as c

class EuclideanDistance:
    @staticmethod
    def euclidean(p1,p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

class EyeLandmarks:
    @staticmethod
    def leftEyeLandmarks(): 
        return[33, 160, 158, 133, 153, 144] 
    
    @staticmethod
    def rightEyeLandmarks():
         return [362, 385, 387, 263, 373, 380]

class MouthLandmarks:
    @staticmethod
    def mouthLandmarks():
       return  [61, 291, 13, 14, 81, 84, 178, 17]

class DrawingUtils:
    @staticmethod
    def draw_landmarks_subset(frame, landmarks, indices, color=(0, 255, 0), radius=2, connect=False):
        points = [landmarks[i] for i in indices]
        for point in points:
            c.circle(frame, point, radius, color, -1)
        if connect:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            c.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


