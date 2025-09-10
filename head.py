''' ---------------------------------------------------------------------------------------------------------------------------------------------------------
Author: Elijah Boon
Created: 2025-08-10
Description:
--------------------------------------------------------------------------------------------------------------------------------------------------------------''' 
import cv2 as c
import numpy as np
from utils import HeadLandmarks as hl
import time 
class HeadPoseEstimation:
    def __init__(self,camera_matrix,dist_coeffs,mildly_down,severely_down,hold_time, head_landmarks=0):
        self.model_points =head_landmarks or  hl.modelPoints()
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.mildly_down = mildly_down
        self.severely_down = severely_down
        self.looking_down = False
        self.down_start_time = None
        self.hold_time = hold_time
    
    def HTE(self, coords):
        image_points = hl.imagePoints(coords)
        success, rotation_vector, _ = c.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,  
            self.dist_coeffs,
            flags=c.SOLVEPNP_ITERATIVE
        )
        
        if success:
            rotation_mat, _ = c.Rodrigues(rotation_vector)
            pose_mat = c.hconcat((rotation_mat, np.zeros((3,1))))
            _, _, _, _, _, _, euler_angles = c.decomposeProjectionMatrix(pose_mat)
            pitch = float(euler_angles[0])
            return pitch
        return None

    def check_head_pose(self,pitch, curr=None):
        if curr is None:
            curr = time.time()
        if self.mildly_down <= pitch <= self.severely_down:
            if self.down_start_time is None:
                self.down_start_time = curr
            elif curr - self.down_start_time>=self.hold_time:
                self.looking_down = True
        else:
            self.down_start_time = None
            self.looking_down = False
        return self.looking_down, pitch

            
