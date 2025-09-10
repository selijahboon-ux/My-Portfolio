import os
import cv2 as c
import mediapipe as mp
import time
import numpy as np
import pandas as pd

from eye import EyeFeatures
from mouth import MouthFeatures
from utils import EyeLandmarks as el
from utils import MouthLandmarks as ml
from utils import DrawingUtils as du

def process_camera():
    rows = []

    # Open webcam directly
    cap = c.VideoCapture(0)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )

    # Landmarks
    ml_landmarks = ml.mouthLandmarks()
    el_left_eye_landmarks = el.leftEyeLandmarks()
    el_right_eye_landmarks = el.rightEyeLandmarks()

    '''
    
    # Camera calibration
    ret, frame = cap.read()
    if not ret:
        return rows

    h, w, _ = frame.shape
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    
    
    
    '''
    

    # Feature classes
    eye_features = EyeFeatures(close_threshold=0.18, open_threshold=0.20, eye_duration=15)
    mouth_features = MouthFeatures(mouth_threshold=0.60, yawn_min=2, yawn_max=7)
    
    # Head pose parameters
    '''
    head_down_start_time = None
    head_down_threshold = -40     
    min_down_duration = 10
    head_state = "UNKNOWN" 
  
    
    
    '''
    perclos_threshold = 0.15
   

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = c.flip(frame, 1)
        frame_rgb = c.cvtColor(frame, c.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
            
            # Draw subsets
            du.draw_landmarks_subset(frame, landmarks, el_left_eye_landmarks, color=(255,255,255), radius=2, connect=True)
            du.draw_landmarks_subset(frame, landmarks, el_right_eye_landmarks, color=(255,255,255), radius=2, connect=True)
            du.draw_landmarks_subset(frame, landmarks, ml_landmarks, color=(255,255,255), radius=2, connect=True)
            
            curr = time.time()    
            ear = eye_features.calc_ear(landmarks)
            eye_features.updated_eye_states(ear, curr)
            perclos_val = eye_features.perclos()
            blink_count = eye_features.blink_counter

            mar = mouth_features.calc_mar(landmarks)
            yawning_duration = mouth_features.check_yawn(mar, curr)
            ''''
            
            
            # Head pose estimation
            model_points = np.array([
                (0.0, 0.0, 0.0),              
                (0.0, -63.6, -12.5),         
                (43.3, 32.7, -26.0),         
                (-43.3, 32.7, -26.0),        
                (28.9, -28.9, -24.1),        
                (-28.9, -28.9, -24.1)        
            ], dtype='double')

            image_points = np.array([
                landmarks[1], landmarks[152], landmarks[263],
                landmarks[33], landmarks[287], landmarks[57]
            ], dtype="double")
            
            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype='double')
            dist_coeffs = np.zeros((4, 1))
    
            success, rotation_vector,_=c.solvePnP(model_points,image_points,camera_matrix,dist_coeffs,flags=c.SOLVEPNP_ITERATIVE)
            if success:
                rotation_mat,_ = c.Rodrigues(rotation_vector)
                pose_mat = c.hconcat((rotation_mat,np.zeros((3,1))))
                _, _, _, _, _, _, euler_angles = c.decomposeProjectionMatrix(pose_mat)
                pitch = float(euler_angles[0].item())

            if pitch <= head_down_threshold:
                if head_down_start_time is None:
                    head_down_start_time = time.time()
                elif time.time() - head_down_start_time >= min_down_duration:
                    head_state = "LOOKING_DOWN"
            else:
                head_down_start_time = None 
                head_state = "LOOKING GOOD"
            
            
            
            
            '''
            
            # Label decision
            label = 0
            if eye_features.perclos() >= perclos_threshold:
                label = 1
            if mouth_features.check_yawn(mar, curr): 
                label = 1
            '''
            if head_state == "LOOKING_DOWN":
                label = 1
            
            pitch_shifted = pitch + 180
            pitch_scaled = pitch_shifted / 360
            '''
            

            # Append data row
            data = {
                'EAR': round(ear, 3),
                'MAR': round(mar, 3),
                'label': label
            }
            rows.append(data)
            
        c.putText(frame, f"EAR: {ear:.3f}", (30, 30),
          c.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        c.putText(frame, f"MAR: {mar:.3f}", (30, 60),
                c.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        c.putText(frame, f"Label: {label}", (30, 90),
                c.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
        # Show frame for real-time
        c.imshow("Dataset Extraction", frame)
        if c.waitKey(1) & 0xFF == 27:  # ESC to exit
            
            break
            
    cap.release()
    c.destroyAllWindows()
    return rows

if __name__ == "__main__":
    rows = process_camera()
    df = pd.DataFrame(rows)
    df.to_csv("drowsiness_dataset.csv", index=False)
    print("Dataset saved as drowsiness_dataset.csv")
