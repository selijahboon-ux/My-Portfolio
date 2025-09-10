''' ---------------------------------------------------------------------------------------------------------------------------------------------------------
Author: Elijah Boon
Created: 2025-08-12
Description: This is the rule-based features of the AI-driven anti drowsiness detection system.
--------------------------------------------------------------------------------------------------------------------------------------------------------------'''
import cv2 as c
import mediapipe as mp
import time
import numpy as np
from eye import EyeFeatures
from mouth import MouthFeatures
from utils import EyeLandmarks as el
from utils import MouthLandmarks as ml
from utils import DrawingUtils as du

def main():
    cap = c.VideoCapture(0)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )
    
    head_down_start_time = None
    head_down_threshold = -20     
    min_down_duration = 5
    head_state = "UNKNOWN" 
    
    ml_landmarks = ml.mouthLandmarks()
    el_left_eye_landmarks = el.leftEyeLandmarks()
    el_right_eye_landmarks = el.rightEyeLandmarks()

    
    ret, frame = cap.read()
    if not ret:
        return

    h, w, _ = frame.shape
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    eye_features = EyeFeatures(
        close_threshold=0.18,
        open_threshold=0.20,
        eye_duration=60
    )
    
    mouth_features = MouthFeatures(
        mouth_threshold=0.60,
        yawn_min=4,
        yawn_max=7
    )
    
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
                
                du.draw_landmarks_subset(frame, landmarks, el_left_eye_landmarks, color=(255, 255, 255), radius=2, connect=True)
                du.draw_landmarks_subset(frame, landmarks, el_right_eye_landmarks, color=(255, 255, 255), radius=2, connect=True)
                du.draw_landmarks_subset(frame, landmarks, ml_landmarks, color=(255, 255, 255), radius=2, connect=True)

                curr = time.time()    
                #EyeFeatures 
                ear = eye_features.calc_ear(landmarks)
                eye_features.updated_eye_states(ear, curr)
                perclos_val = eye_features.perclos()
                blink_count = eye_features.blink_counter(ear, curr)
                
                #MouthFeatures
                mar = mouth_features.calc_mar(landmarks)
                yawning_now = mouth_features.check_yawn(mar, curr)
                
                model_points = np.array([
                    (0.0, 0.0, 0.0),             
                    (0.0, -63.6, -12.5),         
                    (43.3, 32.7, -26.0),         
                    (-43.3, 32.7, -26.0),        
                    (28.9, -28.9, -24.1),        
                    (-28.9, -28.9, -24.1)        
                ], dtype='double')

                image_points = np.array([
                landmarks[1],   
                landmarks[152], 
                landmarks[263], 
                landmarks[33],  
                landmarks[287], 
                landmarks[57]   
            ], dtype="double")
                
                focal_length = w
                center = (w/2,h/2)
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

                 # Timer logic for non-instant detection
                if pitch <= head_down_threshold:
                    if head_down_start_time is None:
                        head_down_start_time = time.time()
                    elif time.time() - head_down_start_time >= min_down_duration:
                        head_state = "LOOKING_DOWN"
                else:
                    head_down_start_time = None
                    head_state = "LOOKING GOOD"
                c.putText(frame, f"HEAD STATUS: {head_state}", (30, 180), c.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                c.putText(frame, f"EAR: {ear:.2f}", (30, 30), c.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                c.putText(frame, f"PERCLOS: {perclos_val:.3f}", (30, 60), c.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                c.putText(frame, f"Blinks: {blink_count}", (30, 90), c.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                c.putText(frame, f"MAR: {mar:.2f}", (30, 120), c.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                c.putText(frame, f"PITCH: {pitch:.2f}", (30, 200), c.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

               
                mouth_state = "Yawning" if yawning_now else "Not yawning"
                c.putText(frame, f"MOUTH STATUS: {mouth_state}", (30, 150), c.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                
        c.imshow('Eye & Mouth Features Real-time', frame)

        if c.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    c.destroyAllWindows()

if __name__ == "__main__":
    main()
