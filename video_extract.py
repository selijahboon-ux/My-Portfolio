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

def process_video(video_path):
    rows = []

    cap = c.VideoCapture(video_path)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )

    ml_landmarks = ml.mouthLandmarks()
    el_left_eye_landmarks = el.leftEyeLandmarks()
    el_right_eye_landmarks = el.rightEyeLandmarks()

    ret, frame = cap.read()
    if not ret:
        return rows  # return empty list if video can't be read

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
        yawn_min=2,
        yawn_max=7
    )
    
    head_down_start_time = None
    head_down_threshold = -40     
    min_down_duration = 10
    head_state = "UNKNOWN" 
    perclos_threshold = 0.30

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
            ear = eye_features.calc_ear(landmarks)
            eye_features.updated_eye_states(ear, curr)
            perclos_val = eye_features.perclos()
            
            mar = mouth_features.calc_mar(landmarks)
            yawning_duration = mouth_features.check_yawn(mar, curr)
            
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

            if pitch <= head_down_threshold:
                if head_down_start_time is None:
                    head_down_start_time = time.time()
                elif time.time() - head_down_start_time >= min_down_duration:
                    head_state = "LOOKING_DOWN"
            else:
                head_down_start_time = None 
                head_state = "LOOKING GOOD"
            
            label = 0
            if eye_features.perclos() >= perclos_threshold:
                label = 1
            if mouth_features.check_yawn(mar, curr): 
                label = 1
            if head_state == "LOOKING_DOWN":
                label = 1

            pitch_shifted = pitch + 180
            pitch_scaled = pitch_shifted / 360

            data = {
                'Video': os.path.basename(video_path),
                'PERCLOS': round(perclos_val, 3),
                'YAWNS': 1 if mouth_features.check_yawn(mar, curr) else 0,
                'Pitch': round(pitch_scaled, 3),
                'label': label
            }
            rows.append(data)
            
    cap.release()
    return rows

if __name__ == "__main__":
    video_folder = "videos"
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4','.avi','.mov'))]

    all_rows = []
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        print(f"Processing {video_file} ...")
        rows = process_video(video_path)
        all_rows.extend(rows)

    # Save all features into a single CSV
    df = pd.DataFrame(all_rows)
    df.to_csv("all_videos_drowsy.csv", index=False)
    print("All videos processed. Features saved to all_videos_drowsy.csv")
