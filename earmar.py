import os
import cv2 as c
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from eye import EyeFeatures
from mouth import MouthFeatures
from utils import EyeLandmarks as el
from utils import MouthLandmarks as ml
from utils import DrawingUtils as du

# --------------------------
# Part 1: Video Feature Extraction (EAR + MAR only)
# --------------------------

def process_video_ear_mar(video_path, ear_threshold=0.25, mar_threshold=0.6, mar_min_duration=4, mar_max_duration=7):
    rows = []
    cap = c.VideoCapture(video_path)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    
    ml_landmarks = ml.mouthLandmarks()
    el_left_eye_landmarks = el.leftEyeLandmarks()
    el_right_eye_landmarks = el.rightEyeLandmarks()

    mouth_features = MouthFeatures(mouth_threshold=mar_threshold, yawn_min=mar_min_duration, yawn_max=mar_max_duration)
    eye_features = EyeFeatures(close_threshold=ear_threshold, open_threshold=ear_threshold+0.05, eye_duration=60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = c.flip(frame, 1)
        frame_rgb = c.cvtColor(frame, c.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) 
                         for lm in results.multi_face_landmarks[0].landmark]
            
            # Calculate EAR and MAR
            ear = eye_features.calc_ear(landmarks)
            eye_features.updated_eye_states(ear, time.time())
            perclos_val = eye_features.perclos()  # can ignore for training, optional

            mar = mouth_features.calc_mar(landmarks)
            sustained_yawn = mouth_features.check_yawn(mar, time.time())  # returns True if MAR sustained

            # Binary label logic
            label = 0
            if ear < ear_threshold:        # Eyes closed
                label = 1
            elif sustained_yawn:           # MAR sustained for 4–7 frames
                label = 1

            rows.append({
                'Video': os.path.basename(video_path),
                'EAR': round(ear, 3),
                'MAR': round(mar, 3),
                'label': label
            })
    
    cap.release()
    return rows

# --------------------------
# Part 2: PyTorch Dataset
# --------------------------

class EarMarDataset(Dataset):
    def __init__(self, csv_file, sequence_length=30):
        self.df = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.videos = self.df['Video'].unique()
        self.data = []

        for video in self.videos:
            video_df = self.df[self.df['Video'] == video]
            features = video_df[['EAR', 'MAR']].values
            labels = video_df['label'].values

            for i in range(len(features) - sequence_length):
                seq = features[i:i+sequence_length]
                target = labels[i+sequence_length-1]
                self.data.append((seq, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# --------------------------
# Part 3: LSTM Model
# --------------------------

class DrowsinessLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(DrowsinessLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

# --------------------------
# Part 4: Main Script
# --------------------------

if __name__ == "__main__":
    video_folder = "videos"
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4','.avi','.mov'))]

    all_rows = []
    for video_file in video_files:
        print(f"Processing {video_file} ...")
        rows = process_video_ear_mar(os.path.join(video_folder, video_file))
        all_rows.extend(rows)

    csv_path = "all_videos_ear_mar.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Features saved to {csv_path}")

    # ------------------ Train LSTM ------------------
    dataset = EarMarDataset(csv_path, sequence_length=30)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DrowsinessLSTM(input_size=2).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for seq, target in dataloader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    print("LSTM training complete!")
