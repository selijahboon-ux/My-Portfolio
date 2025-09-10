''' ---------------------------------------------------------------------------------------------------------------------------------------------------------
Author: Elijah Boon
Created: 2025-08-10
Description:This class contains all the Eye related features that will be used for the AI-driven drowsiness detection system. 
            Eye-Aspect-Ratio (EAR) will be used to compute the openness and closeness of the eye using Euclidean Algebra to compute the
            distance between the horizontal and vertical points of the eye, this will be done using Mediapipe's face mesh to extract the landmark
            points of the user. To avoid instant detection and reduce detection noise(false positives), Percetage of Eye Closure (PERCLOS) was incorporated,
            a metric used to assess drowsiness by measuring the proportion of time the eyelids are closed over a specific time period. Finally,
            blinking a lot maybe a sign of fatigue or even irretation, so adding a blink counter would be another parameter for detection because the 
            average blinks of a human per minute is only 14 to 17 times.
--------------------------------------------------------------------------------------------------------------------------------------------------------------''' 
from collections import deque
import time
from utils import EuclideanDistance as eu
from utils import EyeLandmarks as el
        
class EyeFeatures:
    '''
    Left Eye Landmarks:
    33 = the left eye's outer corner(Horizontal).
    133 = the left eye's inner corner(Horizontal).
    160 and 144 = top and bottom eyelid (First Vertical pair).
    158 and 153 = top and bottom eyelid (Second Vertical pair).
    
    Right Eye Landmarks:
    362 = the right eye's outer corner(Horizontal).
    263 = the right eye's inner corner(Horizontal).
    385 and 380 = top and bottom eyelid (First Vertical pair).
    387 and 373 = top and bottom eyelid (Second Vertical pair).
    '''
    def __init__(self, close_threshold, open_threshold, eye_duration, left_eye_landmarks=0,right_eye_landmark=0):
        self.close_threshold = close_threshold#Threshold for the closeness of the eyes.
        self.open_threshold = open_threshold#Threshold for the openess of the eye.
        self.duration = eye_duration#The time duration of the frames collected in the buffers.
        self.left = left_eye_landmarks or el.leftEyeLandmarks()#Left eye landmarks.
        self.right = right_eye_landmark or el.rightEyeLandmarks()#Right eye landmarks.
        self.eye_closed_perclos = False#Boolean for checking if the eyes are open or close for the PERCLOS.     
        self.eye_closed_blink = False#Boolean for checking if the eyes are open or close for the Blink Counter.
        self.last_ear = 0#To store the most recent EAR calculated.
        self.perclos_value = 0#To store the most recent PERCLOS values.
        self.blink_count = 0#To keep track of the total number of blinks.
        self.blink_timestamps = deque()#Buffer for Blink Counter.
        self.eye_states = deque()#Buffer for PERCLOS.  
        
    def reset(self):#reset man.
        self.eye_states.clear()
        self.eye_closed_perclos = False
        self.eye_closed_blink = False
        self.last_ear = 0
        self.perclos_value = 0
        self.blink_count = 0
        self.blink_timestamps.clear()

        
    def calc_ear(self, landmarks):#Compute the Eye-Aspect_ratio.
        left = [landmarks[i] for i in self.left]
        right = [landmarks[i] for i in self.right]
        def ear(eye):#calculate the distance between points.
            A = eu.euclidean(eye[1], eye[5])
            B = eu.euclidean(eye[2], eye[4])
            C = eu.euclidean(eye[0], eye[3])
            return (A + B) / (2.0 * C)
        self.last_ear = (ear(left) + ear(right)) / 2.0
        return self.last_ear
    
    def updated_eye_states(self, ear, curr=None):#Stores the eye states in the buffer.
        if curr is None:
            curr = time.time()
        if not self.eye_closed_perclos and ear < self.close_threshold:#Check if eyes are closed, handles transition from open to close.
            self.eye_closed_perclos = True#Eyes are close
        elif self.eye_closed_perclos and ear > self.open_threshold:#Check if eyes are open, handles transition from close to open.
            self.eye_closed_perclos = False#Eyes are Open
        self.eye_states.append((curr, int(self.eye_closed_perclos)))#Append to buffer
        while self.eye_states and curr - self.eye_states[0][0] > self.duration:
            self.eye_states.popleft()

    def perclos(self):
        if len(self.eye_states) < 2:#This nigga checks if there are fewer than 2 entries, if there are, those events are disregraded(not enough data).
            return 0
        curr = time.time()
        closed_time = 0
        total_time = min(self.duration, curr - self.eye_states[0][0])
        for i in range(1, len(self.eye_states)):#Compare each element from the previous one 
            prev_t, prev_state = self.eye_states[i - 1]
            curr_t, _ = self.eye_states[i]
            if prev_state == 1:
                closed_time += curr_t - prev_t
        return closed_time / total_time if total_time > 0 else 0

    def blink_counter(self, ear, curr=None):
        if curr is None:
            curr = time.time()
        while self.blink_timestamps and curr - self.blink_timestamps[0] > self.duration:
            self.blink_timestamps.popleft()

        if not self.eye_closed_blink and ear < self.close_threshold:#Checks if the eye is closed(handles open to close state)
            self.eye_closed_blink = True
            self.blink_timestamps.append(curr)
        elif self.eye_closed_blink and ear > self.open_threshold:#Checks if the eye is open(handles close to open state)
            self.eye_closed_blink = False

        return len(self.blink_timestamps)
        

   
   
