import cv2
import numpy as np
import mediapipe as mp
## Inicitalize the MediaPipe to keypoint detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


### TODO: Add more parts of the body

## Detect the specific key_point in the image
def detect(image_path, keypoint):
    ##Load the image
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        ## Change the color of the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ## Detect the keypoints in the image
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            ## Get the position of the keypoints
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width), 
                             int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)]
    
            left_elbow = [int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width), 
                          int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height)]
            
            left_hip = [int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image_width), 
                        int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image_height)]

            points = {"LEFT_SHOULDER" : left_shoulder,
                      "LEFT_ELBOW" : left_elbow,
                      "LEFT_HIP" : left_hip}
            return points[keypoint]