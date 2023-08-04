### Media Pipe
## Landmark model
## 33 Points of pose
## 468 points reference to face
## 21+21 points hands



##Pose_World_landmarks

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

class AnglesTracking:
        def __init__(self, type_video = 0, angles = ["angle_left_elbow"], image_size=[640, 480]):
                self.type_video = type_video
                self.active_angles = angles
                self.image_size = image_size
                self.camera_available = self.check_available_cameras()

                ## Order: "angle_left_elbow", "angle_right_elbow", "angle_left_shoulder","angle_right_shoulder", "angle_left_knee", "angle_right_knee"
                self.current_angles = { "angle_left_elbow_r1" : None,
                                        "shoulder_left_rotation_ra" : None,
                                        "shoulder_left_rotation_rf" : None,
                                        "shoulder_left_rotation_rr" : None,

                                        "angle_right_elbow_r1" : None,
                                        "shoulder_right_rotation_ra" : None,
                                        "shoulder_right_rotation_rf" : None,
                                        "shoulder_right_rotation_rr" : None
                                       }

                if self.camera_available == False and type_video == 0:
                    print("Error: No camera available")
                

        def calc_angle(self,a, b, c):
                a = np.array(a) # First
                b = np.array(b) # Mid
                c = np.array(c) # End
                print(c)
                ba = a - b
                bc = c - b

                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)

                return round(np.degrees(angle))

        def people_tracking(self, angles_considered, video = 0, image_size=[640, 480]):
                cap = cv2.VideoCapture(0)
                with mp_holistic.Holistic(
                    static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5) as holistic:

                    while cap.isOpened():
                        
                        success, image = cap.read()


                        #percent by which the image is resized
                        scale_percent = 50

                        #calculate the 50 percent of original dimensions
                        width = int(image.shape[1] * scale_percent / 100)
                        height = int(image.shape[0] * scale_percent / 100)

                        # dsize
                        dsize = (width, height)

                        # resize image
                        image = cv2.resize(image, dsize)
                        
                        if not success:
                          print("Ignoring empty camera frame.")
                          continue

                        try: 
                            # To improve performance, optionally mark the image as not writeable to
                            # pass by reference. Set the vector as imutable
                            image.flags.writeable = True
                            
                            results = holistic.process(image)

                            #image = cv2.flip(image, 1)

                            if results:
##                                ## Draw Face
##                                mp_drawing.draw_landmarks(image=image,
##                                                           landmark_list=results.face_landmarks,
##                                                           connections=mp_holistic.FACEMESH_TESSELATION,
##                                                           landmark_drawing_spec=None,
##                                                           connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
##                                                           )
                                ## Draw Pose
                                mp_drawing.draw_landmarks(image=image,
                                                          landmark_list=results.pose_landmarks,
                                                          connections=mp_holistic.POSE_CONNECTIONS,
                                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                                          )

##                                ## Draw left hand
##                                mp_drawing.draw_landmarks(image=image,
##                                                          landmark_list=results.left_hand_landmarks,
##                                                          connections=mp_holistic.HAND_CONNECTIONS,
##                                                          landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
##                                                          )
##                                ## Draw right hand
##                                mp_drawing.draw_landmarks(image=image,
##                                                          landmark_list=results.right_hand_landmarks,
##                                                          connections=mp_holistic.HAND_CONNECTIONS,
##                                                          landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
##                                                          )
                                ## Draw world pose
                                #mp_drawing.plot_landmarks(
                                #    results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
                                landmarks = results.pose_world_landmarks.landmark
                                
                                ## Get coordinates
                                ## 11
                                left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].z]
                                ## 12
                                right_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].z]
                                ## 13
                                left_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].z]
                                ## 14
                                right_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].z]
                                ## 15
                                left_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].z]
                                ## 16            
                                right_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].z]
                                ## 23
                                left_hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].z]
                                ## 24
                                right_hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].z]
                                ## 25
                                left_knee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].z]
                                ## 26
                                right_knee = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].z]
                                ## 27
                                left_ankle = [landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].z]
                                ## 28
                                right_ankle = [landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].z]

                                ## Adjust the hip to be a straight line
                                left_hip_straight = [left_shoulder[0], left_hip[1], left_hip[2]]
                                right_hip_straight = [right_shoulder[0], right_hip[1], right_hip[2]]

                                ## Adjust the wrip to be a straight line
                                left_elbow_straight = [left_elbow[0], left_elbow[1], left_elbow[2]-1]
                                right_elbow_straight = [right_elbow[0]+1, right_elbow[1], right_elbow[2]]
                                
                                angles_points = {
                                        "angle_left_elbow_r1" : (left_shoulder, left_elbow, left_wrist),
                                        "angle_right_elbow_r1" : (right_shoulder, right_elbow, right_wrist),

                                        "shoulder_left_rotation_ra" : (left_elbow, left_shoulder, left_hip_straight),
                                        "shoulder_left_rotation_rf" : (left_elbow, left_shoulder, left_hip_straight),
                                        "shoulder_left_rotation_rr" : (left_elbow_straight, left_elbow, left_wrist),

                                        "shoulder_right_rotation_ra" : (right_elbow, right_shoulder, right_hip_straight),
                                        "shoulder_right_rotation_rf" : (right_elbow, right_shoulder, right_hip_straight),
                                        "shoulder_right_rotation_rr" : (right_elbow_straight, right_elbow, right_wrist),
                                            }

                                places_to_show = {
                                        "angle_left_elbow_r1" : (left_elbow[0], left_elbow[1]),
                                        "angle_right_elbow_r1" : (right_elbow[0], right_elbow[1]),

                                        "shoulder_left_rotation_ra" : (left_shoulder[0], left_shoulder[1]),
                                        "shoulder_left_rotation_rf" : (left_shoulder[0], left_shoulder[1]),
                                        "shoulder_left_rotation_rr" : (left_elbow[0], left_elbow[1]),

                                        "shoulder_right_rotation_ra" : (right_shoulder[0], right_shoulder[1]),
                                        "shoulder_right_rotation_rf" : (right_shoulder[0], right_shoulder[1]),
                                        "shoulder_right_rotation_rr" : (right_elbow[0], right_elbow[1]),
                                            }


                                
  
                                #cv2.rectangle(image,(0,0),(200,200),(0,0,0),-1)
                                text_posx = 20
                                text_step = 20

                                for count, i in enumerate(angles_considered):
                                        angle = self.calc_angle(*angles_points[i])
                                        
                                        self.current_angles[i] = angle

                                        cv2.putText(image, f"{i} : " + str("{:0.2f}".format(angle)),
                                                    (10, text_posx+text_step*count),
                                                    cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 1)
                                        #print(f"{i}: {angle}")

                        except Exception as e:
                            print(e)
                            continue
                                         
                            
                        
                        # Flip the image horizontally for a selfie-view display.
                        cv2.imshow('MediaPipe Pose', image)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                          break
                cap.release()

        def check_available_cameras(self):
                available_cameras = []
                index = 0
                while True:
                    cap = cv2.VideoCapture(index)
                    if not cap.read()[0]:
                        break
                    else:
                        available_cameras.append(index)
                    cap.release()
                    index += 1
                if len(available_cameras) > 0:
                    return True
                else:
                    return True    


                                        
execute = AnglesTracking()
execute.people_tracking([
#"shoulder_left_rotation_ra",
#"shoulder_right_rotation_ra"
#"angle_left_elbow_r1"
#"angle_right_elbow_r1",
#"shoulder_left_rotation_rf"#,
#"shoulder_right_rotation_rf"
"shoulder_left_rotation_rr"
#"shoulder_right_rotation_rr"#
                         ],
                        0, [640, 480])
#"angle_left_elbow_r1", "angle_right_elbow_r1",
#"shoulder_left_rotation_rf", "shoulder_right_rotation_rf"
#"shoulder_left_rotation_rr", "shoulder_right_rotation_rr"
#"shoulder_left_rotation_ra", "shoulder_right_rotation_ra"
