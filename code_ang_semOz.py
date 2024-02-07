import utils.video_settings as utils_video
import utils.csv_settings as utils_csv

import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculo_angulo(a,b,c):
    a = np.array(a) #Primeiro ponto 
    b = np.array(b) #Ponto do meio
    c = np.array(c) #Ponto Final

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) #(y1-y2, x1-x2) 
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: #é o angulo máximo esticado
        angle = 360-angle

    
    return angle


def calculo_angulo(shoulder, elbow, wrist):
    """Calcula o ângulo entre o ombro, cotovelo e punho."""

    shoulder = np.array(shoulder)
    elbow = np.array(elbow)
    wrist = np.array(wrist)

    upper_arm = elbow - shoulder
    forearm = wrist - elbow

    dot_product = np.dot(upper_arm, forearm)
    upper_arm_length = np.linalg.norm(upper_arm)
    forearm_length = np.linalg.norm(forearm)

    cosine_angle = dot_product / (upper_arm_length * forearm_length)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def processa_frame(image):
    """Processa um frame de vídeo e retorna os pontos de referência do corpo."""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return results


def desenha_landmarks(image, landmarks, connections):
    """Desenha os pontos de referência do corpo na imagem."""

    mp_drawing.draw_landmarks(image, landmarks, connections,
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                             mp_drawing.DrawingSpec(color=(245, 66, 280), thickness=2, circle_radius=2))


def main():
    """Função principal do programa."""
    cv2.namedWindow('Imagem WebCam', cv2.WINDOW_NORMAL)
    FOLDER_IMAGE_PATH = 'frame_images'
    IMPORT_NAME_VIDEO = 'video_test.mp4'
    
    OUT_FILENAME_CSV = 'angles'
    OUT_FILENAME_VIDEO = 'mulher'

    csv_file = utils_csv.create_csv(OUT_FILENAME_CSV)

    frame_count = 0
    cap = utils_video.importar_video(IMPORT_NAME_VIDEO)
    
    utils_video.remove_files()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = processa_frame(frame)
        try:

            landmarks = results.pose_landmarks.landmark

            # Calcula o ângulo entre o ombro, cotovelo e punho.

            
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, 
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            
            '''
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, 
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, 
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
             '''

            angle = calculo_angulo(wrist,  elbow, shoulder)

            # Escreve o frame e o ângulo no arquivo CSV.
            #utils_csv.write_csv_ang(csv_file, frame_count, angle, OUT_FILENAME_CSV)
            utils_csv.write_csv_ang(csv_file, frame_count, angle, OUT_FILENAME_CSV)


            # Escreve os pontos x/y do ombro
      
            # Desenha os pontos de referência do corpo na imagem.

            cv2.putText(frame, str(angle), tuple(np.multiply(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass
    
        
        desenha_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        utils_video.frames_from_video(frame, frame_count, OUT_FILENAME_VIDEO, FOLDER_IMAGE_PATH)
    
        cv2.imshow('Imagem WebCam', frame)
        cv2.resizeWindow('Imagem WebCam', 720, 1366)
        if cv2.waitKey(10) & 0XFF == ord('q'):
            break

        frame_count += 1  # Incrementa frame

    cap.release()
    cv2.destroyAllWindows()
    print("Fim")

if __name__ == '__main__':
    main()