import utils.video_settings as utils_video
import utils.csv_settings as utils_csv
import cv2
import csv
import os
import mediapipe as mp
import numpy as np
import math
import seaborn as sns
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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


def calculate_midpoint(x1, y1, z1, x2, y2, z2):
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    midpoint_z = (z1 + z2) / 2
    return midpoint_x, midpoint_y, midpoint_z


# Definir cores únicas para cada landmark
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)]

def processa_frame(image, pose):
    """Processa um frame de vídeo e retorna os pontos de referência do corpo."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results
def desenha_landmarks(image, landmarks):
    """Desenha os pontos de referência do corpo na imagem."""
    if landmarks:
        for idx, landmark in enumerate(landmarks.landmark):
            landmark_px = tuple(map(int, np.array([landmark.x, landmark.y]) * np.array([image.shape[1], image.shape[0]])))
            color = colors[idx % len(colors)]  # Seleciona uma cor diferente para cada ponto de landmark
            cv2.circle(image, landmark_px, 5, color, -1)
            cv2.putText(image, str(idx), (landmark_px[0] + 10, landmark_px[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)





def main(cap, video_name):
    """Função principal do programa.
    cv2.namedWindow('Imagem WebCam', cv2.WINDOW_NORMAL)
    FOLDER_IMAGE_PATH = 'frame_images'
    IMPORT_NAME_VIDEO = 'video.mp4'
    
    OUT_FILENAME_CSV = 'angle_MMSS_video60'
    OUT_FILENAME_CSV2 = 'angle_Coluna_video60'
    OUT_FILENAME_VIDEO = 'mulher'
  
    csv_file = utils_csv.create_csv(OUT_FILENAME_CSV)
    csv_file2 = utils_csv.create_csv(OUT_FILENAME_CSV2)

    cap = utils_video.importar_video(IMPORT_NAME_VIDEO)
    """
    frame_count = 0
    #utils_video.processar_videos('sample_videos')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Atribuir essas informações às variáveis image_width e image_height
    image_width = width
    image_height = height

    vetor_wrist1 = [[],[],[]]
    vetor_wrist2 = [[],[],[]]
    vetor_shoulder1 = [[],[],[]]
    vetor_shoulder2 = [[],[],[]]
    vetor_elbow1 = [[],[],[]]
    vetor_elbow2 = [[],[],[]]
    vetor_hip1 = [[],[],[]]
    vetor_hip2 = [[],[],[]]
    vetor_knee1 = [[],[],[]]
    vetor_knee2 = [[],[],[]]
    vetor_olho1 = [[],[],[]]
    vetor_olho2 = [[],[],[]]
    #utils_video.remove_files()

    
    # Cria uma instância do objeto Pose fora do loop
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = processa_frame(frame,pose)
        try:
            if frame_count % 1 == 0:
                landmarks = results.pose_landmarks.landmark
                
                #funçao que calcula o comprimento de um vetor
                def modulo_vetor(x, y, z):
                    
                    v = (x**2 + y**2 + z**2)**0.5
                    
                    return v
                
                '''funçao que retona um vetor normalizado, ou seja, um vetor na mesma direção mas com norma igual a 1'''
                def normaliza_vetor(x, y, z):
                    
                    aux = modulo_vetor(x, y, z)
                    
                    v = [x/aux, y/aux, z/aux]
                    
                    return v

                '''funçao que recebe como parametro as coordenadas de dois pontos, que sao os extremos das duas retas, ou seja, esses dois pontos definem essa reta. ela retorna a direção da reta no espaço'''
                def calcula_inclinacao(x1, y1, z1, x2, y2, z2):
                    
                    vetor_diretor = [(x2 - x1), (y2 - y1), (z2 - z1)]
                    
                    return normaliza_vetor(*vetor_diretor)
                    
                '''função recebe as coordenadas de duas retas e de dois pontos'''
                def angulo_de_flexao(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
                    
                    inclinacao_reta1 = calcula_inclinacao(x1, y1, z1, x2, y2, z2)
                    inclinacao_reta2 = calcula_inclinacao(x3, y3, z3, x4, y4, z4)
                    
                    produto_escalar = 0.0
                    
                    for v1, v2 in zip(inclinacao_reta1, inclinacao_reta2):
                        produto_escalar += v1 * v2
                    

                    angulo = math.acos(produto_escalar)

                    return math.degrees(angulo)   
            #_________________________________________________________________________
            # Coleta de X, Y E Z de todos os pontos necessários
            #_________________________________________________________________________

                wrist1 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                
                a,b,c = wrist1
                vetor_wrist1[0].append(a)
                vetor_wrist1[1].append(b)
                vetor_wrist1[2].append(c)
                

                shoulder1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                #print(shoulder1)
                #print(landmarks[12])
                a,b,c = shoulder1
                vetor_shoulder1[0].append(a)
                vetor_shoulder1[1].append(b)
                vetor_shoulder1[2].append(c)
            

                elbow1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image_height, 
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                a,b,c = elbow1
                vetor_elbow1[0].append(a)
                vetor_elbow1[1].append(b)
                vetor_elbow1[2].append(c)


                wrist2 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                a,b,c = wrist2
                vetor_wrist2[0].append(a)
                vetor_wrist2[1].append(b)
                vetor_wrist2[2].append(c)          


                shoulder2 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                a,b,c = shoulder2
                vetor_shoulder2[0].append(a)
                vetor_shoulder2[1].append(b)
                vetor_shoulder2[2].append(c)


            
                elbow2 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height, 
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                a,b,c = elbow2
                vetor_elbow2[0].append(a)
                vetor_elbow2[1].append(b)
                vetor_elbow2[2].append(c)
            

                hip1 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                a,b,c = hip1
                vetor_hip1[0].append(a)
                vetor_hip1[1].append(b)
                vetor_hip1[2].append(c)

              
                hip2 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                a,b,c = hip2
                vetor_hip2[0].append(a)
                vetor_hip2[1].append(b)
                vetor_hip2[2].append(c)  

             
                olho1 = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].z]
                a,b,c = olho1
                vetor_olho1[0].append(a)
                vetor_olho1[1].append(b)
                vetor_olho1[2].append(c)  


                olho2 = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].z]
                a,b,c = olho2
                vetor_olho2[0].append(a)
                vetor_olho2[1].append(b)
                vetor_olho2[2].append(c)  

            
                knee1 = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                a,b,c = knee1
                vetor_knee1[0].append(a)
                vetor_knee1[1].append(b)
                vetor_knee1[2].append(c)  

                knee2 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                a,b,c = knee2
                vetor_knee2[0].append(a)
                vetor_knee2[1].append(b)
                vetor_knee2[2].append(c)  





            # Escreve o frame e o ângulo no arquivo CSV.
                #utils_csv.write_csv_ang(csv_file2, frame_count, angles2, OUT_FILENAME_CSV2)


            # Escreve o frame e o ângulo no arquivo CSV.
                #utils_csv.write_csv_ang(csv_file, frame_count, angles, OUT_FILENAME_CSV)



            # Escreve os pontos x/y do ombro
      
            # Desenha os pontos de referência do corpo na imagem.

                #cv2.putText(frame, str(angles), tuple(np.multiply(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, [640, 480]).astype(int)),
                 #   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass


    
        #desenha_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        desenha_landmarks(frame, results.pose_landmarks)
        #utils_video.frames_from_video(frame, frame_count, OUT_FILENAME_VIDEO, FOLDER_IMAGE_PATH)
    
        cv2.imshow('Imagem WebCam', frame)
        cv2.resizeWindow('Imagem WebCam', 720, 1366)
      
        if cv2.waitKey(10) & 0XFF == ord('q'):
            break

        frame_count += 1  # Incrementa frame







    #_________________________________________________________________________
    # Filtro Savgol_filter em todos os pontos, nos eixos X, Y E Z
    #_________________________________________________________________________

    filter_wrist1_x = savgol_filter(vetor_wrist1[0], 7, 2)
    filter_wrist1_y = savgol_filter(vetor_wrist1[1], 7, 2)
    filter_wrist1_z = savgol_filter(vetor_wrist1[2], 7, 2)
    

    #print(filter_wrist1_x, filter_wrist1_y, filter_wrist1_z)

    filter_wrist2_x = savgol_filter(vetor_wrist2[0], 7, 2)
    filter_wrist2_y = savgol_filter(vetor_wrist2[1], 7, 2)
    filter_wrist2_z = savgol_filter(vetor_wrist2[2], 7, 2)

    filter_shoulder1_x = savgol_filter(vetor_shoulder1[0], 7, 2)
    filter_shoulder1_y = savgol_filter(vetor_shoulder1[1], 7, 2)
    filter_shoulder1_z = savgol_filter(vetor_shoulder1[2], 7, 2)

    filter_shoulder2_x = savgol_filter(vetor_shoulder2[0], 7, 2)
    filter_shoulder2_y = savgol_filter(vetor_shoulder2[1], 7, 2)
    filter_shoulder2_z = savgol_filter(vetor_shoulder2[2], 7, 2)

    filter_elbow1_x = savgol_filter(vetor_elbow1[0], 7, 2)
    filter_elbow1_y = savgol_filter(vetor_elbow1[1], 7, 2)
    filter_elbow1_z = savgol_filter(vetor_elbow1[2], 7, 2)

    filter_elbow2_x = savgol_filter(vetor_elbow2[0], 7, 2)
    filter_elbow2_y = savgol_filter(vetor_elbow2[1], 7, 2)
    filter_elbow2_z = savgol_filter(vetor_elbow2[2], 7, 2)


    filter_hip1_x = savgol_filter(vetor_hip1[0], 7, 2)
    filter_hip1_y = savgol_filter(vetor_hip1[1], 7, 2)
    filter_hip1_z = savgol_filter(vetor_hip1[2], 7, 2)


    filter_hip2_x = savgol_filter(vetor_hip2[0], 7, 2)
    filter_hip2_y = savgol_filter(vetor_hip2[1], 7, 2)
    filter_hip2_z = savgol_filter(vetor_hip2[2], 7, 2)

    filter_knee1_x = savgol_filter(vetor_knee1[0], 7, 2)
    filter_knee1_y = savgol_filter(vetor_knee1[1], 7, 2)
    filter_knee1_z = savgol_filter(vetor_knee1[2], 7, 2)


    filter_knee2_x = savgol_filter(vetor_knee2[0], 7, 2)
    filter_knee2_y = savgol_filter(vetor_knee2[1], 7, 2)
    filter_knee2_z = savgol_filter(vetor_knee2[2], 7, 2)


    filter_olho1_x = savgol_filter(vetor_olho1[0], 7, 2)
    filter_olho1_y = savgol_filter(vetor_olho1[1], 7, 2)
    filter_olho1_z = savgol_filter(vetor_olho1[2], 7, 2)

    filter_olho2_x = savgol_filter(vetor_olho2[0], 7, 2)
    filter_olho2_y = savgol_filter(vetor_olho2[1], 7, 2)
    filter_olho2_z = savgol_filter(vetor_olho2[2], 7, 2)

    
   
    
    #input_file = 'D:/Faculdade/PET/Fisioterapia/Fisioterapia_3D_Videos/teste/output.tsv'
    # Crie uma lista com todos os pontos filtrados
    filtered_points = [
        filter_wrist1_x, filter_wrist1_y, filter_wrist1_z,
        filter_wrist2_x, filter_wrist2_y, filter_wrist2_z,
        filter_shoulder1_x, filter_shoulder1_y, filter_shoulder1_z,
        filter_shoulder2_x, filter_shoulder2_y, filter_shoulder2_z,
        filter_elbow1_x, filter_elbow1_y, filter_elbow1_z,
        filter_elbow2_x, filter_elbow2_y, filter_elbow2_z,
        filter_hip1_x, filter_hip1_y, filter_hip1_z,
        filter_hip2_x, filter_hip2_y, filter_hip2_z,
        filter_knee1_x, filter_knee1_y, filter_knee1_z,
        filter_knee2_x, filter_knee2_y, filter_knee2_z,
        filter_olho1_x, filter_olho1_y, filter_olho1_z,
        filter_olho2_x, filter_olho2_y, filter_olho2_z
    ]

    # ROW: 
    # [0:3] wrist1, [3:6] wrist2, [6:9] shoulder1, [9:12] shoulder2, [12:15] elbow1, [15:18] elbow2
    # [18:21] hip1, [21:24] hip2, [24:27] knee1, [27:30] knee2, [30:33] olho1, [33:36] olho2

    output_file_points = f'D:/Faculdade/PET/Fisioterapia/Fisioterapia_3D_Videos/teste/{video_name}.tsv'


    # Transpondo a lista de pontos filtrados
    filtered_points_transposed = np.array(filtered_points).T.tolist()

    # Escrevendo os pontos transpostos no arquivo TSV
    with open(output_file_points, 'w', newline='') as tsv_out:
        writer = csv.writer(tsv_out, delimiter='\t')
        writer.writerows(filtered_points_transposed)

 
    
    
    
    
    
    #sns.lineplot(vetortest[0], marker = '_', color = 'green', alpha = 0.5)

    # plota o grafico do seno com o ruido adicionado
    #sns.lineplot(yhat, marker = '+', color = 'black', alpha = 0.5)
    #plt.show()
    cap.release()
    cv2.destroyAllWindows()

    pass
if __name__ == '__main__':
    utils_video.processar_videos('D:/Faculdade/PET/Fisioterapia/Fisioterapia_3D_Videos/sample_videos/')