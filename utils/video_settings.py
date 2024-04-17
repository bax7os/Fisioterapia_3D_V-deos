import glob
import cv2
import os
from code_ang_comOz import main

def processar_videos(diretorio):
    """
    Processa todos os vídeos em um diretório específico.

    Args:
        diretorio: O caminho do diretório que contém os vídeos.

    Returns:
        None
    """
    # Obtém uma lista de todos os arquivos de vídeo no diretório.
    videos = glob.glob(f'{diretorio}/*.mp4')

    # Itera sobre cada arquivo de vídeo.
    for video in videos:
        # Importa o vídeo.
        cap = importar_video(video)

        # Obtém o nome do arquivo de vídeo sem a extensão.
        video_name = os.path.basename(video).split('.')[0]

        # Processa o vídeo.
        main(cap, video_name)

def importar_video(filename):
    """
    Importa um vídeo de algum diretório.

    Args:
        filename: O caminho completo do arquivo de vídeo.

    Returns:
        Um objeto VideoCapture da biblioteca OpenCV.
    """
    # Abre o arquivo de vídeo no modo de leitura.
    video_clip = cv2.VideoCapture(filename)

    # Verifica se o arquivo foi aberto com sucesso.
    if not video_clip.isOpened():
        raise FileNotFoundError(f"O arquivo '{filename}' não foi encontrado.")

    # Retorna o objeto VideoCapture.
    return video_clip



# # Exemplo de uso.
# caminho_do_arquivo = './sample_videos/video_test.mp4'
# video_clip = importar_video(caminho_do_arquivo)
# gerar_fotos_de_video(video_clip, 0.01, './frame_images')