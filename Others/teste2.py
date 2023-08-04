import cv2

# Função para verificar as câmeras ativas
def verificar_cameras():
    cameras_ativas = []

    # Testar índices de câmera de 0 a 9
    for i in range(10):
        cap = cv2.VideoCapture(i)

        # Verificar se a câmera está aberta
        if cap.isOpened():
            cameras_ativas.append(i)
            cap.release()

    return cameras_ativas

# Chamar a função para verificar as câmeras ativas
cameras = verificar_cameras()

# Exibir as câmeras ativas
if len(cameras) > 0:
    print("Câmeras ativas encontradas nos seguintes índices:")
    for camera in cameras:
        print(camera)
else:
    print("Nenhuma câmera ativa encontrada.")
