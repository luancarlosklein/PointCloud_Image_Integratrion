import cv2

# Função para exibir o vídeo da webcam
def exibir_video():
    # Inicializar o objeto de captura de vídeo
    cap = cv2.VideoCapture(6)

    while True:
        # Ler o próximo quadro do vídeo
        ret, frame = cap.read()

        # Exibir o quadro resultante
        cv2.imshow('Webcam', frame)

        # Verificar se a tecla 'q' foi pressionada para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    cap.release()
    cv2.destroyAllWindows()

# Chamar a função para exibir o vídeo da webcam
exibir_video()