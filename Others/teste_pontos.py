
import cv2
import numpy as np

# Função para desenhar pontos específicos em uma imagem
def desenhar_ponto(img, x, y):
    # Definir a cor do ponto (verde no exemplo)
    cor = (0, 255, 0)
    # Definir o tamanho do ponto
    tamanho = 3
    # Desenhar o ponto na imagem
    cv2.circle(img, (x, y), tamanho, cor, -1)

# Criar uma imagem em branco
largura, altura = 800, 600
imagem = np.zeros((altura, largura, 3), dtype=np.uint8)

# Exemplo de coordenadas de pontos
pontos = [(100, 100), (200, 300), (400, 200)]

# Desenhar os pontos na imagem
for ponto in pontos:
    x, y = ponto
    desenhar_ponto(imagem, x, y)

# Mostrar a imagem com os pontos desenhados
cv2.imshow("Imagem com Pontos", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()