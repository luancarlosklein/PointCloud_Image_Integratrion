from cvlib.object_detection import YOLO
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

# Carregar o modelo YOLO
yolo = YOLO()

# Carregar a imagem
image_path = "image334.jpg"
image = cv2.imread(image_path)
print(image)
# Detectar objetos na imagem
bbox, label, conf = cv.detect_common_objects(image, confidence=0.25, model='yolov4-tiny')

print(bbox)
print(label)
print(conf)

# Desenhar os retângulos delimitadores na imagem
image = draw_bbox(image, bbox, label, conf)

# Exibir a imagem com os objetos detectados
cv2.imshow("Detecção de objetos", image)
a = input()
cv2.destroyAllWindows()