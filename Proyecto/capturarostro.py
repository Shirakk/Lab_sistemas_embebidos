from picamera2 import Picamera2, Preview
import cv2
import time
import os
import imutils

# Nombre del usuario para quien se capturarán las imágenes
nombre = 'Agustin'

# Ruta base donde se almacenarán las imágenes
datapath = 'data'

# Ruta completa donde se almacenarán las imágenes del usuario
nombrepath = os.path.join(datapath,nombre)

# Crea la carpeta 'nombrepath' si no existe
if not os.path.exists(nombrepath):
    os.makedirs(nombrepath)

# Inicializa la cámara Picamera2
picam2 = Picamera2()

# Configura la cámara con resoluciones específicas
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)}, lores={"size": (320,240)})
picam2.configure(camera_config)

# Inicia la cámara
picam2.start()

# Espera 2 segundos para permitir que la cámara se estabilice
time.sleep(2)

# Carga el clasificador de Haar para la detección de rostros
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Contador para el número de imágenes capturadas
count = 0

# Bucle principal para captura de imágenes
while True:
    # Captura una imagen de la cámara
    frame = picam2.capture_array()
    
    # Convierte la imagen capturada de RGB a BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Redimensiona la imagen a un ancho de 640 píxeles
    frame = imutils.resize(frame, width= 640)
    
    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Copia el fotograma para la detección de rostros
    auxFrame = frame.copy()
    
    # Detecta rostros en la imagen en escala de grises
    faces = faceClassif.detectMultiScale(gray, 1.3,5)
    
    # Itera sobre cada rostro detectado
    for(x, y, w, h) in faces:
        # Dibuja un rectángulo alrededor de cada rostro detectado
        cv2.rectangle (frame, (x,y), (x + w, y + h), (0,255,0), 2)
        
        # Extrae el rostro detectado
        rostro = auxFrame[y:y + h, x:x + w]
        
        # Redimensiona el rostro detectado a 150x150 píxeles
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        # Guarda el rostro detectado en el directorio correspondiente
        cv2.imwrite(os.path.join(nombrepath, 'rostro_{}.jpg'.format(count)), rostro)
        
        # Incrementa el contador de imágenes
        count += 1

    # Muestra el fotograma con los rostros detectados
    cv2.imshow('frame', frame)

    # Espera por la tecla presionada
    k = cv2.waitKey(1)
    
    # Si se presiona 'ESC' o se capturan 300 imágenes, sale del bucle
    if k == 27 or count >= 300:  
        break
    
# Cierra todas las ventanas de OpenCV
cv2.destroyAllWindows()

# Cierra la cámara Picamera2
picam2.close() 
