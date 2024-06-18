import cv2
import os
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import time

# Configuración del pin del LED
led1 = 21
detectado = False

# Configuración de GPIO
# Desactivar advertencias
GPIO.setwarnings(False)

# Usar numeración BCM
GPIO.setmode(GPIO.BCM)

# Configurar el pin como salida
GPIO.setup(led1, GPIO.OUT)

# Asegurarse de que el LED esté apagado inicialmente
GPIO.output(led1, False)

# Ruta base donde se almacenan las imágenes de entrenamiento
datapath = 'data'
imagepath = os.listdir(datapath)
print('imagePaths =', imagepath)

# Crear el reconocedor de caras utilizando el método LBPH (Local Binary Patterns Histograms)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
face_recognizer.read('modeloLBPH.xml')

# Cargar el clasificador de cascada para la detección de rostros
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inicializar la cámara con Picamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)}, lores={"size": (320, 240)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2) # Esperar a que la cámara se estabilice

# Variable para controlar el tiempo de encendido del LED
led_on_time = 0

while True:
    # Capturar un fotograma de la cámara
    frame = picam2.capture_array()
    
    # Convertir de RGB a BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Crear una copia del fotograma en gris
    auxFrame = gray.copy()

    # Detectar rostros en la imagen
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    detectado = False

    for (x, y, w, h) in faces:
        # Extraer el rostro detectado y redimensionarlo
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        # Realizar predicción utilizando el reconocedor de caras
        result = face_recognizer.predict(rostro)
        
        # Mostrar resultados en el fotograma
        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 70:
            cv2.putText(frame, '{}'.format(imagepath[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detectado = True
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Mostrar el fotograma en una ventana
    cv2.imshow('Reconocimiento Facial', frame)
    k = cv2.waitKey(1)
    
    # Control del LED
    if detectado:
        GPIO.output(led1, detectado) # Encender el LED
        led_on_time = time.time() # Actualizar el tiempo de encendido del LED
    elif time.time() - led_on_time > 3:
        GPIO.output(led1, False)  # Apagar el LED después de 3 segundos
        
         
    if k == 27:  # Presionar 'ESC' para salir
        break


print("Cerrando...")
cv2.destroyAllWindows()
GPIO.output(led1, False)  # Asegurarse de que el relé esté apagado
GPIO.cleanup() # Limpiar la configuración de GPIO
picam2.close() # Cerrar la cámara