import cv2
import os
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import time

led1 = 21
detectado = False

GPIO.setwarnings(False)  # Desactivar advertencias
GPIO.setmode(GPIO.BCM)
GPIO.setup(led1, GPIO.OUT)
GPIO.output(led1, False)

datapath = 'data'
imagepath = os.listdir(datapath)
print('imagePaths =', imagepath)

# Crear el reconocedor de caras
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


# Leyendo el modelo
face_recognizer.read('modeloLBPH.xml')

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inicializar la cámara con Picamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)}, lores={"size": (320, 240)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

led_on_time = 0

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    detectado = False

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        # EigenFaces
        if result[1] < 70:
            cv2.putText(frame, '{}'.format(imagepath[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detectado = True
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Reconocimiento Facial', frame)
    k = cv2.waitKey(1)
    if detectado:
        GPIO.output(led1, detectado)
        led_on_time = time.time()
    elif time.time() - led_on_time > 3:
        GPIO.output(led1, False)
    if k == 27:  # Presionar 'ESC' para salir
        break


print("Cerrando...")
cv2.destroyAllWindows()
GPIO.output(led1, False)  # Asegurarse de que el relé esté apagado
GPIO.cleanup()
picam2.close()