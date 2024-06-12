from picamera2 import Picamera2, Preview
import cv2
import time
import os
import imutils

nombre = 'Agustin'
datapath = 'data'
nombrepath = os.path.join(datapath,nombre)

if not os.path.exists(nombrepath):
    os.makedirs(nombrepath)

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)}, lores={"size": (320,240)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = imutils.resize(frame, width= 640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    
    faces = faceClassif.detectMultiScale(gray, 1.3,5)
    
    for(x, y, w, h) in faces:
        cv2.rectangle (frame, (x,y), (x + w, y + h), (0,255,0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(nombrepath, 'rostro_{}.jpg'.format(count)), rostro)
        count += 1

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:  # Presionar 'ESC' para salir o cuando se tengan 300 imágenes
        break

cv2.destroyAllWindows()
picam2.close() 