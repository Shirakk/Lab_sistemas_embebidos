from picamera2 import Picamera2, Preview
import cv2
import time

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (640, 480)}, lores={"size": (320,240)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if faceClassif.empty():
    print("error")

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceClassif.detectMultiScale(gray, 1.3,5)
    
    for(x, y, w, h) in faces:
        cv2.rectangle (frame, (x,y), (x + w, y + h), (0,255,0), 2)
        
    cv2.imshow('Ventana', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('s'):
            break
            
cv2.destroyAllWindows()
picam2.close()