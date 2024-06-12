from picamera2 import Picamera2, Preview
import cv2
import time
import os
import numpy as np

datapath = 'data'
listaPersonas = os.listdir(datapath)

labels = []
facesData = []
label = 0

# Lectura de las imágenes
for nameDir in listaPersonas:
    personPath = os.path.join(datapath, nameDir)
    print('Leyendo las imágenes de', personPath)

    for fileName in os.listdir(personPath):
        print('Rostros:', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))
    label += 1

# Métodos para entrenar el reconocedor
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
modelPath = 'modeloLBPH.xml'
face_recognizer.write(modelPath)
print(f"Modelo almacenado en {modelPath}")