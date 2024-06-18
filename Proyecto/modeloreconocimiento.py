from picamera2 import Picamera2, Preview
import cv2
import time
import os
import numpy as np

# Ruta base donde se almacenan las imágenes de entrenamiento
datapath = 'data'

# Lista de nombres de las personas (carpetas en el directorio de datos)
listaPersonas = os.listdir(datapath)

# Inicialización de listas para etiquetas y datos de rostros
labels = []
facesData = []
label = 0

# Lectura de las imágenes para cada persona en la lista
for nameDir in listaPersonas:
    # Ruta completa de la persona actual
    personPath = os.path.join(datapath, nameDir)
    print('Leyendo las imágenes de', personPath)

    # Lectura de cada imagen en el directorio de la persona
    for fileName in os.listdir(personPath):
        print('Rostros:', nameDir + '/' + fileName)
        # Añade la etiqueta correspondiente y lee la imagen en escala de grises
        labels.append(label)
        facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))
    label += 1

# Crear el reconocedor de rostros utilizando el método LBPH (Local Binary Patterns Histograms)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros con las imágenes leídas y sus etiquetas
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido en un archivo
modelPath = 'modeloLBPH.xml'
face_recognizer.write(modelPath)
print(f"Modelo almacenado en {modelPath}")
