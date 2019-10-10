import cv2
import numpy as np
import winsound

# Direccion del proyecto
path = "C:/User/chffjkrv/Documents/Python/DETECTOR SEGURIDAD"

# Archivos con los datos de las cascadas Haar para detección
# de caras y de los ojos.

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cv2.namedWindow("Camara del Movil") # Crea ventana de OpenCV
cap = cv2.VideoCapture(0) # Le asigna a la ventana el canal de DroidCam

# Se inicia el bucle en el que se analiza en directo la imagen
# en busca de caras y ojos.
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    # Emite un sonido si detecta una cara en la imagen
   # if len(faces) != 0:
    #winsound.Beep(700, 100)

    # Coloca rectangulos marcando la posición de la cara y de los ojos
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow("Camara del Movil", img)

    # Rompe el bucle si se presiona la tecla de escape
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
