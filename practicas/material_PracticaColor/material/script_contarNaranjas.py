import cv2
import matplotlib.pyplot as plt
import numpy as np

i = cv2.imread('naranjo.jpg')
cv2.imshow('imagen',i)

# generar una matriz naranjas binaria con los pixeles pertenecientes a naranjas en 1 y 0 en los otros.
######################################################################################################


#####################################################################################################
cv2.imshow('naranjas',naranjas)

cont0, hierarchy = cv2.findContours(naranjas.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contornos = cont0
# graficar los contornos restantes
cv2.namedWindow('contornos filtrados')
h, w = naranjas.shape[:2]
vis = np.zeros((h, w), np.uint8)
for i in range(0,len(contornos)-1):
    cv2.drawContours( vis, contornos, i, 255)
cv2.imshow('contornos filtrados',vis)

print('Hay %d naranjas' % len(contornos))