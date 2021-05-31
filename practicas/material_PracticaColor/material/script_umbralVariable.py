import cv2
import matplotlib.pyplot as plt
import numpy as np

ig = cv2.imread('./img/cameraman.tif',0)
cv2.imshow('imagen',ig)
h = cv2.calcHist([ig],[0],None,[256],[0,256])
# normalizo el histograma
h = h / h.sum()
h = h.flatten()
plt.bar(range(0,256),h)
plt.show()
# elijo el primer umbral a la mitad de la dinamica de grises
T = 180
# defino un error calculado entre el anterior umbral y el actual
e = 3
dif = 6

while dif > e:
    x1 = np.array(range(1,T))
    h1 = h[1:T] / sum(h[1:T])
    m1 = np.sum(x1 * h1)
    
    x2 = np.array(range(T,len(h)-1))
    h2 = h[T:-1] / sum(h[T:-1])
    m2 = np.sum(x2 * h2)
    
    Tant = T
    
    T = np.int((m2 - m1) / 2)
    print('m2 ', T)
    
    dif = np.abs(T - Tant)    


plt.figure(2)
plt.imshow(ig > T)
plt.show()