import numpy as np
import math
import matplotlib.pyplot as plt


def compresion_de_rango_dinamico(array):
    c = 255 / math.log(array.max() + 1, 10)
    compression_function = lambda r: round(c * math.log(r + 1, 10))
    return np.array([compression_function(r) for r in array])


def negativo(img):
    return np.array([255 - pixel for pixel in img])


def show_histograma(img):
    plt.hist(img)
    plt.show()


def contrast_function_for_points(r1, r2, s1, s2):
    f1_coefficients = np.polyfit([0, r1], [0, s1], 1)
    f2_coefficients = np.polyfit([r1, r2], [s1, s2], 1)
    f3_coefficients = np.polyfit([r2, 255], [s2, 255], 1)
    line_function_for_coefs = lambda coef: (lambda x: coef[0] * x + coef[1])
    return contrast_function(
        r1,
        r2,
        line_function_for_coefs(f1_coefficients),
        line_function_for_coefs(f2_coefficients),
        line_function_for_coefs(f3_coefficients)
    )


def contrast_function(r1, r2, f1, f2, f3):
    return lambda x: f1(x) if x < r1 else (f3(x) if x > r2 else f2(x))


def contrastear(img, funcion_de_constraste):
    return np.array([funcion_de_constraste(pixel) for pixel in img])


def binarizar(img, umbral):
    binary_function = lambda x: 0 if x <= umbral else 255
    return np.array([binary_function(pixel) for pixel in img])
