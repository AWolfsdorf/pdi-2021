import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


def compresion_de_rango_dinamico(array):
    c = 255 / math.log(array.max() + 1, 10)
    compression_function = lambda r: round(c * math.log(r + 1, 10))
    return np.array([compression_function(r) for r in array])


def negativo(img):
    return np.array([255 - pixel for pixel in img])


def show_histograma(img):
    plt.hist(img, bins=256)
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


def accumulative_relative_frequency_function(r, elems, count, n):
    accum = 0
    for index in range(len(elems)):
        frequency = count[index]
        accum += frequency / n
        r_index = elems[index]
        if r_index == r:
            break
    return accum


def ecualizar_histograma(img):
    n = img.size
    shape = img.shape
    arrayed_img = img.reshape((-1))
    elems, count = np.unique(arrayed_img, return_counts=True)
    return np.array([accumulative_relative_frequency_function(r, elems, count, n) for r in arrayed_img]).reshape(shape)


def histogram_enhancement(img, lam):
    flattened_img = img.ravel()
    hi, _ = np.histogram(flattened_img, bins=256)
    h_opt = (hi + np.random.uniform(0, 1, len(hi)) * lam) / (lam + 1)
    plt.plot(list(range(len(h_opt))), hi, color='blue')
    plt.plot(list(range(len(h_opt))), h_opt, color='red')
    plt.figlegend(["original", "enhanced"])
    plt.show()
    # enhanced_img = ((flattened_img + np.random.uniform(0, 1, len(flattened_img)) * lam) / (lam + 1)).astype(np.uint8).reshape(img.shape)
    # show_imgs([img, enhanced_img])


def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)


def show_imgs(imgs, cmap='gray'):
    _, axarr = plt.subplots(1, len(imgs))
    for i in range(len(imgs)):
        axarr[i].imshow(imgs[i], cmap)
    plt.show()