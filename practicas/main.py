import matplotlib.pyplot as plt
import imageio as io
import click
import cv2

from OperadorePuntualesEHistogramas import *


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    global DEBUG
    DEBUG = debug
    pass


@cli.command()
@click.option("-p", default="imagenes/boat.png")
def comprimir_en_rango_dinamico(p):
    """
    Lee la imagen y devuelve la imagen comprimida en rango dinamico
    """
    img = convert_to_grayscale(io.imread(p))
    original_shape = img.shape
    img_compressed = compresion_de_rango_dinamico(img.reshape((-1)))
    show_imgs([img, img_compressed.reshape(img.shape)])


@cli.command()
@click.option("-p", default="imagenes/boat.png")
def negate_image(p):
    """
    Lee la imagen y devuelve la imagen en negativo
    """
    img = convert_to_grayscale(io.imread(p))
    img_negative = negativo(img.reshape((-1)))
    show_imgs([img, img_negative.reshape(img.shape)])


@cli.command()
@click.option("-p", default="imagenes/boat.png")
def histograma(p):
    """
    Lee la imagen y muestra el histograma de los niveles de gris
    """
    img = convert_to_grayscale(io.imread(p))
    show_histograma(img.reshape((-1)))


@cli.command()
@click.option("-p", default="imagenes/boat.png")
@click.option("-r1", default=50)
@click.option("-r2", default=180)
@click.option("-s1", default=20)
@click.option("-s2", default=230)
def contrastear_img(p, r1, r2, s1, s2):
    """
    Lee la imagen y muestra el la imagen contrasteada
    """
    img = convert_to_grayscale(io.imread(p))
    funcion_de_constraste = contrast_function_for_points(r1, r2, s1, s2)
    img_contrasteada = contrastear(img.reshape((-1)), funcion_de_constraste)
    show_imgs([img, img_contrasteada.reshape(img.shape)])



@cli.command()
@click.option("-p", default="imagenes/boat.png")
@click.option("-umbral", default=123)
@click.option("-use_contrast", default=False)
def binarizar_img(p, umbral, use_contrast):
    """
    Lee la imagen y muestra el la imagen binarizada en un cierto umbral
    """
    img = convert_to_grayscale(io.imread(p))
    if (use_contrast):
        funcion_de_constraste = contrast_function_for_points(umbral, umbral, 0, 255)
        img_binarizada = contrastear(img.reshape((-1)), funcion_de_constraste)
    else:
        img_binarizada = binarizar(img.reshape((-1)), umbral)
    show_imgs([img, img_binarizada.reshape(img.shape)])


def convert_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)


def show_imgs(imgs, cmap='gray'):
    _, axarr = plt.subplots(1, len(imgs))
    for i in range(len(imgs)):
        axarr[i].imshow(imgs[i], cmap)
    plt.show()


if __name__ == "__main__":
    cli()
