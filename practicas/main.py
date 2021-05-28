import imageio as io
import click

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
    img = read_img(p)
    img_compressed = compresion_de_rango_dinamico(img.reshape((-1)))
    show_imgs([img, img_compressed.reshape(img.shape)])
    plt.hist(img_compressed, bins=256)
    plt.show()


@cli.command()
@click.option("-p", default="imagenes/boat.png")
def negate_image(p):
    """
    Lee la imagen y devuelve la imagen en negativo
    """
    img = read_img(p)
    img_negative = negativo(img.reshape((-1)))
    show_imgs([img, img_negative.reshape(img.shape)])


@cli.command()
@click.option("-p", default="imagenes/boat.png")
def histograma(p):
    """
    Lee la imagen y muestra el histograma de los niveles de gris
    """
    img = read_img(p)
    show_histograma(img.reshape((-1)))


@cli.command()
@click.option("-p", default="imagenes/boat.png")
@click.option("-r1", default=150)
@click.option("-r2", default=250)
@click.option("-s1", default=50)
@click.option("-s2", default=250)
def contrastear_img(p, r1, r2, s1, s2):
    """
    Lee la imagen y muestra el la imagen contrasteada
    """
    img = read_img(p)
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
    img = read_img(p)
    if (use_contrast):
        funcion_de_constraste = contrast_function_for_points(umbral, umbral, 0, 255)
        img_binarizada = contrastear(img.reshape((-1)), funcion_de_constraste)
    else:
        img_binarizada = binarizar(img.reshape((-1)), umbral)
    show_imgs([img, img_binarizada.reshape(img.shape)])


@cli.command()
@click.option("-p", default="imagenes/boat.png")
def ecualizar_img(p):
    """
    Lee la imagen y muestra la imagen y su histograma
    """
    img = read_img(p)

    img1_histogram, bin_edges1 = np.histogram(img.ravel(), bins=256)
    ecualized_img = ecualizar_histograma(img, img1_histogram)
    img2_histogram, bin_edges2 = np.histogram(ecualized_img.ravel(), bins=256)
    double_ecualized_img = ecualizar_histograma(ecualized_img, img2_histogram)
    img3_histogram, bin_edges3 = np.histogram(double_ecualized_img, bins=256)

    _, axarr = plt.subplots(1, 3)
    axarr[0].plot(bin_edges1[0:-1], img1_histogram)
    axarr[1].plot(bin_edges2[0:-1], img2_histogram)
    axarr[2].plot(bin_edges3[0:-1], img3_histogram)
    plt.show()

    show_imgs([img, ecualized_img, double_ecualized_img])

@cli.command()
@click.option("-p", default="imagenes/kodim02.png")
@click.option("-l", default=1)
@click.option("-g", default=1)
def enhance_histogram(p, l, g):
    """
    Lee la imagen y muestra la imagen enhanced con el lambda l
    """
    img = read_img(p)
    hi, h_opt, enhanced_img = histogram_enhancement(img, l, g)
    plt.plot(list(range(len(h_opt))), hi, color='blue')
    plt.plot(list(range(len(h_opt))), h_opt, color='red')
    plt.figlegend(["original", "enhanced"])
    plt.show()
    show_imgs([img, enhanced_img])



def read_img(path):
    img = io.imread(path)
    try:
        return convert_to_grayscale(img)
    except:
        return img


if __name__ == "__main__":
    cli()
