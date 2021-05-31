% lectura de la imagen
img = imread('2011_003410.jpg');

[M,N,pipo] = size(img);

% extraccion de un parche para identificar la region de piel
r = roipoly(img);

% Utilizar el espacio RGB y el espacio YCbCr para hacer deteccion de pixels de piel.
% Comparar.