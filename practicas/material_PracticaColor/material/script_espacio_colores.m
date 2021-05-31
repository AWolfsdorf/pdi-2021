close all
% conversion a distintos espacios de color
% lectura de la imagen color
img = imread('crayones.tif');

figure(1)
subplot(2,2,1)
imshow(img)
subplot(2,2,2)
imshow(img(:,:,1));
subplot(2,2,3)
imshow(img(:,:,2));
subplot(2,2,4);
imshow(img(:,:,3));

% conversion a niveles de gris
img_gris = rgb2gray(img);
figure(2)
subplot(1,2,1)
imshow(img)
subplot(1,2,2)
imshow(img_gris)

% conversion a hsv
img_hsv = rgb2hsv(img);
figure(3)
subplot(2,2,1)
imshow(img)
title('original')
subplot(2,2,2)
imshow(img_hsv(:,:,1));
title('hue');
subplot(2,2,3)
imshow(img_hsv(:,:,2));
title('saturacion')
subplot(2,2,4);
imshow(img_hsv(:,:,3));
title('intensidad');

% conversion a ycbcr
img_ycbcr = rgb2ycbcr(img);
figure(4)
subplot(2,2,1)
imshow(img)
title('imagen original')
subplot(2,2,2)
imshow(img_ycbcr(:,:,1));
title('Y');
subplot(2,2,3)
imshow(img_ycbcr(:,:,2));
title('cb')
subplot(2,2,4);
imshow(img_ycbcr(:,:,3));
title('cr')