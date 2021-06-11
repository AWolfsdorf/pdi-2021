%% Verificacion de Datos de Entrada

% Instrucciones
% -------------
%
% Descargar desde la pagina http://www.ipol.im/pub/art/2018/173/
% el archivo correspondiente al source code: svmsmo_1.tar.gz
% Descomprimirlo en este directorio.
%%======================================================================

%% Convolution and Pooling Verificacion

%  Instrucciones
%  ------------
% 
%  Vamos a verificar la implementacion del las operaciones de convolucion 
%  y pooling. Se deben modificar las funciones en los archivos 
%  cnnConvolve.m and cnnPool.m. 
%%======================================================================
%% PASO 0: Levantar los datos e inicializar variables
%  Se inicializar parametros para la verificacion.

source = 0;
%  Hay 2 bases:
%           source = 0 -> Es la base MNIST de digitos manuscritos.
%           source = 1 -> Los caracteres de la base corresponden a 
%           numeros de matriculas de vehiculos. Tiene una forma 
%           rectangular, que vamos a redimensionar a un cuadrado de 
%           tamanio imageDim x imageDim
imageDim = 28;         % image dimension

if source == 0
    % Lectura de MNIST training imagenes
    addpath('common');
    images = loadMNISTImages(fullfile('common','train-images.idx3-ubyte'));
    images = reshape(images,imageDim,imageDim,[]);
    labels = loadMNISTLabels(fullfile('common','train-labels.idx1-ubyte'));
else
    assert(exist(fullfile('svm_smo','SVMCode','Datasets','BaseOCR_MultiStyle'),'dir') > 0,...
        'Descargar y descomprimir el archivo svmsmo_1.tar.gz de http://www.ipol.im/pub/art/2018/173/');
    % Lectura de set LPR
    [images,labels] = loadLPRImages(imageDim);
    numImages = length(labels);
    images = reshape(images,imageDim,imageDim,numImages);
end
%%======================================================================
%% PASO 1: Implementar y Testear la convolucion
%  En esta etapa, se debe implementar la operacion de convolucion. Se
%  ofrece un test en un pequeño set para asegurar que la implementacion es
%  correcta.
filterDim = 8;          % tamanio del filtro
numFilters = 100;         % cantidad de filtros
poolDim = 3;          % dimension of pooling region

%%  1a: Implementar la funcion cnnConvolve en cnnConvolve.m
%  Usamos un set pequeño de imagenes: las 8 primeras
convImages = images(:, :, 1:8); 

% Inicializo matrices aleatorias de pesos que van a ser utilizadas en la
% verificacion de la funcion de Convolucion
W = randn(filterDim,filterDim,numFilters);
b = rand(numFilters);
% Ahora realizamos la operacion de convolucion de las imagenes llamando a
% cnnConvolve y pasandole todos los parametros reque
convolvedFeatures = cnnConvolve(filterDim, numFilters, convImages, W, b);

%% 1b: Chequeamos la convolucion encontrada
%  El codigo siguiente se chequea el resultado devuelto por cnnConvolve.
%  To ensure that you have convolved the features correctly, we have
%  provided some code to compare the results of your convolution with
%  activations from the sparse autoencoder

%  Tomando 1000 puntos random
for i = 1:1000   
    filterNum = randi([1, numFilters]);
    imageNum = randi([1, 8]);
    imageRow = randi([1, imageDim - filterDim + 1]);
    imageCol = randi([1, imageDim - filterDim + 1]);    
   
    patch = convImages(imageRow:imageRow + filterDim - 1, imageCol:imageCol + filterDim - 1, imageNum);

    feature = sum(sum(patch.*W(:,:,filterNum)))+b(filterNum);
    feature = 1./(1+exp(-feature)); % sigmoide
    
    if abs(feature - convolvedFeatures(imageRow, imageCol,filterNum, imageNum)) > 1e-9
        fprintf('La salida de cnnConvolve no coincide con el test\n');
        fprintf('Numero de filtro  : %d\n', filterNum);
        fprintf('Numero de imagen  : %d\n', imageNum);
        fprintf('Fila Imagen       : %d\n', imageRow);
        fprintf('Columna Imagen    : %d\n', imageCol);
        fprintf('Convolved feature : %0.5f\n', convolvedFeatures(imageRow, imageCol, filterNum, imageNum));
        fprintf('Test feature : %0.5f\n', feature);       
        error('La salida de cnnConvolve no coincide con el test');
    end 
end

disp('Felicitacines! Su implementacion paso el test.');

%%======================================================================
%% PASO 2: Implementacion y Test de Pooling
%  Aqui implementamos la funcion cnnPool en cnnPool.m

%% 2a: Implementar pooling
%  Vamos a usar la salida de cnnConvolve como entrada al pooling
pooledFeatures = cnnPool(poolDim, convolvedFeatures);

%% 2b: Chequeo de la implementacion

testMatrix = reshape(1:64, 8, 8);
expectedMatrix = [mean(mean(testMatrix(1:4, 1:4))) mean(mean(testMatrix(1:4, 5:8))); ...
                  mean(mean(testMatrix(5:8, 1:4))) mean(mean(testMatrix(5:8, 5:8))); ];
            
testMatrix = reshape(testMatrix, 8, 8, 1, 1);
        
pooledFeatures = squeeze(cnnPool(4, testMatrix));

if ~isequal(pooledFeatures, expectedMatrix)
    disp('Pooling incorrecto');
    disp('Esperado');
    disp(expectedMatrix);
    disp('Resultado');
    disp(pooledFeatures);
    error('Pooling incorrecto');
else
    disp('Felicitaciones! El codigo de pooling es correcto.');
end

%% Convolution Neural Network 

%  Instrucciones
%  ------------
% 
%  En esta parte de construye una red neuronal convolucional simple.
%  El archivo que hay que modificar es cnnCost.m. Aqui tambien existe un
%  test para verificar si los valores de los gradientes son correctos

% Preparacion de los datos
labels(labels==0) = 10; % Remapeo la etiqueta del cero a 10
%%======================================================================
%% PASO 1: Implementar cnnCost
% Configuracion
imageDim = 28;
numClasses = 10;  % Numbero de clases (MNIST images fall into 10 classes)
filterDim = 9;    % Tamaño del filtro en la capa de convolucion
numFilters = 20;   % Numbero de filtros
poolDim = 2;      % Dimension del pooling, (debe dividir a imageDim-filterDim+1)

%%======================================================================
%% PASO 2: Chequeo del gradiente
%  Se chequea el calculo del gradiente de la funcion cnnCost.m.  

DEBUG = true;  % cambiar a verdadero para el test
if DEBUG
    % El chequeo se realiza sobre un set mas pequeño de imagenes
    db_numFilters = 2;
    db_filterDim = 9;
    db_poolDim = 5;
    db_images = images(:,:,1:11);
    db_labels = labels(1:11);
    db_theta = cnnInitParams(imageDim,db_filterDim,db_numFilters,...
                db_poolDim,numClasses);
    
    [cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,...
                                db_filterDim,db_numFilters,db_poolDim);
    

    % Calculo de gradiente numerico o aproximado
    numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
                                db_labels,numClasses,db_filterDim,...
                                db_numFilters,db_poolDim), db_theta);
 
   
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Diff deberia ser muy pequeño. 
    disp(diff); 
 
    assert(diff < 1e-9,...
        'Se obtuvo un gradiente con una diferencia muy grande. La implementacion no es correcta.'); 
end

%%======================================================================
%% PASO 3: Aplicar el aprendizaje sobre una base de Entrenamiento
% Inicializacion de los Pesos
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);

if source == 0
    options.epochs = 3;
    options.minibatch = 256;
else
    options.epochs = 10;
    options.minibatch = 48;
end
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,...
                      numFilters,poolDim),theta,images,labels,options);

%%======================================================================
%% PASO 4: Test
%  Test de performance del modelo entrenado. 
%  La base MNIST tiene disponibles ejemplos de testing. En el caso de la 
%  base LPR lo testeo sobre el mismo set de aprendizaje.

if source == 0
    testImages = loadMNISTImages(fullfile('common','t10k-images.idx3-ubyte'));
    testImages = reshape(testImages,imageDim,imageDim,[]);
    testLabels = loadMNISTLabels(fullfile('common','t10k-labels.idx1-ubyte'));
    testLabels(testLabels==0) = 10; % Remap 0 to 10
    [~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                    filterDim,numFilters,poolDim,true);
else
    [~,cost,preds]=cnnCost(opttheta,images,labels,numClasses,...
                    filterDim,numFilters,poolDim,true);
end
acc = sum(preds==labels(:))/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);

% Vamos a graficar los filtros encontrados
[Wc, Wd, bc, bd] = cnnParamsToStack(opttheta,imageDim,filterDim,numFilters,poolDim,numClasses);
% Hay numFilters 
qCol = 4;
qLin = numFilters / qCol;
figure
for i=1:numFilters
    subplot(qLin,qCol,i)
    filtro = squeeze(Wc(:,:,i));
    filtro = (filtro - min(filtro(:)));
    filtro = filtro / max(filtro(:));
    imshow(filtro);
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    axis equal
end



