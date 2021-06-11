function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcula costo y gradiente de una red neuronal convolucional simple
% seguida de una capa densa con una funcion objetivo de tipo softmax
%                            
% Parametros:
%  theta      -  unrolled vector de pesos
%  images     -  la lista de imagenes de entrenamiento imageDim x imageDim x numImges
%                
%  numClasses -  numbero de clases
%  filterDim  -  dimensiones del filtro convolucional                            
%  numFilters -  numero de filtros
%  poolDim    -  dimension del area de agrupamiento
%  pred       -  booleano indicando si solo se precisa hacer un fordware
%  propagete y delvolver predicciones
%             
%
%
% Devuelve:
%  cost       -  costro softmax
%  grad       -  gradientes con respecto a theta (si pred==False)
%  preds      -  lista de predicciones para cada ejemplo (si pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width de imagen
numImages = size(images,3); % numbero de imagenes

%% Obtener las matrices de parametros del roll de entrada 
% Wc es la matriz de pesos filterDim x filterDim x numFilters
% bc es el bias correspondiente

% Wd es la matriz densa de la capa escondida con numClasses x hiddenSize
% hiddenSize es el numero de unidades a la salida de la capa de pooling
% bd es su bias correspondient
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% reservar para los gradientes de los parametros.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% PASO 1a: Forward Propagation
%  Este paso propaga la entrada de en la capa convolucional, 
%  sabsampleo (mean pooling), y que luego son usadas como entradas de la 
%  capa de softmax.

%% Convolutional Layer
%  Para cada imagen de entrada y filtro, realizar la convolucion, sumar el
%  bias y aplicar la funcion de activacion sigmoid. Luego, subsamplear
%  con mean pooling. Guardar los resultados de la convolucion en 
%  activations y los resultados del pooling en activationsPooled.
%  Es preciso conservarlos para el backpropagation.
%  Utilizar las funciones cnnConvolve y cnnPool

convDim = imageDim-filterDim+1; % dimension de la salida convolved
outputDim = (convDim)/poolDim; % dimension de la salida subsampled

% tensor convDim x convDim x numFilters x numImages 
activations = zeros(convDim,convDim,numFilters,numImages);
% tensor outputDim x outputDim x numFilters x numImages
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

activations = cnnConvolve(filterDim,numFilters,images,Wc,bc);
activationsPooled = cnnPool(poolDim,activations);

% Redimensionar activations obteniendo 2D matriz, hiddenSize x numImages,
% para Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Continuamos con la Forward propagation luego de las pooledActivations 
%  para aplicarlas a una capa softmax standard. probs recibe los resultados
%  numClasses x numImages que va a guardar la probabilidad de la pertenencia
%  de cada imagen a una clase.
probs = zeros(numClasses,numImages);

%%% IMPLEMENTACION AQUI %%%
hiddenSize = size(activationsPooled,1);

%%======================================================================
%% PASO 1b: Calculo del Costo
%  Se van a usar los labels y las probabilidades para calcular la funcion
%  objetivo de tipo softmax. El resultado se guarda en cost.

cost = 0; % inicializo cost

%%% IMPLEMENTAR AQUI %%%

%%%%%%%%%%%%%%%%%%%%%%%%%
% Realizar predicciones usando probs y devolver la funcion si calculo de gradientes.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% PASO 1c: Backpropagation
%  Backpropagar los errores a traves de la capa de salida, convolucional y pooling.
%  Guardar los errores para el siguiente paso para el calculo del gradiente.
%  Backpropagar el error para la capa softmax es similar al Ejercicio 1.
%  Para backpropagar en la capa de pooling, hay que des-agrupar el error
%  con respecto a la capa de de pooling para cada filtro y cada imagen.
%  Se puede usar la funcion kron y una matriz de unos para hacerlo de manera
%  eficiente.

%%% IMPLEMENTAR AQUI %%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%======================================================================
%% PASO 1d: Calculo del Gradiente
%  Luego de propagar los errores, deben ser usados para calcular el gradiente
%  con respecto a los pesos. El calculo del gradiente de la capa softmax se 
%  calculo de la manera usual. Para calcular el gradiente de un filtro en la capa convolucional
%  se debe hacer la convolucion entren el error propagado de ese filtro 
%  para cada imagen y acumularlo sobre todas las imagenes.

%%% YOUR CODE HERE %%%


%% Desenrollar los gradientes en un vector para usar luego por minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
