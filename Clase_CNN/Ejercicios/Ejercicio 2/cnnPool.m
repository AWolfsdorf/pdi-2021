function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%  cnnPool Pools los descriptores provenientes de la convolucion
%  La funcion usa el Pool promedio
% Parametros:
%  poolDim - dimension de la regiom de pool
%  convolvedFeatures - los descriptores a realizar el pool 
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Devuelve:
%  pooledFeatures - matriz de los features agrupados
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instrucciones:
%   Realizar el pool de los features en regiones de tamaño poolDim x poolDim,
%   para obtener la matriz pooledFeatures de 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 

%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) es el valor 
%   del descriptor featureNum de la imagen imageNum agrupada sobre la 
%   region (poolRow, poolCol). 
%   

%%% IMPLEMENTAR AQUI %%%
