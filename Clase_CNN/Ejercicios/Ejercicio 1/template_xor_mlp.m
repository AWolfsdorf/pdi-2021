% La entrada consiste en un array cuyas columnas son los ejemplos de
% aprendizaje y cuyas lineas son los descriptores
X = [1 1 -1 -1; 1 -1 1 -1];
[Ni, M]          = size(X);
No		         = 1;
Nh               = 2; 
eta              = 0.25;  % ratio de aprendizaje
epochs           = 10; % son las iteraciones para el aprendizaje
% Las salidas esperadas o targets
T = [-1 1 1 -1];

% Se inicializa la red
% Hay una sola neurona de salida, con lo cual entre las neuronas escondidas
% y la salida hay un vector de pesos. Entre la entrada y las neuronas
% escondidas hay una matriz de pesos.

% Hidden weights (NhxNi)
Wh = [0.3615 -1.4145; -0.8916 0.2010];
%Output weights (NoxNh)
Wo = [-1.1678 -0.2166];
% Los bias se fijan a un valor igual a 1. No hay aprendizaje en ellos.
bo = 1;
bh = 1;

J(1)    = 1e3;
m       = 0;
while m < epochs
    m = m + 1;
    for i=1:M
        % voy a tomar uno a uno los puntos para actualizar los pesos
        Xm = X(:,i); 
        tk = T(i);
        % Forward propagation desde la entrada X
        % Calcular primero las aj en las neuronas escondidas
        aj				= Wh * Xm  + bh*ones(Ni,1); %% ESCRIBIR EL CODIGO AQUI - #0: [-0.0530 0.3094]
        [y, dfh]		= activation(aj);
        % Calcular ahjora el valor de salida utilizando lo precedente
        ak				= Wo * y  + bo;  %% ESCRIBIR EL CODIGO AQUI - #0: 1.0133
        [zk, dfo]	= activation(ak);
        % Ya se puede calcular la salida y el error cometido
        % Evaluar ahora el delta_k a la salida: delta_k = (tk-zk)*f'(ak)
        delta_k		= (tk-zk)*dfo; %% ESCRIBIR EL CODIGO AQUI - #0: -0.1238
        %...y delta_j: delta_j = f'(aj)*w_j*delta_k
        delta_j		= dfh' .* (Wo*delta_k); %% ESCRIBIR EL CODIGO AQUI - #0: [0.3550 0.0388]
        % Ahora se actualizan los pesos
        %w_kj <- w_kj + eta*delta_k*y_j
        Wo				= Wo + (eta*delta_k*y)'; %% ESCRIBIR CODIGO AQUI
        %w_ji <- w_ji + eta*delta_j*Xm
        Wh				= Wh + (eta*Xm*delta_j)'; %% ESCRIBIR CODIGO AQUI
    end
    %Calcular el error total
    J(m)    = 0;
    for i = 1:M
        J(m) = J(m) + (T(i) - activation(Wo*activation(Wh*X(:,i) + bh*ones(Ni,1)) + bo)).^2;
    end
    J(m) = J(m)/M; 
    disp(['Iteracion ' num2str(m) ': Error Total ' num2str(J(m))])
end
% El error a la salida debe ser aproximadamente 0.00071965
expErr = 0.00071965;
assert(abs(J(epochs)-expErr) < 1e-6, 'Error de implementacion');

% Resultado
for i = 1:M
    test_res = activation(Wo*activation(Wh* X(:,i) + bh*ones(Ni,1)) + bo);
    disp(['X: [' num2str(X(1,i)) ',' num2str(X(2,i)) '] -> Esperado : ' num2str(T(i)) ' - Calculado: ' num2str(test_res)]);
end