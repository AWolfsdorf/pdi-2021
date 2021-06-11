function [f, df] = activation(x)

b = 2.5;
f	= tanh(b*x);
df	= b*sech(b*x).^2;