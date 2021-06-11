function [Base,Labels] = loadLPRImages(imageDim)

disp('Lectura de la base de caracteres');

folder = fullfile(fullfile('svm_smo','SVMCode','Datasets','BaseOCR_MultiStyle'));

dchars = {
    '0';
    '1';
    '2';
    '3';
    '4';
    '5';
    '6';
    '7';
    '8';
    '9'}';


Base = [];
Labels = [];

for d=1:length(dchars)
    % get file list
    files = dir(fullfile(folder,dchars{d},'*.bmp'));
    files = [files ; dir(fullfile(folder,dchars{d},'*.tif'))];
    for f=1:length(files)
        filename = fullfile(folder,dchars{d},files(f).name);
        Ip = double(imread(filename));
        % redimensiono el caracter
        Ip = imresize(Ip,[imageDim imageDim]);
        % normalizo entre 0 y 1
        Ip = Ip - min(Ip(:));
        Ip = Ip / max(Ip(:));
        
        Base = [Base Ip(:)];      
        Labels = [Labels d-1];
    end
end

