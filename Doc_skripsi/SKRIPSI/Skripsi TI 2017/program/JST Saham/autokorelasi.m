function [lagsdiambil input] = autokorelasi(Data,batas)
% clear all;
% clc;

% code autokorelasi
% x = xlsread('uji.xls','x');
% x = x(:,1);
x = Data(:,1);

sumX = sum(x);
[rowData colData] = size(x);
mean = sumX / rowData;

yt = x - mean;
ytkuadrat = yt .* yt;
sumytkuadrat = sum(ytkuadrat);

for i=1:rowData
    ytmin(:,i) = [zeros(i,1); yt(1:rowData-i)];
end

for i=1:rowData
    ytytmin(:,i) = yt .* ytmin(:,i);
end

sumytytmin = sum(ytytmin);

lags = [sumytytmin ./ sumytkuadrat];

lags = lags';

%% batas = 0.6

lagsdiambil = [];
for i=1:rowData
    if ((lags(i) >= batas && lags(i) < (batas + 0.1)) == 1)
        lagsdiambil(end+1,:) = [lags(i); i; rowData - i];
    end
end

% lagsdiambil
[bar kol] = size(lagsdiambil);
minimum = min(lagsdiambil(:,3));

input = [];
for j = 1: bar
    nilai = zeros(1,minimum);
    dat = (lagsdiambil(j,3)) + 1;
    for k = 1:minimum
        dat = dat-1;
        nilai(1,k) = x(dat,1);
    end
    input = [input;nilai];
end

nilaiTar = zeros(1,minimum);
    datTar = rowData + 1;
    for k = 1:minimum
        datTar = datTar-1;
        nilaiTar(1,k) = x(datTar,1);
    end
    
    input = [input;nilaiTar];

input = input';
hasilAutokorelasi = lags;
save hasilAutokorelasi hasilAutokorelasi;