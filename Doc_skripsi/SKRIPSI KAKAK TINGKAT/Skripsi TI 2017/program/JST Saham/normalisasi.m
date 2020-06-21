function [hasil Maks Mini] = normalisasi(input)

[bar kol] = size(input);
nilaiMin = min(input);
nilaiMax = max(input);
NormMinMax = nilaiMax-nilaiMin;

hasilNormalisasi = zeros(bar,kol);
for x = 1:bar
    for y = 1:kol
        hasilNormalisasi(x,y) = (input(x,y)-nilaiMin(1,y))/NormMinMax(1,y);
    end
end

Maks = nilaiMax(1,kol);
Mini = nilaiMin(1,kol);
hasil = hasilNormalisasi;
save hasilNormalisasi hasilNormalisasi;