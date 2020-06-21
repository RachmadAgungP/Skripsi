function hasil = HitungMape(hasilDenor,Target)

[bar kol]=size(hasilDenor);

result=0;
for i = 1:bar
    for j = 1:kol
        result = result + abs((Target(i,j)-hasilDenor(i,j))/Target(i,j));
    end
end

hasil = (1/bar)*result*100;