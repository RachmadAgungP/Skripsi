function [hasil] = crossVal(Data)

[b_data k_data] = size(Data);

y=3;
k=[];
k2=1;
ban=0;
for i = 1:3
    bani=ban;
    ban = ceil(b_data/y);
    b_data=b_data-ban;
    y=y-1;
    if(i==1)
        k=[k;1 ban];
    else
        k2=k2+bani;
        k=[k;k2 ((k2-1)+ban)];
    end
end

hasil=k;