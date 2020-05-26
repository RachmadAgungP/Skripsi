function [hasilden] = denormalisasi2(data,dataAct)

[b_data k_data]=size(data);
% MaxJST = max(data);
% MinJST = min(data);

[b_dataAct k_dataAct]=size(dataAct);
MaxAct = max(dataAct);
MinAct = min(dataAct);

MinMaxAct = MaxAct-MinAct;

minmaxAct=[];
minAct =[];
for i=1:b_data
    minmaxAct=[minmaxAct;MinMaxAct];
    minAct =[minAct;MinAct];
end

denorm = ceil((data.*minmaxAct)+minAct);
hasilden = denorm;