function hasil_in = bias_in(kol_input,hidden_layer)

%% Bobot awal input ke hidden layer (v) kol_input = Data, hidden_layer = banyak Z/ hidden layernya
colm = kol_input+1;
bias_in1 = zeros(hidden_layer,colm);
for i = 1 : hidden_layer
    for j = 1 : colm
        bias_in1(i,j) = -0.5 + (0.5+0.5)*rand(1,1);
    end
end

hasil_in = bias_in1;
% save biasin1 hasil_in;