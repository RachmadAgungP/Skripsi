function hasil_out = bias_out(kol_input,bias)

%% Bobot awal ke Output (W) (Bias diubah-ubah) kol_input = 1
bias1 = bias+1;
bias_out1 = zeros(bias1,kol_input);
for i = 1 : bias1
    for j = 1 : kol_input
        bias_out1(i,j) = -0.5 + (0.5+0.5)*rand(1,1);
    end
end

hasil_out = bias_out1;

% save biasout1 hasil_out;