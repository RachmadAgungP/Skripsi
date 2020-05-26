function [hasil] = protest(data_test,hidden_layer,biasV,biasW)

[b_dtest k_dtest]=size(data_test);
Input = data_test(1:b_dtest,1:(k_dtest-1));

[b_bias_in k_bias_in]=size(biasV);
[b_bias_out k_bias_out]=size(biasW);
%% Inisialisasi Bias
    bias_in_old = zeros(b_bias_in,k_bias_in);
    %%bias_in_new = zeros(b_bias_in,k_bias_in);
    
    bias_out_old = zeros(b_bias_out,k_bias_out);
    %%bias_out_new = zeros(b_bias_out,k_bias_out);
    
    FYNET = [];
for ld = 1 : b_dtest %% +======
        %% Memberikan Nilai Bias yang Baru ke Lama
            bias_in_old = biasV;
            bias_out_old = biasW;
        
        %% Pertama Menghitung Z_Net ===============================================================================
        z_net = zeros(1,hidden_layer); %matriks kosong untuk nilai Z_in (hidden layer pertama)
        
        for x = 1:1
            for y = 1:hidden_layer
                z_net(x,y) = bias_in_old(y,k_bias_in)+ sum(sum((Input(ld,:).*bias_in_old(y,1:(k_bias_in-1)))));
            %%
            end
        end
        
        %% Aktifasi Z_Net =========================================================================================
        fz_net = zeros(1,hidden_layer); %matriks kosong untuk nilai Z_in yang sudah di aktivasi
        
        for x = 1:1
            for y = 1:hidden_layer
                
            %% Loop Aktifasi ZNet
            fz_net(x,y) = (1/(1 +( exp(-z_net(x,y)))));
            %%
            end
        end
        
        %% Y_Net ==============================================================================================
        y_net = zeros(1,1); %matriks kosong untuk nilai Y_in (output)
        fz = fz_net(1,:)';
        %% Mencari Y_Net
        y_net = bias_out_old(b_bias_out,1)+sum(sum((fz.*bias_out_old(1:(b_bias_out-1),1))));
        
        %% Aktifasi Y_Net =======================================================================================
        fy_net = zeros(1,1); %matriks kosong untuk nilai Y_in (output) yang sudah di aktivasi
        fy_net = (1/(1 +( exp(-y_net(1,1)))));
        FYNET = [FYNET;fy_net];
end

hasil=FYNET;