function [bobotW bobotV MSE niloop] = Backpro(InputData,hasil_in,hasil_out,hidden_layer,alfa,maxepoh,momentum,target_error)
%% learning rate 0 s/d 1
%%rate = 0.1;
% alfa=0.08;
% maxepoh = 1000;
% momentum = 0.02;
% target_error = 0.001;

%%hidden_layer = 1;
[bar_input kol_input] = size(InputData);

%% Input Data X1, X2, dst
Input = InputData(1:bar_input,1:(kol_input-1)); %% Ambil Data 1-4 dulu
[b_input k_input] = size(Input);

%% Target Data
Target = InputData(1:bar_input,kol_input); %% Ambil Data 1-4 dulu
[b_target k_target] = size(Target);

%% hasil Random Bias
% % [hasil_bias_in] = bias_in(k_input,hidden_layer);
% % [hasil_bias_out] = bias_out(k_target,hidden_layer);
%% bar dan kolom nilai Bias
[b_bias_in,k_bias_in]=size(hasil_in);
[b_bias_out,k_bias_out]=size(hasil_out);

%% Inisialisasi Bias
    bias_in_old = zeros(b_bias_in,k_bias_in);
    bias_in_new = zeros(b_bias_in,k_bias_in);
    
    bias_out_old = zeros(b_bias_out,k_bias_out);
    bias_out_new = zeros(b_bias_out,k_bias_out);
%% --------------------------Loop Backpro--------------------------------
niloop = 0;
nilaiMSE =[];
for loop = 1:maxepoh %% ===============Loop
mse = [];
    %% --------------------------Loop Data--------------------------------
    for ld = 1 : b_input %% +======
        %% Memberikan Nilai Bias yang Baru ke Lama
        if(loop == 1 && ld ==1)
            bias_in_old = hasil_in;
            bias_out_old = hasil_out;
        else
            bias_in_old = bias_in_new;
            bias_out_old = bias_out_new;
        end
        
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
        %% y_net = zeros(1,1); %matriks kosong untuk nilai Y_in (output)
        fz = fz_net(1,:)';
        %% Mencari Y_Net
        y_net = bias_out_old(b_bias_out,1)+sum(sum((fz.*bias_out_old(1:(b_bias_out-1),1))));
        
        %% Aktifasi Y_Net =======================================================================================
        fy_net = zeros(1,1); %matriks kosong untuk nilai Y_in (output) yang sudah di aktivasi
        fy_net = (1/(1 +( exp(-y_net(1,1)))));
        
        mse = [mse;fy_net];
        %% Mencari Delta Langkah 6 (Bias W baru)===============================================================================
        delta_k = (Target(ld,1)-fy_net)* fy_net *(1-fy_net); %% Rumus == (Targer-Fy_Net) * Fy_Net * (1 - Fy_Net)
        deltaw0 = alfa*delta_k*momentum;
        
        for i=1:hidden_layer
            deltaW(1,i) = alfa*delta_k*momentum*fz_net(1,i);
        end
        
        %% Mencari Delta Net Langkah 7 (Bias V baru) ===============================================================================
        for i=1:hidden_layer
            Anet(1,i) = delta_k*bias_out_old(i,1);
        end
        
        for i=1:hidden_layer
            Bnet(1,i) = Anet(1,i)*fz_net(1,i)*(1 - fz_net(1,i));
        end
        
        for i=1:hidden_layer
            deltaV0(1,i) = alfa*Bnet(1,i)*momentum;
        end
        
        %% Tergantung Banyak Input (Lags)
         for x = 1:k_input
            for y = 1:hidden_layer
                deltaV(x,y) = alfa*Bnet(1,y)*momentum*Input(ld,x);
            end
         end
         
         %% Langkah 8 Bobot Baru W
         help = 1;
         for i = 1:(b_bias_out-1)
            help=help+1;
            bias_out_new(i,1)=bias_out_old(i,1)+deltaW(1,i);
         end
         bias_out_new(help,1)= bias_out_old(help,1)+deltaw0;
         
         %% Langkah 8 Bobot Baru V
         for x = 1:hidden_layer
            for y = 1:(k_bias_in-1)
                bias_in_new(x,y)= bias_in_old(x,y)+deltaV(y,x);
            end
         end
         
         for i = 1:hidden_layer
            bias_in_new(i,k_bias_in)=bias_in_old(i,k_bias_in)+deltaV0(1,i);
         end
         
    end
    
    %% Menghitung Error MSE
    nilai_error = 0;
    
    nilai_mse0 = Target-mse;
    nilai_mse1 = sum(sum(nilai_mse0.*nilai_mse0));
    nilai_error = (nilai_mse1 /(bar_input));
    niloop = niloop + 1;
    nilaiMSE =[nilaiMSE;nilai_error];
    if (nilai_error <= target_error) %% http://javaneural.blogspot.com/2009/11/algoritma-pembelajaran-backpropagation.html
        break;
    end
    
    
end

bobotV = bias_in_new;
bobotW = bias_out_new;
MSE = nilaiMSE;

save bobotV bobotV;
save bobotW bobotW;
