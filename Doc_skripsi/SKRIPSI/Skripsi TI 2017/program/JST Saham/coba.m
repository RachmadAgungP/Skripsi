
for i=1:10
    for j=1:10
        [hasil_in] = bias_in(j,i);%% kol_input ==Lag
        vBobot{i,j}=hasil_in;
    end
        [hasil_out] = bias_out(1,i);%%
        wBobot{i,1}=hasil_out;
end

save BobotV/vBobot vBobot;
save BobotW/wBobot wBobot;