
clc; 
clear; 
close all;

% /media/mihir/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/female/control/training_feat/mcc

SRC={'M12'};
TGT={'New_FC_Feat'};

for sr=1:length(SRC)
    for tg=1:length(TGT)
        
        fprintf(SRC{sr});
        fprintf('\n');
         
        noisy_path = ['/media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/',SRC{sr},'/dysarthric/testing_feat/mcc/'];
         
        files_noisy = dir([noisy_path,'/*.mcc']);
        %filelist = dir(['/media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/features/',SRC{sr},'/dysarthric/testing_data/mcc/.*'])
        %files_noisy = natsortfiles({files_noisy.name});
         
        save_path = ['/media/maitreya/Dysarthia/dysarthic_interspeech/UA/speaker_specific/batches/',SRC{sr},'/dysarthric/testing_batches/'];
        k=0;

        for i=1:length(files_noisy)
            disp(['Processing file : ', num2str(i)])
            noisy_file = [noisy_path,files_noisy(i).name];
            %[clean,fs] = audioread(clean_file);
            fid2=fopen(noisy_file);
            Noisy_gtm=fread(fid2,Inf,'float');
            Log_Noisy_gtm=reshape(Noisy_gtm,40,length(Noisy_gtm)/40); 
            fclose('all');

            % input and output features concatenated
            IP_feats = Log_Noisy_gtm';
            
            Feat = IP_feats;

            save([save_path,'Test_Batch_',num2str(i-1)],'Feat','-v6');
            k=k +1;
        end   

    end
end




