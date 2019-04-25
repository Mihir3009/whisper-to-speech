
clc; 
clear all; 
close all;

% Use for creating cepstral features for all the .wav file
SRC={'female_feat'};

% path for mcc files
filelist=dir(['/media/mihir/Dysarthia/dysarthic_interspeech/TORGO/new_data/',SRC{1},'/mcc/*.mcc']);

dim=40;result=[];mpsrc=[];mptgt=[];t_scores=[];l=[];lz=[];wr=[];mpsrc1=[];mptgt1=[];Z1=[];Z=[];Z2=[];
x=[];y=[];X=[];Y=[];path=[];

for index=1:length(filelist)
    
    fprintf('Processing %s\n',filelist(index).name);
    
% same path for mcc file
	
    fid=fopen(['/media/mihir/Dysarthia/dysarthic_interspeech/TORGO/new_data/',SRC{1},'/mcc/',filelist(index).name]);
    x=fread(fid,Inf,'float');
    x=reshape(x,dim,length(x)/dim);
    
    Z=[Z x]; 
    
end

save(['/media/mihir/Dysarthia/dysarthic_interspeech/TORGO/data/training_data/female/src/Z.mat'],'Z');  



