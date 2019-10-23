% by H. Niu, 2017
% read in SwellEx96 data, process for FNN inputs, save as text file

SamplingFreq=3276.8; 
load('data/positions_hlanorth.txt');
ArrayShape(:,1) = positions_hlanorth(:,3); % x-axis
ArrayShape(:,2) = positions_hlanorth(:,2); % y-axis

ChannelNum = 27;  
ind = 1:ChannelNum;
Text_tau1 =  1.25; % second              
TextNumber = 10;%8*2;        

datapath = 'data/';
File=dir([datapath '*.north.sio']);

SigDuration=Text_tau1;
SigLength=SigDuration*SamplingFreq;
MaxNsnapNum=65*60/Text_tau1;

f_pro = 79; % frequency
N_FFT = SamplingFreq*Text_tau1;
df=SamplingFreq/N_FFT;

Freq=(0:N_FFT-1)*df;
index_f_pro = floor(f_pro/df)+1;  
id = 0; ii=1; Mid =0;
fname = [datapath File.name];
disp(fname)
datain = sioread(fname,1,-1,ind);
tick=0;
for isnap=1:MaxNsnapNum
%     id = id + 1;
    Mid=Mid+1;
    id = mod(Mid-1,TextNumber)+1;
    signal = datain((isnap-1)*Text_tau1*SamplingFreq+1:isnap*Text_tau1*SamplingFreq,:);
    Y=fft( signal,SamplingFreq*Text_tau1);
    xx = Y(index_f_pro,:).' / norm( Y(index_f_pro,:));
    A(:,id,ii) = xx;
    ss(:,:,id) = xx * xx';
    if Mid>=TextNumber
        tmp2 = squeeze(mean(ss,3));
        tmp2 = triu(tmp2);
       % tmp3 = reshape(tmp2,size(tmp2,1)*size(tmp2,2),1);
        tmp3 = tmp2(tmp2~=0);
        x_test(ii,:) = [real(tmp3);imag(tmp3)];     
        ii = ii + 1;
%         id =0;
    end

end

save(['data/Experiment_test_data_' num2str(TextNumber) 'snap.txt'],'-ascii','x_test');
save(['data/Experiment_test_arraydata_' num2str(TextNumber) 'snap'],'A');
