%% Figure 2.3
chan2plot = 'pz'; % you can pick any electrode (type {EEG.chanlocs.labels} for all electrodes)
% compute ERP (time-domain trial average from selected electrode)
erp = squeeze(mean(EEG.data(strcmpi(chan2plot,{EEG.chanlocs.labels}),:,:),3));

% low-pass filter data
nyquist       = EEG.srate/2;
filter_cutoff = 40;  % Hz
trans_width   = 0.1; % transition width, in fraction of 1
ffrequencies  = [ 0 filter_cutoff filter_cutoff*(1+trans_width) nyquist ]/nyquist;
idealresponse = [ 1 1 0 0 ];
filterweights = firls(100,ffrequencies,idealresponse);
filtered_erp  = filtfilt(filterweights,1,double(erp));

figure
% plot ERP with thicker line
plot(EEG.times,filtered_erp,'k-','LineWidth',2)

set(gca,'xlim',[-200 1000],'ydir','r')
set(gca,'FontSize',12)  % 軸の数字のフォントサイズを大きく
xlabel('Time (ms)','FontSize',14)  % x軸ラベルを大きく
ylabel('Voltage (\muV)','FontSize',14)  % y軸ラベルを大きく
title([ 'ERP from electrode ' chan2plot ],'FontSize',16)  % タイトルをさらに大きく
%%