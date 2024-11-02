

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

%% Plot single trial data from PZ electrode - Raw and Filtered
% Find PZ electrode index
pz_idx = find(strcmpi('pz', {EEG.chanlocs.labels}));

% Get first trial data from PZ
trial_num = 1;
pz_data = squeeze(EEG.data(pz_idx, :, trial_num));

% Plot raw data
figure
plot(EEG.times, pz_data, 'b-', 'LineWidth', 2)
set(gca, 'xlim', [-200 1000])
set(gca, 'FontSize', 12)
xlabel('Time (ms)', 'FontSize', 14)
ylabel('Voltage (\muV)', 'FontSize', 14)
title(sprintf('Raw EEG from electrode PZ - Trial %d', trial_num), 'FontSize', 16)
grid on

% Apply 40Hz low-pass filter
nyquist = EEG.srate/2;
filter_cutoff = 40; % Hz
trans_width = 0.1; % transition width, in fraction of 1
ffrequencies = [0 filter_cutoff filter_cutoff*(1+trans_width) nyquist]/nyquist;
idealresponse = [1 1 0 0];
filterweights = firls(100, ffrequencies, idealresponse);
filtered_pz_data = filtfilt(filterweights, 1, double(pz_data));

% Plot filtered data
figure
plot(EEG.times, filtered_pz_data, 'r-', 'LineWidth', 2)
set(gca, 'xlim', [-200 1000])
set(gca, 'FontSize', 12)
xlabel('Time (ms)', 'FontSize', 14)
ylabel('Voltage (\muV)', 'FontSize', 14)
title(sprintf('40Hz Low-pass Filtered EEG from electrode PZ - Trial %d', trial_num), 'FontSize', 16)
grid on

% データの最初の数値を表示して確認
fprintf('First 5 timepoints of Raw PZ data (Trial 1):\n')
disp(pz_data(1:5))
fprintf('\nFirst 5 timepoints of Filtered PZ data (Trial 1):\n')
disp(filtered_pz_data(1:5))