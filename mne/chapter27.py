import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from scipy.fft import fft, ifft

# Load the MATLAB file
file_path = '../sampleEEGdata.mat'
mat_data = scipy.io.loadmat(file_path)

# Extract EEG data structure
eeg_data = mat_data['EEG']

# Extract relevant fields from the EEG structure
sampling_rate = eeg_data['srate'][0][0][0][0]
n_channels = eeg_data['nbchan'][0][0][0][0]
times = np.squeeze(eeg_data['times'][0][0])  # Correct extraction and squeezing of times array
data = eeg_data['data']
channel_locations = eeg_data['chanlocs'][0][0]
channel_names = [str(channel[0][0]) for channel in channel_locations[0]]

# Convert time windows from ms to indices
timewin1 = [-300, -100]
timewin2 = [200, 400]
timeidx1 = [np.argmin(np.abs(times - t)) for t in timewin1]
timeidx2 = [np.argmin(np.abs(times - t)) for t in timewin2]

# Find indices for the specified sensors
sensor1 = 'POz'
sensor2 = 'Fz'
sensor1_idx = channel_names.index(sensor1)
sensor2_idx = channel_names.index(sensor2)

# Extract data for the sensors
data_actual = eeg_data['data'][0][0]
data_sensor1 = data_actual[sensor1_idx, :, :].T
data_sensor2 = data_actual[sensor2_idx, :, :].T

# Wavelet convolution parameters
time = np.arange(-1, 1, 1/sampling_rate)
half_wavelet_size = (len(time) - 1) // 2
wavelet_cycles = 4.5

def wavelet_convolution(data, freq, srate, cycles):
    n_trials = data.shape[0]
    n_points = data.shape[1]
    wavelet = np.exp(2 * 1j * np.pi * freq * time) * np.exp(-time ** 2 / (2 * (cycles / (2 * np.pi * freq)) ** 2))
    n_wavelet = len(time)
    n_convolution = n_wavelet + n_points - 1
    half_wavelet_size = (n_wavelet - 1) // 2

    # Initialize the output array
    convolved_data = np.zeros((n_trials, n_points))

    # Perform convolution for each trial
    for trial in range(n_trials):
        fft_data = fft(data[trial], n_convolution)
        fft_wavelet = fft(wavelet, n_convolution)
        convolution_result_fft = ifft(fft_wavelet * fft_data, n_convolution) * np.sqrt(wavelet_cycles / (2 * np.pi * freq))
        convolution_result_fft = convolution_result_fft[half_wavelet_size:half_wavelet_size + n_points]
        convolved_data[trial, :] = np.abs(convolution_result_fft) ** 2

    return convolved_data

# Perform wavelet convolution for both sensors
centerfreq1 = 6  # Hz
centerfreq2 = 6  # Hz
analyticsignal1 = wavelet_convolution(data_sensor1, centerfreq1, sampling_rate, wavelet_cycles)
analyticsignal2 = wavelet_convolution(data_sensor2, centerfreq2, sampling_rate, wavelet_cycles)

# Panel A: correlation in a specified window
tfwindowdata1 = analyticsignal1[:, timeidx1[0]:timeidx1[1]].mean(axis=1)
tfwindowdata2 = analyticsignal2[:, timeidx2[0]:timeidx2[1]].mean(axis=1)

# Calculate Pearson and Spearman correlations
pearson_corr = np.corrcoef(tfwindowdata1, tfwindowdata2)[0, 1]
spearman_corr = spearmanr(tfwindowdata1, tfwindowdata2)[0]

# Plotting Panel A
fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=100)

# Pearson correlation plot
axes[0].scatter(tfwindowdata1, tfwindowdata2, color='blue')
axes[0].set_title(f'TF window correlation, Pearson r={pearson_corr:.2f}', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'{sensor1}: {timewin1[0]}-{timewin1[1]} ms; {centerfreq1} Hz', fontsize=12, fontweight='bold')
axes[0].set_ylabel(f'{sensor2}: {timewin2[0]}-{timewin2[1]} ms; {centerfreq2} Hz', fontsize=12, fontweight='bold')
axes[0].axis('square')

# Spearman correlation plot (rank-transformed data)
axes[1].scatter(rankdata(tfwindowdata1), rankdata(tfwindowdata2), color='blue')
axes[1].set_title(f'TF window correlation, Spearman r={spearman_corr:.2f}', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'{sensor1}: {timewin1[0]}-{timewin1[1]} ms; {centerfreq1} Hz', fontsize=12, fontweight='bold')
axes[1].set_ylabel(f'{sensor2}: {timewin2[0]}-{timewin2[1]} ms; {centerfreq2} Hz', fontsize=12, fontweight='bold')
axes[1].axis('square')

plt.tight_layout()
plt.show()

# Panel B: correlation over time
corr_ts = np.array([spearmanr(analyticsignal1[:, ti], analyticsignal2[:, ti])[0] for ti in range(analyticsignal1.shape[1])])

# Plot Panel B
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(times, corr_ts, color='blue')
plt.xlim([-200, 1200])
plt.xlabel('Time (ms)', fontsize=12, fontweight='bold')
plt.ylabel('Spearman\'s rho', fontsize=12, fontweight='bold')
plt.title('Correlation over time', fontsize=14, fontweight='bold')
plt.show()

# Panel C: exploratory time-frequency power correlations
times2save = np.arange(-200, 1201, 25)
frex = np.logspace(np.log10(2), np.log10(40), 20)

# Find indices for times to save
times2save_idx = [np.argmin(np.abs(times - t)) for t in times2save]

# Rank-transform the data
seeddata_rank = rankdata(tfwindowdata2)

# Initialize output correlation matrix
expl_corrs = np.zeros((len(frex), len(times2save)))

# Perform wavelet convolution and correlation calculation
for fi, freq in enumerate(frex):
    analyticsignal1 = wavelet_convolution(data_sensor1, freq, sampling_rate, wavelet_cycles)
    for ti, t_idx in enumerate(times2save_idx):
        expl_corrs[fi, ti] = 1 - 6 * np.sum((seeddata_rank - rankdata(analyticsignal1[:, t_idx])) ** 2) / (data_sensor1.shape[0] * (data_sensor1.shape[0] ** 2 - 1))

# Plot Panel C
fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
contour = ax.contourf(times2save, frex, expl_corrs, 40, cmap='jet')
ax.set_yscale('log')
cbar = fig.colorbar(contour)
cbar.set_ticks([-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45])
cbar.ax.tick_params(labelsize=16)  # Increase colorbar tick label size
contour.set_clim([-0.45, 0.45])
ax.set_yticks([2, 4, 6, 8, 10, 20, 30, 40])
ax.set_yticklabels(['2', '4', '6', '8', '10', '20', '30', '40'], fontsize=16, fontweight='bold')  # Increase y-tick label size
ax.set_xticks(np.arange(times2save.min(), times2save.max(), 200))
ax.set_xticklabels(np.arange(times2save.min(), times2save.max(), 200), fontsize=16, fontweight='bold')  # Increase x-tick label size
ax.set_xlabel('Time (ms)', fontsize=18, fontweight='bold')  # Increase x-label size
ax.set_ylabel('Frequency (Hz)', fontsize=18, fontweight='bold')  # Increase y-label size
ax.set_title(f'Correlation over trials from seed {sensor2}, {centerfreq2} Hz and {timewin2[0]}-{timewin2[1]} ms', fontsize=20, fontweight='bold')  # Increase title size
plt.show()