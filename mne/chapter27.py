import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.io import loadmat

# Load sample EEG dataset
data = loadmat('../sampleEEGdata.mat')

# Extract the EEG data and channel information
eeg_data = data['EEG'].data
chan_info = data['EEG'].chanlocs

# Create an MNE Raw object from the data
info = mne.create_info(chan_info, 1000, 'eeg')
eeg_data = mne.io.RawArray(eeg_data, info)

# Define sensors and time windows
sensor1 = 'Pz'
sensor2 = 'Fz'
timewin1 = [-300, -100]  # in ms relative to stim onset
timewin2 = [200, 400]

# Convert time from ms to index
time_idx1 = np.argmin(np.abs(eeg_data.timEs - timewin1[0]))
time_idx2 = np.argmin(np.abs(eeg_data.times - timewin2[0]))

# Define wavelet parameters
centerfreq1 = 6  # in Hz
centerfreq2 = 6
wavelet_cycles = 4.5

# Create wavelet and run convolution
def create_wavelet(time, freq, cycles):
    return np.exp(2 * 1j * np.pi * freq * time) * np.exp(-time**2 / (2 * (cycles / (2 * np.pi * freq))**2))

def convolve_data(data, wavelet):
    return convolve(data, wavelet, mode='same')

# Panel A: correlation in a specified window
analyticsignal1 = np.abs(convolve_data(eeg_data.get_data(sensor1), create_wavelet(eeg_data.times, centerfreq1, wavelet_cycles)))**2
analyticsignal2 = np.abs(convolve_data(eeg_data.get_data(sensor2), create_wavelet(eeg_data.times, centerfreq2, wavelet_cycles)))**2

tfwindowdata1 = np.mean(analyticsignal1[:, time_idx1:time_idx2], axis=1)
tfwindowdata2 = np.mean(analyticsignal2[:, time_idx1:time_idx2], axis=1)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(tfwindowdata1, tfwindowdata2, '.')
plt.axis('square')
plt.title(f'TF window correlation, r_p={np.corrcoef(tfwindowdata1, tfwindowdata2)[0, 1]}')
plt.xlabel(f'{sensor1}: {timewin1[0]}-{timewin1[1]} ms; {centerfreq1} Hz')
plt.ylabel(f'{sensor2}: {timewin2[0]}-{timewin2[1]} ms; {centerfreq2} Hz')

# Also plot rank-transformed data
plt.subplot(122)
plt.plot(np.argsort(tfwindowdata1), np.argsort(tfwindowdata2), '.')
plt.axis('square')
plt.xlabel(f'{sensor1}: {timewin1[0]}-{timewin1[1]} ms; {centerfreq1} Hz')
plt.ylabel(f'{sensor2}: {timewin2[0]}-{timewin2[1]} ms; {centerfreq2} Hz')
plt.title(f'TF window correlation, r_p={np.corrcoef(tfwindowdata1, tfwindowdata2)[0, 1]}')

# Panel B: correlation over time
corr_ts = np.zeros((eeg_data.times.shape[0],))
for ti in range(eeg_data.times.shape[0]):
    corr_ts[ti] = np.corrcoef(analyticsignal1[:, ti], analyticsignal2[:, ti])[0, 1]

plt.figure(figsize=(10, 4))
plt.plot(eeg_data.times, corr_ts)
plt.xlim([-200, 1200])
plt.xlabel('Time (ms)')
plt.ylabel("Spearman's rho")

# Panel C: exploratory time-frequency power correlations
times2save = np.arange(-200, 1200, 25)
frex = np.logspace(np.log10(2), np.log10(40), 20)

times2save_idx = np.argmin(np.abs(eeg_data.times[:, None] - times2save[None, :]), axis=0)

expl_corrs = np.zeros((len(frex), len(times2save)))

for fi in range(len(frex)):
    analyticsignal1 = np.abs(convolve_data(eeg_data.get_data(sensor1), create_wavelet(eeg_data.times, frex[fi], wavelet_cycles)))**2
    for ti in range(len(times2save)):
        expl_corrs[fi, ti] = 1 - 6 * np.sum((np.argsort(tfwindowdata2) - np.argsort(analyticsignal1[:, times2save_idx[ti]]))**2) / (eeg_data.trials * (eeg_data.trials**2 - 1))

plt.figure(figsize=(10, 4))
plt.contourf(times2save, frex, expl_corrs, 40, cmap='RdBu')
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title(f'Correlation over trials from seed {sensor2}, {centerfreq2} Hz and {timewin2[0]}-{timewin2[1]} ms')