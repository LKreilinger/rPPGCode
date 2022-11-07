import numpy as np

def get_rfft_pulse(signal,fps):
    signal_size = len(signal)
    signal = signal.flatten()
    fft_data = np.fft.rfft(signal)  # FFT
    fft_data = np.abs(fft_data)
    minFreq = 0.67  #
    maxFreq = 3  #
    fps = float(fps)
    fft_spec = []
    freq = np.fft.rfftfreq(signal_size, 1./fps)  # Frequency data

    inds = np.where((freq < minFreq) | (freq > maxFreq))[0]
    fft_data[inds] = 0
    bps_freq = 60.0*freq
    max_index = np.argmax(fft_data)
    fft_data[max_index] = fft_data[max_index]**2
    fft_spec.append(fft_data)
    pulse = bps_freq[max_index]
    return pulse