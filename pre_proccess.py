import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from obspy import read
from scipy import signal
import cupy as cp
import os
import calc
import vis
import matplotlib.pyplot as plt

class SeismicPreProcessor:
    def __init__(self, sr_factor=8):
        self.sr_factor = sr_factor

    def sta_lta(self, data, sr):
        """
        Calculate the STA/LTA ratio using CuPy for GPU acceleration.
        """
        data_gpu = cp.asarray(data)
        sta_length = self.sr_factor * int(sr)
        lta_length = 2 * self.sr_factor * int(sr)

        sta_coeffs = cp.ones(sta_length) / sta_length
        sta_gpu = signal.lfilter(sta_coeffs.get(), 1, (data_gpu**2).get())

        lta_coeffs = cp.ones(lta_length) / lta_length
        lta_gpu = signal.lfilter(lta_coeffs.get(), 1, (data_gpu**2).get())

        lta_gpu[lta_gpu == 0] = 1e-10
        sta_lta_ratio_gpu = cp.asarray(sta_gpu) / cp.asarray(lta_gpu)

        return cp.asnumpy(sta_lta_ratio_gpu)

    def binarize(self, data, sta_lta_ratio):
        """
        Binarize the data where STA/LTA ratio is zero.
        """
        data_copy = np.copy(data)
        x = np.where(sta_lta_ratio == 0)
        for i in x[0]:
            data_copy[i] /= 10
        return data_copy

    def binarize_win(self, data, sta_lta_ratio):
        """
        Binarize the data in windows where STA/LTA ratio is zero.
        """
        data_copy = np.copy(data)
        x = np.where(sta_lta_ratio == 0)
        prev = 0
        for i in x[0]:
            if i - prev < 6 * self.sr_factor:
                continue
            data_copy[i:i + 6 * self.sr_factor] /= 10
            prev = i
        return data_copy

    def pre_process_signal(self, y):
        """
        Preprocess the signal with mean subtraction, scaling, and filtering.
        """
        # y = y - np.mean(y)
        # S = StandardScaler()
        # y = S.fit_transform(y.reshape(-1, 1)).flatten()
        # M = MinMaxScaler()
        # y = M.fit_transform(y.reshape(-1, 1)).flatten()

        f, t, sxx = signal.spectrogram(y)
        mean_freq = calc.calculate_mean_frequency(f, sxx)
        print("Mean Frequency:", mean_freq)

        high_cutoff = 12 * mean_freq
        low_cutoff = 4 * mean_freq
        sos = signal.butter(4, [low_cutoff, high_cutoff], btype='band', fs=6.625, output='sos')
        y = signal.sosfilt(sos, y)
        y[:100] = 0

        return y

    def process_file(self, file_name, arrival_time, file_path):
        """
        Process each seismic file, apply pre-processing, STA/LTA, and binarization.
        """
        dat = read(file_path).traces[0].copy()
        y = dat.data
        x = dat.times()
        sr = dat.stats.sampling_rate

        # Pre-process the signal
        y_filtered = self.pre_process_signal(y)

        # Calculate STA/LTA ratio and apply threshold
        sta_lta_ratio = self.sta_lta(y_filtered, sr)
        sta_lta_ratio[sta_lta_ratio >= 2 * 0.7] = 0

        # Binarize the signal based on STA/LTA ratio
        y_binarized = self.binarize_win(y_filtered, sta_lta_ratio)

        # Save the processed data to CSV
        processed_data = pd.DataFrame({"time": x, "velocity": y_binarized})
        processed_data.to_csv(f"processed_data/processed_{file_name}.csv")

        # Plot the result
        plt.plot(x, y, label="Original Signal")
        plt.plot(x, y_binarized, label="Processed Signal")
        vis.mark_detection(arrival_time)
        plt.legend()
        plt.show()

    def run_pipeline(self, catalog_path, data_folder):
        """
        Run the entire processing pipeline for all files in the catalog.
        """
        cat = pd.read_csv(catalog_path)
        files = os.listdir(data_folder)

        for i in range(len(files)):
            file_name = cat.iloc[i]['filename']
            arrival_time = cat.iloc[i]['time_rel(sec)']
            file_path = os.path.join(data_folder, f"{file_name}.mseed")
            print(f"Processing: {file_name}")
            self.process_file(file_name, arrival_time, file_path)

# Example usage
if __name__ == "__main__":
    preprocessor = SeismicPreProcessor(sr_factor=8)
    preprocessor.run_pipeline(r'data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv',
                              r'data/lunar/training/data/S12_GradeA')
