import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io
import os
from collections import OrderedDict

def standardize_sensors(raw_data, channel_config, return_montage=True):

	# channel_names = [x.upper() for x in raw_data.ch_names]
	
	NUM_REDUCED_SENSORS = 8
	montage_sensor_set = ["F7", "F3", "F8", "F4", "T3", "C3", "T4", "C4", "T5", "P3", "T6", "P4", "O1", "O2"]
	first = ["F7", "F8", "T3", "T4", "T5", "T6", "O1", "O2"]
	second = ["F3", "F4", "C3", "C4", "P3", "P4", "P3", "P4"]
	
	if channel_config in ["01_tcp_ar", "03_tcp_ar_a"]:
		montage_sensor_set = [str("EEG "+x+"-REF") for x in montage_sensor_set]
		first = [str("EEG "+x+"-REF") for x in first]
		second = [str("EEG "+x+"-REF") for x in second]

	elif channel_config == "02_tcp_le":
		montage_sensor_set = [str("EEG "+x+"-LE") for x in montage_sensor_set]
		first = [str("EEG "+x+"-LE") for x in first]
		second = [str("EEG "+x+"-LE") for x in second]

	raw_data = raw_data.pick_channels(montage_sensor_set, ordered=True)

	# return channels without subtraction - 14 of them
	if return_montage == False:
		return raw_data, raw_data

	# use a sensor's data to get total number of samples
	reduced_data = np.zeros((NUM_REDUCED_SENSORS, raw_data.n_times))

	# create derived channels
	for idx in range(NUM_REDUCED_SENSORS):
		reduced_data[idx, :] = raw_data[first[idx]][0][:] - raw_data[second[idx]][0][:]
	
	# create new info object for reduced sensors
	reduced_info = mne.create_info(ch_names=[
											"F7-F3", "F8-F4",
											"T3-C3", "T4-C4",
											"T5-P3", "T6-P4",
											"O1-P3", "O2-P4"
											], sfreq=raw_data.info["sfreq"], ch_types=["eeg"]*NUM_REDUCED_SENSORS)
	
	# https://mne.tools/dev/auto_examples/io/plot_objects_from_arrays.html?highlight=rawarray
	reduced_raw_data = mne.io.RawArray(reduced_data, reduced_info)
	# return reduced_raw_data, raw_data
	return reduced_raw_data


def downsample(raw_data, freq=250):
	raw_data = raw_data.resample(sfreq=freq)
	return raw_data, freq


def highpass(raw_data, cutoff=1.0):
	raw_data.filter(l_freq=cutoff, h_freq=None)
	return raw_data


def remove_line_noise(raw_data, ac_freqs = np.arange(50, 101, 50)):
	raw_data.notch_filter(freqs=ac_freqs, picks="eeg", verbose=False)
	return raw_data

# accepts PSD of all sensors, returns band power for all sensors
def get_brain_waves_power(psd_welch, freqs):

	brain_waves = OrderedDict({
		"delta" : [1.0, 4.0],
		"theta": [4.0, 7.5],
		"alpha": [7.5, 13.0],
		"lower_beta": [13.0, 16.0],
		"higher_beta": [16.0, 30.0],
		"gamma": [30.0, 40.0]
	})

	# create new variable you want to "fill": n_brain_wave_bands
	band_powers = np.zeros((psd_welch.shape[0], 6))

	for wave_idx, wave in enumerate(brain_waves.keys()):
		# identify freq indices of the wave band
		if wave_idx == 0:
			band_freqs_idx = np.argwhere((freqs <= brain_waves[wave][1]))
		else:
			band_freqs_idx = np.argwhere((freqs >= brain_waves[wave][0]) & (freqs <= brain_waves[wave][1]))

		# extract the psd values for those freq indices
		band_psd = psd_welch[:, band_freqs_idx.ravel()]

		# sum the band psd data to get total band power
		total_band_power = np.sum(band_psd, axis=1)

		# set power in band for all sensors
		band_powers[:, wave_idx] = total_band_power    

	return band_powers