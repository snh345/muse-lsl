# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it compares the alpha waves from the two threads (which are running the same stream).

Adapted from https://github.com/NeuroTechX/bci-workshop
"""

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import utils
import threading

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

""" EXPERIMENTAL PARAMETERS """
BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
INDEX_CHANNEL = 0  # Single channel

def stream_data(stream_address, index_channel):
    inlet = StreamInlet(stream_address, max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    info = inlet.info()
    fs = int(info.nominal_srate())

    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None

    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))
    band_buffer = np.zeros((n_win_test, 4))

    last_100_samples = []

    try:
        while True:
            eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
            ch_data = np.array(eeg_data)[:, index_channel]
            ch_data = ch_data.reshape(-1, 1)  # Reshape to 2D array
            eeg_buffer, filter_state = utils.update_buffer(eeg_buffer, ch_data, notch=True, filter_state=filter_state)

            data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)
            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))
            smooth_band_powers = np.mean(band_buffer, axis=0)

            alpha_metric = smooth_band_powers[Band.Alpha]
            print(f'Alpha Relaxation for thread: ', alpha_metric)

            # Store the last 100 samples
            last_100_samples = ch_data.flatten().tolist()[-100:]

    except KeyboardInterrupt:
        print('Closing!')

    return last_100_samples

def calculate_similarity(list1, list2):
    assert len(list1) == len(list2), "Lists must have the same length"
    similarity_sum = 0
    for i in range(len(list1)):
        similarity_sum += 1 - abs(list1[i] - list2[i]) / max(abs(list1[i]), abs(list2[i]), 1e-8)
    similarity_value = similarity_sum / len(list1)
    return similarity_value

if __name__ == "__main__":
    """ 1. CONNECT TO EEG STREAM """
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find an EEG stream.')

    stream_address = streams[0]

    """ 2. START THREADS FOR THE SAME STREAM """
    thread1 = threading.Thread(target=stream_data, args=(stream_address, INDEX_CHANNEL))
    thread2 = threading.Thread(target=stream_data, args=(stream_address, INDEX_CHANNEL))

    thread1.start()
    thread2.start()

    """ 3. COMPARE ALPHA WAVES """
    last_100_samples_thread1 = thread1.join()
    last_100_samples_thread2 = thread2.join()

    if last_100_samples_thread1 is not None and last_100_samples_thread2 is not None:
        similarity_value = calculate_similarity(last_100_samples_thread1, last_100_samples_thread2)
        print(f'Similarity value: {similarity_value}')
    else:
        print("Error: One or both threads returned None.")

    print('Both threads have ended')