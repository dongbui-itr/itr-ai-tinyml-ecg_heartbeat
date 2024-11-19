import numpy as np
import matplotlib.pyplot as plt

DATA_TYPE = 'float64'

PAC = np.asarray(['A', 'j', 'J', 'e', 'S'])
PVC = np.asarray(['V', 'a', 'E', '!'])
Other = np.asarray(['F', '|', 'Q'])


def good_types():
   """ Of annotations """
   # www.physionet.org/physiobank/annotations.shtml
   # good = ['N', 'L', 'R', 'B', 'S', 'S', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
   good = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
   return good


def beat_annotations(annotation):
    """ Get rid of non-beat markers """
    # Declare beat types
    good = good_types()
    ids = np.in1d(annotation.symbol, good)
    # We want to know only the positions
    beats = annotation.sample[ids]
    beats_symbol = np.asarray(annotation.symbol)[ids]
    s_index = np.asarray([i for i in range(len(beats_symbol)) if beats_symbol[i] in PAC])
    v_index = np.asarray([i for i in range(len(beats_symbol)) if beats_symbol[i] in PVC])
    if len(s_index) > 0:
        beats_symbol[s_index] = 'S'
    if len(v_index) > 0:
        beats_symbol[v_index] = 'V'

    return beats, beats_symbol


def beat_width(signal, sample, bef_ind, aft_ind, debug=False):
    sample_amp = signal[sample]
    tmp_bef_ind = sample - bef_ind
    tmp_aft_ind = sample + aft_ind
    if tmp_bef_ind < 0:
        tmp_bef_ind = 0

    if tmp_aft_ind > len(signal) - 1:
        tmp_aft_ind = len(signal)

    def argmin_position(signal, start, stop, amplitude, ratio, bef_flag=False):
        tmp_signal = np.abs(signal[start:stop]) - np.abs(ratio * amplitude)
        indx_segment = np.flatnonzero(np.abs(np.diff(np.sign(tmp_signal))) == 2) + 1
        if len(indx_segment) == 0:
            if bef_flag:
                return start - np.argmin(tmp_signal)
            else:
                return start + np.argmin(tmp_signal)
        else:
            segments = np.split(tmp_signal, indx_segment)
            if bef_flag:
                segment = segments[-1]
                return start - np.argmin(segment) + indx_segment[-1]
            else:
                segment = segments[0]
                return start + np.argmin(segment)

    ind_bef_20 = argmin_position(signal, tmp_bef_ind, sample, sample_amp, 0.8, True)
    ind_aft_20 = argmin_position(signal, sample, tmp_aft_ind, sample_amp, 0.8, False)

    ind_bef_50 = argmin_position(signal, tmp_bef_ind, sample, sample_amp, 0.5, True)
    ind_aft_50 = argmin_position(signal, sample, tmp_aft_ind, sample_amp, 0.5, False)

    ind_bef_80 = argmin_position(signal, tmp_bef_ind, sample, sample_amp, 0.2, True)
    ind_aft_80 = argmin_position(signal, sample, tmp_aft_ind, sample_amp, 0.2, False)

    sample_start = sample - 3 * bef_ind // 2
    sample_stop = sample + 3 * aft_ind // 2
    if sample_start < 0:
        sample_start = 0
    if sample_stop > len(signal):
        sample_stop = len(signal)

    if debug and sample - 200 > 0:
        plt.plot(signal[sample - 200: sample + 200])
        plt.plot(200, signal[sample - 200: sample + 200][200], 'yo')
        plt.plot(ind_bef_20 - sample + 200, signal[sample - 200: sample + 200][ind_bef_20 - sample + 200], 'ro')
        plt.plot(ind_aft_20 - sample + 200, signal[sample - 200: sample + 200][ind_aft_20 - sample + 200], 'ro')
        plt.plot(ind_bef_50 - sample + 200, signal[sample - 200: sample + 200][ind_bef_50 - sample + 200], 'k*')
        plt.plot(ind_aft_50 - sample + 200, signal[sample - 200: sample + 200][ind_aft_50 - sample + 200], 'k*')
        plt.plot(ind_bef_80 - sample + 200, signal[sample - 200: sample + 200][ind_bef_80 - sample + 200], 'g+')
        plt.plot(ind_aft_80 - sample + 200, signal[sample - 200: sample + 200][ind_aft_80 - sample + 200], 'g+')
        plt.show()
        # plt.close()
    width_20 = ind_aft_20 - ind_bef_20
    width_50 = ind_aft_50 - ind_bef_50
    width_80 = ind_aft_80 - ind_bef_80

    ## adjust width ##
    if width_50 > width_80:
        width_50 = width_80 - 3 if width_80 - 3 > 0 else width_80 // 2
    if width_20 > width_50:
        width_20 = width_50 - 3 if width_50 - 3 > 0 else width_50 // 2

    ##################

    return ind_bef_20, ind_aft_20, ind_bef_50, ind_aft_50, ind_bef_80, ind_aft_80, \
        width_20, width_50, width_80, \
        np.max(signal[sample_start:sample_stop]), np.min(signal[sample_start:sample_stop]), \
        np.max(signal[sample_start:sample_stop]) - np.min(signal[sample_start:sample_stop])


def calibrate_position_beat(signal, sample, bef_ind, aft_ind):
    if sample - bef_ind < 0:
        if (np.max(signal[0: sample + aft_ind]) > 0 > np.min(signal[0: sample + aft_ind]) and
                np.abs(np.max(signal[0: sample + aft_ind]) / np.min(signal[0: sample + aft_ind])) > 0.65):
            return np.argmax(signal[0: sample + aft_ind])
        else:
            return np.argmax(np.abs(signal[0: sample + aft_ind]))

    elif sample + aft_ind > len(signal) - 1:
        if (np.max(signal[sample - bef_ind:]) > 0 > np.min(signal[sample - bef_ind:]) and
                np.abs(np.max(signal[sample - bef_ind: sample + aft_ind]) / np.min(
                    signal[sample - bef_ind: sample + aft_ind])) > 0.65):
            return sample - bef_ind + np.argmax(signal[sample - bef_ind:]) - 1
        else:
            return sample - bef_ind + np.argmax(np.abs(signal[sample - bef_ind:]))

    else:
        if (np.max(signal[sample - bef_ind: sample + aft_ind]) > 0 > np.min(
                signal[sample - bef_ind: sample + aft_ind]) and
                np.abs(np.max(signal[sample - bef_ind: sample + aft_ind]) / np.min(signal[sample - bef_ind: sample + aft_ind])) > 0.65):
            return sample - bef_ind + np.argmax(signal[sample - bef_ind: sample + aft_ind])
        else:
            return sample - bef_ind + np.argmax(np.abs(signal[sample - bef_ind: sample + aft_ind]))
