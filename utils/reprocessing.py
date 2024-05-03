from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, iirnotch, sosfiltfilt, iirfilter
from scipy.ndimage.filters import maximum_filter1d
import scipy.signal as signal
import matplotlib.pyplot as plt

import warnings
import math

import numpy as np
from scipy.fftpack import hilbert
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,
                          remez)

from scipy.signal import sosfilt
from scipy.signal import zpk2sos
from utils import define
from scipy.signal import find_peaks


def beat_annotations(annotation):
    """ Get rid of non-beat markers """
    # N		Normal beat (displayed as "·" by the PhysioBank ATM, LightWAVE, pschart, and psfd)
    # L		Left bundle branch block beat
    # R		Right bundle branch block beat
    # B		Bundle branch block beat (unspecified)
    # A		Atrial premature beat
    # a		Aberrated atrial premature beat
    # J		Nodal (junctional) premature beat
    # S		Supraventricular premature or ectopic beat (atrial or nodal)
    # V		Premature ventricular contraction
    # r		R-on-T premature ventricular contraction
    # F		Fusion of ventricular and normal beat
    # e		Atrial escape beat
    # j		Nodal (junctional) escape beat
    # n		Supraventricular escape beat (atrial or nodal)
    # E		Ventricular escape beat
    # /		Paced beat
    # f		Fusion of paced and normal beat
    # Q		Unclassifiable beat
    # ?		Beat not classified during learning

    # !		Ventricular flutter wave
    # |		Isolated QRS-like artifact

    g = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?', '!', '|', 'P']

    ids = np.in1d(annotation.symbol, g)
    samples = annotation.sample[ids]
    symbols = np.asarray(annotation.symbol)[ids]
    return samples, symbols


def afib_annotations(annotation, convert2int=True):
    """ Get rid of non-beat markers """

    event = ['(AB', '(AFL', '(B', '(BII', '(IVR', '(N', '(NOD', '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT',
             '(AFIB', '~']
    ids = np.in1d(annotation.aux_note, event)
    samples = annotation.sample[ids]
    symbols = np.asarray(annotation.aux_note)[ids]
    return samples, symbols


def bwr(raw, fs, l1=0.2, l2=0.6):
    flen1 = int(l1 * fs / 2)
    flen2 = int(l2 * fs / 2)

    if flen1 % 2 == 0:
        flen1 += 1

    if flen2 % 2 == 0:
        flen2 += 1

    out1 = smooth(raw, flen1)
    out2 = smooth(out1, flen2)
    return raw - out2


def bwr2(raw, fs, l1=0.2, l2=0.6):
    flen1 = int(l1 * fs / 2)
    flen2 = int(l2 * fs / 2)

    if flen1 % 2 == 0:
        flen1 += 1

    if flen2 % 2 == 0:
        flen2 += 1

    out1 = smooth(raw, flen1)
    out2 = smooth(out1, flen2)
    return raw - out2, out2


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window_len % 2 == 0:
        window_len += 1

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    # output = np.argwhere(np.isnan(y))
    # if len(output) > 0:
    #     print(output)
    return y[int(window_len / 2):-int(window_len / 2)]


def norm(raw, window_len, samp_from=-1, samp_to=-1):
    # The window size is the number of samples that corresponds to the time analogue of 2e = 0.5s
    if window_len % 2 == 0:
        window_len += 1

    abs_raw = abs(raw)
    # Remove outlier
    while True:
        g = maximum_filter1d(abs_raw, size=window_len)
        if np.max(abs_raw) <= 5.0:
            break

        abs_raw[g > 5.0] = 0

    g_smooth = smooth(g, window_len, window='hamming')
    g_mean = max(np.mean(g_smooth) / 3.0, 0.1)
    g_smooth = np.clip(g_smooth, g_mean, None)
    # Avoid cases where the value is )
    g_smooth[g_smooth < 0.01] = 1
    normalized = np.divide(raw, g_smooth)

    # if samp_from < 18986081 < samp_to:
    #     print(samp_from)
    #     from matplotlib import pyplot as plt
    #     plt.plot(raw, label="raw")
    #     plt.plot(g_smooth, label="g_smooth")
    #     plt.legend(loc=4)
    #     plt.show()
    #     plt.close()

    return normalized


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_notch_filter(x, fscut, fs, Q=30.0):
    w0 = fscut / (fs / 2)  # Normalized Frequency
    # Design notch filter
    b, a = iirnotch(w0, Q)
    y = filtfilt(b, a, x)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def multi_butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y0 = filtfilt(b, a, data[:, 0])
    y1 = filtfilt(b, a, data[:, 1])
    y2 = filtfilt(b, a, data[:, 2])
    return np.vstack((y0, y1, y2)).T


def iir_bandpass(data, freqmin, freqmax, df, corners=4, zerophase=True):
    """
    :copyright:
    The ObsPy Development Team (devs@obspy.org)

    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def highpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def eclipse_distance(a, b):
    return math.sqrt(math.pow((a - b), 2))


def agglomerative_clustering(labels, fs):
    positions = np.where(labels == 1)[0]
    groups = []
    groups_len = []
    if len(positions) > 0:
        groups_temp = [positions[0]]
        for index in range(1, len(positions)):
            if eclipse_distance(positions[index], groups_temp[-1]) > 0.080 * fs:
                beat_position = int(np.mean(groups_temp))
                groups.append(beat_position)
                groups_len.append(len(groups_temp))

                groups_temp.clear()

            groups_temp.append(positions[index])

        if len(groups_temp) > 0:
            groups.append(int(np.mean(groups_temp)))
            groups_len.append(len(groups_temp))
        groups = np.asarray(groups)
        groups_len = np.asarray(groups_len)

    return groups, groups_len


def agglomerative_clustering_modified(labels, fs):
    positions = np.where(labels == 1)[0]
    groups = []
    for index1 in range(0, len(positions)):
        groups_temp = [positions[index1]]
        for index in range(1, len(positions)):
            if index != index1 and \
                    len(groups_temp) > 5 and \
                    eclipse_distance(positions[index], positions[index1]) > 0.080 * fs:
                beat_position = int(np.round(np.mean(groups_temp)))
                groups.append(beat_position)
                groups_temp.clear()
                break
            groups_temp.append(positions[index])
        if len(groups_temp) > 0:
            groups.append(int(np.round(np.mean(groups_temp))))
        groups_temp.clear()
    return np.asarray(groups)


def calculate_slope(ecg, index_beat, fs):
    # mean slope of the waveform at that position
    slope = np.mean(np.diff(ecg[index_beat - round(0.075 * fs):index_beat]))
    return slope


def is_t_wave(ecg, peak, repeak, fs, qrs_radius=0.05):
    segment_slope = np.rad2deg(np.arctan2((ecg[peak] - ecg[peak - int(qrs_radius * fs)]), int(qrs_radius * fs)))
    last_qrs_slope = np.rad2deg(np.arctan2((ecg[repeak] - ecg[repeak - int(qrs_radius * fs)]), int(qrs_radius * fs)))

    # Should we be using absolute values?
    if abs(segment_slope) <= 0.5 * abs(last_qrs_slope):
        return True
    else:
        return False


def select_beat_from_group(beats, groups_len, ecg, fs, num_before_qrs_candidate, min_rr):
    beats = np.add(beats, num_before_qrs_candidate)
    selected_beats = []
    re_peak = beats[0]
    re_peak_len = groups_len[0]
    selected_beats.append(beats[0])
    for i in range(1, len(beats)):
        peak = beats[i]
        if abs(peak - re_peak) < 0.36 * fs and is_t_wave(ecg, peak, re_peak, fs):
            continue

        if groups_len[i] < 5:
            continue

        if peak - re_peak > (min_rr * fs):
            re_peak = peak
            re_peak_len = groups_len[i]
            selected_beats.append(peak)
        elif groups_len[i] > re_peak_len:
            re_peak = peak
            re_peak_len = groups_len[i]
            selected_beats[-1] = peak

    return np.asarray(selected_beats)


def get_beat1d(buf_ecg, qrs_candidate, sampling_rate, offset_len=0, min_rr=0.2):
    """

    :param buf_ecg:
    :param qrs_candidate:
    :param sampling_rate:
    :param offset_len:
    :param min_rr:
    :return:
    """
    groups, groups_len = agglomerative_clustering(qrs_candidate,
                                                  sampling_rate)
    if len(groups) > 0:
        beats = select_beat_from_group(groups,
                                       groups_len,
                                       buf_ecg,
                                       sampling_rate,
                                       offset_len,
                                       min_rr)
    else:
        beats = []
    return np.asarray(beats, dtype=int)


def get_beat2d(ecg_bug, group_data, qrs_candidate, beat_class, fs, overlap, debug=True):
    """

    :param group_data:
    :param qrs_candidate:
    :param beat_class:
    :return:
    """
    beats = []
    syms = []
    amps = []
    index = 0
    ind_revert = {i: k for i, k in enumerate(beat_class.keys())}
    for i, p in enumerate(qrs_candidate):
        if i == 24522:
            print()

        if p > 0:
            beat_candidate = []
            amp_candidate = []
            try:
                for g in group_data[i]:
                    peaks, _ = find_peaks(g)
                    peaks_, _ = find_peaks(g * -1.0)
                    if len(peaks_) and len(peaks) > 0:
                        peaks = np.concatenate((peaks, peaks_))

                    if len(peaks) > 1:
                        ma = np.abs(g[peaks] - np.mean(g))
                        mb = np.argmax(ma)
                        amp_candidate.append(ma[mb])
                        beat_candidate.append(peaks[mb])
                    elif len(peaks) == 1:
                        amp_candidate.append(np.abs(g[peaks[0]] - np.mean(g)))
                        beat_candidate.append(peaks[0])
                    #
                    # plt.plot(g)
                    # plt.annotate('*', xy=(peaks[mb], g[peaks[mb]]), color='red')
                    # plt.show()
                if len(beat_candidate) == 0:
                    index += overlap
                    continue

                beat_candidate = np.asarray(beat_candidate, dtype=int)
                amp_candidate = np.asarray(amp_candidate, dtype=float)
                amb = np.argmax(amp_candidate)
            except Exception as err:
                plt.plot(g)
                plt.show()
                print("qrs_candidate[{}] get err {}".format(i, err))

            peak = beat_candidate[amb] + amb + index
            if len(beats) > 0:
                re_peak = beats[-1]
                if abs(peak - re_peak) < 0.2 * fs:
                    index += overlap
                    continue

                if abs(peak - re_peak) < 0.36 * fs and is_t_wave(ecg_bug, peak, re_peak, fs):
                    index += overlap
                    continue

            beats.append(beat_candidate[amb] + amb + index)
            syms.append(beat_class[ind_revert[p]][0])
            amps.append(amp_candidate[amb])
            if debug:
                plt.title(str(i))
                plt.plot(group_data[i][amb])
                plt.annotate(beat_class[ind_revert[p]][0],
                             xy=(beat_candidate[amb], group_data[i][amb][beat_candidate[amb]]),
                             color='red')
                plt.show()

        index += overlap

    syms = [x for _, x in sorted(zip(beats, syms))]
    return np.sort(np.asarray(beats, dtype=int)), np.asarray(syms), np.asarray(amps)


def check_type_event(event, type):
    return int(event & type == type)


def remove_short_event(list_event,
                       event,
                       afib_len,
                       brady_len,
                       tachy_len,
                       pause_len,
                       is_last_beat,
                       num_beat_remove=6):
    """

    :param list_event:
    :param event:
    :param afib_len:
    :param brady_len:
    :param tachy_len:
    # :param pause_len:
    :param is_last_beat:
    :param num_beat_remove:
    :return:
    """

    if check_type_event(event, define.NOISE) == 1:
        event = define.NOISE

    if len(list_event) > 0:
        # TACHY
        if check_type_event(list_event[-1], define.TACHY) == 0:
            if check_type_event(event, define.TACHY) == 1:
                list_event[-1] |= define.TACHY
                tachy_len += 1

        elif check_type_event(event, define.TACHY) == 0:
            if tachy_len < num_beat_remove:
                for i in range(-tachy_len, 0):
                    list_event[i] = list_event[i] & (define.BRADY + define.AFIB + define.PAUSE + define.NOISE)

            tachy_len = 0

        # BRADY
        if check_type_event(list_event[-1], define.BRADY) == 0:
            if check_type_event(event, define.BRADY) == 1:
                list_event[-1] |= define.BRADY
                brady_len += 1

        elif check_type_event(event, define.BRADY) == 0:
            if brady_len < num_beat_remove:
                for i in range(-brady_len, 0):
                    list_event[i] = list_event[i] & (define.TACHY + define.AFIB + define.PAUSE + define.NOISE)

            brady_len = 0

        # AFIB
        if check_type_event(list_event[-1], define.AFIB) == 0:
            if check_type_event(event, define.AFIB) == 1:
                list_event[-1] |= define.AFIB
                afib_len += 1

        elif check_type_event(event, define.AFIB) == 0:
            if afib_len < num_beat_remove:
                for i in range(-afib_len, 0):
                    list_event[i] = list_event[i] & (define.TACHY + define.BRADY + define.PAUSE + define.NOISE)

            afib_len = 0

        # PAUSE
        if check_type_event(list_event[-1], define.PAUSE) == 0:
            if check_type_event(event, define.PAUSE) == 1:
                list_event[-1] |= define.PAUSE
                pause_len += 1

        elif check_type_event(event, define.PAUSE) == 0:
            pause_len = 0

    tachy_len += check_type_event(event, define.TACHY)
    brady_len += check_type_event(event, define.BRADY)
    afib_len += check_type_event(event, define.AFIB)
    pause_len += check_type_event(event, define.PAUSE)

    if is_last_beat:
        if tachy_len < num_beat_remove and (check_type_event(event, define.TACHY) == 1):
            event = event & (define.BRADY + define.AFIB + define.PAUSE)
            for i in range(-tachy_len, 0):
                list_event[i] = list_event[i] & (define.BRADY + define.AFIB + define.PAUSE + define.NOISE)

        if brady_len < num_beat_remove and (check_type_event(event, define.BRADY) == 1):
            event = event & (define.TACHY + define.AFIB + define.PAUSE)
            for i in range(-brady_len, 0):
                list_event[i] = list_event[i] & (define.TACHY + define.AFIB + define.PAUSE + define.NOISE)

        if afib_len < num_beat_remove and (check_type_event(event, define.AFIB) == 1):
            event = event & (define.TACHY + define.BRADY + define.PAUSE)
            for i in range(-afib_len, 0):
                list_event[i] = list_event[i] & (define.TACHY + define.BRADY + define.PAUSE + define.NOISE)

    list_event.append(event)
    return list_event, tachy_len, brady_len, afib_len, pause_len


def beat_select(ibeats, isymbols, iamps, buf_bwr_ecg, fs, thr=0.5, pre_peak=2):
    """

    """
    selected_beats = []
    selected_amps = []
    selected_symbols = []
    st = 0

    selected_beats.append(ibeats[st])
    selected_symbols.append(isymbols[st])
    selected_amps.append(iamps[st])
    for i in range(st + 1, len(ibeats)):
        peak = ibeats[i]
        amp = iamps[i]
        if abs(peak - selected_beats[-1]) < 0.36 * fs and is_t_wave(buf_bwr_ecg, peak, selected_beats[-1], fs):
            continue

        cnt = 0
        mean_amp = 0
        for st in reversed(range(i)):
            if isymbols[st] not in ["Q", "V"]:
                mean_amp += iamps[st]
                cnt += 1
                if cnt > pre_peak:
                    break

        if cnt > 0:
            mean_amp = (mean_amp / cnt)
            if amp < mean_amp * thr and isymbols[i] not in ["Q", "V"]:
                isymbols[i] = "Q"
                continue

            if amp >= mean_amp * thr * 3 and isymbols[i] not in ["Q", "V"]:
                isymbols[i] = "Q"

        symbol = isymbols[i]
        selected_beats.append(peak)
        selected_symbols.append(symbol)
        selected_amps.append(amp)

    return np.asarray(selected_beats), np.asarray(selected_symbols), np.asarray(selected_amps)


def beat_cluster(beats, symbols, amps, min_rr):
    _beats = []
    _symbols = []
    _amps = []
    _lens = []
    if len(beats) > 0:
        groups_beat = [beats[0]]
        groups_syms = [symbols[0]]
        groups_amps = [amps[0]]
        for index in range(1, len(beats)):
            if (beats[index] - groups_beat[-1]) > min_rr:
                _beats.append(groups_beat.copy())
                _symbols.append(groups_syms.copy())
                _amps.append(groups_amps.copy())
                _lens.append(len(groups_beat))

                groups_beat.clear()
                groups_syms.clear()
                groups_amps.clear()

            groups_beat.append(beats[index])
            groups_syms.append(symbols[index])
            groups_amps.append(amps[index])

        if len(groups_beat) > 0:
            _beats.append(groups_beat.copy())
            _symbols.append(groups_syms.copy())
            _amps.append(groups_amps.copy())
            _lens.append(len(groups_beat))

    return _beats, _symbols, _amps, _lens
