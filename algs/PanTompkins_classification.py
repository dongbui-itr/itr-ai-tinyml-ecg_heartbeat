import numpy as np

import wfdb
from algs import config_qrs as cf
import scipy.signal as signal

from algs.preprocessing_beat import *
from algs.peak_detection import peakdetect
from algs.DecisionTree_rr_update import *
from algs.pan_tompkins_plus import Pan_Tompkins_Plus_Plus
from algs.decision_tree import QRSAmplitudeSpecs, QRSRRSpecs, DecisionTree
from utils.reprocessing import butter_bandpass_filter, butter_lowpass_filter, butter_highpass_filter

DATA_TYPE = 'float64'


class PanTompkinsClassification:
    def __init__(
            self,
            data_path: str,
            file: str,
            ext: str,
            **kwargs
    ):
        self.fs = cf.SAMPLING_RATE
        self.file_name = file
        self.file_path = data_path + '/'
        self.ext = ext
        self.ecg = None
        self.decision_tree_func = DecisionTree()

        try:
            self.view_signal = kwargs.get('view_signal')
        except (Exception,):
            self.view_signal = False

    @staticmethod
    def find_max_locate(ecg_array):
        arr = ecg_array
        y_i = np.amax(arr)
        i_1, = np.where(arr == y_i)
        x_i = i_1[0]
        return y_i, i_1, x_i

    @staticmethod
    def derivative_filter(fs):
        if fs != 200:
            int_c = (4 - 0) / (fs * (1 / 40))
            x = np.arange(0, 4)
            xp = np.array([1, 2, 0, -2, -1]) * (1 / 8) * fs
            fp = np.linspace(0, int_c, 5)  # mang 5 gia tri buoc nhay int_c
            b = np.interp(x, xp, fp)  # 1-D data interpolation (trả về mảng 1 chiều nội suy)b = np.interp(x, xp, fp)
        else:
            b = np.array([1, 2, 0, -2, -1]) * (1 / 8) * fs
        return b

    def pan_tompkins(self, ecg, fs):
        THR_interval = cf.THR_Beat2Beat * fs

        qrs_c = []  # np.zeros(shape=(1,1)) # amplitude of R
        qrs_i = []  # index
        nois_c = []
        nois_i = []
        m_selected_RR = 0
        mean_RR = 0
        qrs_i_raw = []
        qrs_amp_raw = []
        ser_back = []
        ecg = np.nan_to_num(ecg)
        if fs == 200:
            ecg = ecg - np.mean(ecg)  # remove mean of Signal
            # Low pass filter
            ecg_l = butter_lowpass_filter(data=ecg, cutoff=12, fs=fs, order=3)
            ecg_l = ecg_l / (np.max(abs(ecg_l)))
            # Highpass filter
            ecg_h = butter_highpass_filter(data=ecg_l, cutoff=5, fs=fs, order=3)
            ecg_h = ecg_h / np.max(abs(ecg_h))
        else:
            # Bandpass filter for Noise cancelation of other sampling frequencies(Filtering)
            ecg_h = butter_bandpass_filter(data=ecg, lowcut=5, highcut=15, fs=fs, order=3)
            ecg_h = ecg_h / np.max(abs(ecg_h))

        # Derivative filter H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))#######################
        b = self.derivative_filter(fs)
        ecg_d = signal.filtfilt(b, 1, ecg_h)
        ecg_d = ecg_d / np.max(ecg_d)

        # Squaring nonlinearly enhance the dominant peaks
        ecg_s = ecg_d ** 2
        window_len = round(0.150 * fs)
        if window_len % 2 == 0:
            window_len += 1
        s = np.r_[
            ecg_s[window_len - 1:0:-1], ecg_s, ecg_s[
                                               -2:-window_len - 1:-1]]  # add element into a array. ex.between ecg_s
        ecg_m = np.convolve(s, np.ones(window_len), mode='valid') / window_len
        ecg_m = ecg_m[int(window_len / 2):-int(window_len / 2)]

        # Find all peaks in ecg signal
        # locs = signal.find_peaks(ecg_m, threshold=0.00000000001, distance=fs * 0.22)[0]

        locs, _ = peakdetect(ecg_m, None, int(window_len))
        locs = np.asarray(np.asarray(locs)[:, 0], dtype=int)

        pks = ecg_m[locs]
        # Initialize the training phase (2 seconds of the signal) to determine the THR_SIG and THR_NOISE
        THR_SIG = (np.max(ecg_m[0:2 * fs])) * 1 / 3  # 0.25 of the max amplitude
        THR_NOISE = (np.mean(ecg_m[0:2 * fs])) * 1 / 2  # 0.5 of the mean signal is considered to be noise
        SIG_LEV = THR_SIG
        NOISE_LEV = THR_NOISE
        # Initialize bandpass filter threshold(2 seconds of the bandpass signal)
        THR_SIG1 = (np.max(ecg_h[0:2 * fs])) * 1 / 3
        THR_NOISE1 = (np.mean(ecg_h[0:2 * fs])) * 1 / 2
        SIG_LEV1 = THR_SIG1
        NOISE_LEV1 = THR_NOISE1

        # plt.plot(ecg_h)
        # plt.plot(ecg_m)
        # plt.show()

        for i in range(0, len(pks)):
            # locate the corresponding peak in the filtered signal

            if (locs[i] - int(round(0.150 * fs))) >= 1 and locs[i] <= len(ecg_h) - 1:
                y_i, i_l, x_i = self.find_max_locate(ecg_h[(locs[i] - int(round(0.150 * fs))):(locs[i])])
            else:
                if i == 0:
                    y_i, i_1, x_i = self.find_max_locate(ecg_h[0:locs[i]])
                    ser_back = 1
                elif locs[i] >= len(ecg_h):
                    y_i, i_2, x_i = self.find_max_locate(ecg_h[(locs[i] - round(0.150 * fs)):-1])

            # update the number of beats (Two heart rate means one the most recent and the other selector)
            x = len(qrs_c)
            # calculate the mean of the last 8 R waves to make sure that QRS is not
            # missing(If no R detected, trigger a search back) 1.66*mean
            if x >= 9:
                x1 = np.asarray(qrs_i[x - 9: x])
                diffRR = np.diff(x1)  # calculate RR interval
                mean_RR = np.mean(diffRR)  # calculater the mean of 8 previous R waves interval
                comp = qrs_i[-1] - qrs_i[-2]  # lastest RR
                if comp <= 0.92 * mean_RR or comp >= 1.16 * mean_RR:  # most recent RR interval that fell between the accpetable low and high RR-interval limits
                    # lower down thresholds to detect better in MVI
                    THR_SIG = 0.5 * THR_SIG
                    # lower down thresholds to detect better in Bandpass filtered
                    THR_SIG1 = 0.5 * THR_SIG1
                else:
                    m_selected_RR = mean_RR  # the lastest regular beats mean

            if m_selected_RR:
                test_m = m_selected_RR  # if the regular RR available use it
            elif mean_RR and m_selected_RR == 0:
                test_m = mean_RR
            else:
                test_m = 0

            if test_m:
                if ((locs[i] - qrs_i[-1]) >= round(
                        1.66 * test_m)):  # and (qrs_i[-1] + round(0.2 * fs)) > (locs[i] - round(0.2 * fs)):  # it shows a QRS is missed
                    try:
                        pks_temp, i_3, locs_temp = self.find_max_locate(
                            ecg_m[(qrs_i[-1] + round(0.2 * fs)):(locs[i] - round(0.2 * fs))])
                    except (Exception, ) as e:
                        pks_temp, i_3, locs_temp = self.find_max_locate(
                            ecg_m[(qrs_i[-1] + round(0.15 * fs)):(locs[i] - round(0.15 * fs))])
                        # print("{} get error: {}".format(i, e))
                        # continue

                    locs_temp = qrs_i[-1] + round(0.2 * fs) + locs_temp - 1  # location
                    if pks_temp > THR_NOISE:
                        qrs_c.append(pks_temp)
                        qrs_i.append(locs_temp)
                    # find the location in filtered sig
                    if locs_temp <= len(ecg_h):
                        y_i_t, i_4, x_i_t = self.find_max_locate(ecg_h[(locs_temp - round(0.150 * fs)):locs_temp])
                    else:
                        y_i_t, i_5, x_i_t = self.find_max_locate(ecg_h[(locs_temp - round(0.150 * fs)):locs_temp])
                    # take care of bandpass signal threshold
                    if y_i_t > THR_NOISE1:
                        qrs_i_raw.append(locs_temp - round(0.150 * fs) + (x_i_t - 1))  # save index of bandpass
                        qrs_amp_raw.append(y_i_t)  # save amplitude of bandpass
                        SIG_LEV1 = 0.25 * y_i_t + 0.75 * SIG_LEV1  # when found with the second thres

                    not_nois = 1
                    SIG_LEV = 0.25 * pks_temp + 0.75 * SIG_LEV

            # find noise and QRS peaks
            if pks[i] >= THR_SIG:
                # if a QRS candidate occurs within 360ms of the previous QRS
                # the algorithm determines if its T wave or QRS
                skip = 0
                if len(qrs_c) >= 3:
                    if (locs[i] - qrs_i[-1]) <= round(0.3600 * fs):
                        Slope1 = np.mean(np.diff(
                            ecg_m[locs[i] - round(0.075 * fs):locs[i]]))  # mean slope of the waveform at that position
                        Slope2 = np.mean(
                            np.diff(ecg_m[(qrs_i[-1] - round(0.075 * fs)): qrs_i[-1]]))  # mean slope of previous R wave
                        if abs(Slope1) <= abs(0.5 * Slope2):  # slope less then 0.5 of previous R
                            nois_c.append(pks[i])
                            nois_i.append(locs[i])
                            skip = 1  # T wave identification
                            # adjust noise level in both filtered and MVI
                            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                            NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV
                        # else:
                        #     skip = 0
                if skip == 0:  # skip is 1 when a T wave is detected
                    qrs_c.append(pks[i])
                    qrs_i.append(locs[i])
                    # bandpass filter check threshold
                    if y_i >= THR_SIG1:
                        if ser_back:
                            qrs_i_raw.append(x_i)
                        else:
                            qrs_i_raw.append(locs[i] - round(0.150 * fs) + (x_i - 1))
                    qrs_amp_raw.append(y_i)  # save amplitude of bandpass
                    if 0.5 * (0.875 * SIG_LEV1) < 0.125 * y_i < 1.5 * (0.875 * SIG_LEV1):
                        SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1  # adjust threshold for bandpass filtered sig

                # adjust Signal level
                if 0.5 * (0.875 * SIG_LEV) < 0.125 * pks[i] < 1.5 * (0.875 * SIG_LEV):
                    SIG_LEV = 0.125 * pks[i] + 0.875 * SIG_LEV

            elif (THR_NOISE <= pks[i]) and (pks[i] < THR_SIG):
                # adjust Noise level in filtered sig
                NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                # adjust Noise level in MVI
                NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

            elif pks[i] < THR_NOISE:
                nois_c.append(pks[i])
                nois_i.append(locs[i])
                NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

            # adjust the threshold with SNR
            if NOISE_LEV != 0 or SIG_LEV != 0:
                THR_SIG = NOISE_LEV + 0.25 * (abs(SIG_LEV - NOISE_LEV))
                THR_NOISE = 0.5 * THR_SIG

            # adjust the threshold with SNR for bandpassed signal
            if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
                THR_SIG1 = NOISE_LEV1 + 0.25 * (abs(SIG_LEV1 - THR_NOISE1))
                THR_NOISE1 = 0.5 * THR_SIG1

        indx = np.flatnonzero(np.diff(np.asarray(qrs_i)) < THR_interval) + 1
        if len(indx) > 0:
            _qrs_i = np.delete(qrs_i, indx)
        else:
            _qrs_i = qrs_i

        symbol = np.asarray('N' * len(_qrs_i))

        return _qrs_i, symbol

    @staticmethod
    def pan_tompkins_2(ecg, fs):
        pan_tompkins_func = Pan_Tompkins_Plus_Plus()
        r_peak_raw = pan_tompkins_func.rpeak_detection(ecg, fs)
        symbol = np.asarray(['N'] * len(r_peak_raw))
        r_peak = np.asarray(r_peak_raw, dtype=int)

        return r_peak, symbol

    def svm_classification(self, ecg, fs, samples, samples_ref, symbols_ref):
        width_qrs = 0.16  # 20
        before = 0.06
        bef_ind = int(before * fs)
        aft_ind = int((width_qrs - before) * fs)

        samples_raw = samples.copy()
        signal = butter_bandpass_filter(ecg, 1, 30, fs)

        list_width_20 = []
        list_width_50 = []
        list_width_80 = []
        list_max_amp = []
        list_min_amp = []
        list_peak_peak = []
        list_cali_sample = []
        list_rr_interval = []
        list_sample = []
        list_symbol = []
        cali_samples = []

        cali_flag = True

        for i_sample, sample in enumerate(samples):
            if i_sample == len(samples) - 1:
                continue

            sample = calibrate_position_beat(signal, sample, bef_ind, aft_ind)
            cali_samples.append(sample)
            list_cali_sample.append(sample)

            if len(list_cali_sample) < 2:
                list_rr_interval.append(-1)
            else:
                list_rr_interval.append(sample - list_cali_sample[-2])

            (ind_bef_20, ind_aft_20, ind_bef_50, ind_aft_50, ind_bef_80, ind_aft_80,
             width_20, width_50, width_80, max_amp, min_amp, peak_peak) = beat_width(signal, sample, bef_ind, aft_ind)

            list_width_20.append(width_20)
            list_width_50.append(width_50)
            list_width_80.append(width_80)
            list_min_amp.append(min_amp)
            list_max_amp.append(max_amp)
            list_peak_peak.append(peak_peak)
            list_sample.append(sample)
            list_symbol.append("N")

            ########## Calibrate ############
            # Cali_flag == True
            # when i_sample > cf.NUM_BEAT_TO_CALI * 2 (1 min), length of cali may be less than cf.NUM_ENOUGH_BEAT
            if i_sample < cf.NUM_BEAT_TO_CALI * 2 and cali_flag:
                if len(list_sample) >= cf.NUM_BEAT_TO_CALI:

                    tmp_sample = np.asarray(list_sample)[:i_sample]
                    rr_interval = np.diff(tmp_sample)
                    rr_interval = np.concatenate(([rr_interval[0]], rr_interval))
                    rr_ratio = rr_interval[:-1] / rr_interval[1:]
                    # Calibrate RR, AMP, WIDTH
                    abnormal_rr = np.flatnonzero(
                        (rr_ratio > cf.SVM_THR_RRratio_MAX) | (cf.SVM_THR_RRratio_MIN > rr_ratio))
                    if len(tmp_sample) - len(abnormal_rr) > len(tmp_sample) / 3:
                        # Normal > Abnormal
                        """
                        1/ Remove abnormal RR and calculate mean of RR = Cali_RR
                        """
                        cali_list_RR = np.delete(rr_interval, abnormal_rr + 1)
                        cali_list_W20 = np.delete(np.asarray(list_width_20), abnormal_rr)
                        cali_list_W50 = np.delete(np.asarray(list_width_50), abnormal_rr)
                        cali_list_W80 = np.delete(np.asarray(list_width_80), abnormal_rr)
                        cali_list_min_amp = np.delete(np.asarray(list_min_amp), abnormal_rr)
                        cali_list_max_amp = np.delete(np.asarray(list_max_amp), abnormal_rr)
                        cali_list_peak_peak = np.delete(np.asarray(list_peak_peak), abnormal_rr)

                        _tmp_std = np.std(cali_list_W20)
                        _normal_indx = np.flatnonzero(np.abs(cali_list_W20 - np.mean(cali_list_W20)) <= _tmp_std)
                        cali_list_W20 = cali_list_W20[_normal_indx]
                        _tmp_std = np.std(cali_list_W50)
                        _normal_indx = np.flatnonzero(np.abs(cali_list_W50 - np.mean(cali_list_W50)) <= _tmp_std)
                        cali_list_W50 = cali_list_W50[_normal_indx]
                        _tmp_std = np.std(cali_list_W80)
                        _normal_indx = np.flatnonzero(np.abs(cali_list_W80 - np.mean(cali_list_W80)) <= _tmp_std)
                        cali_list_W80 = cali_list_W80[_normal_indx]

                        _tmp_std = np.std(cali_list_max_amp)
                        _normal_indx = np.flatnonzero(
                            np.abs(cali_list_max_amp - np.mean(cali_list_max_amp)) <= _tmp_std)
                        cali_list_max_amp = cali_list_max_amp[_normal_indx]
                        _tmp_std = np.std(cali_list_min_amp)
                        _normal_indx = np.flatnonzero(
                            np.abs(cali_list_min_amp - np.mean(cali_list_min_amp)) <= _tmp_std)
                        cali_list_min_amp = cali_list_min_amp[_normal_indx]
                        _tmp_std = np.std(cali_list_peak_peak)
                        _normal_indx = np.flatnonzero(
                            np.abs(cali_list_peak_peak - np.mean(cali_list_peak_peak)) <= _tmp_std)
                        cali_list_peak_peak = cali_list_peak_peak[_normal_indx]

                    else:
                        # Normal < Abnormal
                        """
                        1/ Find 2 RR at abnormal RR, if abnormal 2RR ~ normal 2RR ==> abnormal 2 RR / 2 = normal RR
                        """
                        _abnormal_rr = np.flatnonzero(rr_ratio > cf.SVM_THR_RRratio_MAX)
                        _2rr_interval = rr_interval[_abnormal_rr] + rr_interval[_abnormal_rr - 1]
                        _2rr_interval_std = np.std(_2rr_interval)
                        _normal_2rr_interval_indx = np.flatnonzero(
                            np.abs(_2rr_interval - np.mean(_2rr_interval)) < _2rr_interval_std)
                        _normal_2rr_interval = _2rr_interval[_normal_2rr_interval_indx]
                        cali_list_RR = _normal_2rr_interval // 2
                        cali_list_W20 = np.asarray(list_width_20)[_abnormal_rr[_normal_2rr_interval_indx]]
                        cali_list_W50 = np.asarray(list_width_50)[_abnormal_rr[_normal_2rr_interval_indx]]
                        cali_list_W80 = np.asarray(list_width_80)[_abnormal_rr[_normal_2rr_interval_indx]]
                        cali_list_max_amp = np.asarray(list_max_amp)[_abnormal_rr[_normal_2rr_interval_indx]]
                        cali_list_min_amp = np.asarray(list_min_amp)[_abnormal_rr[_normal_2rr_interval_indx]]
                        cali_list_peak_peak = np.asarray(list_peak_peak)[_abnormal_rr[_normal_2rr_interval_indx]]

                        _tmp_std = np.std(cali_list_W20)
                        _normal_indx = np.flatnonzero(np.abs(cali_list_W20 - np.mean(cali_list_W20)) <= _tmp_std)
                        cali_list_W20 = cali_list_W20[_normal_indx]
                        _tmp_std = np.std(cali_list_W50)
                        _normal_indx = np.flatnonzero(np.abs(cali_list_W50 - np.mean(cali_list_W50)) <= _tmp_std)
                        cali_list_W50 = cali_list_W50[_normal_indx]
                        _tmp_std = np.std(cali_list_W80)
                        _normal_indx = np.flatnonzero(np.abs(cali_list_W80 - np.mean(cali_list_W80)) <= _tmp_std)
                        cali_list_W80 = cali_list_W80[_normal_indx]

                        _tmp_std = np.std(cali_list_max_amp)
                        _normal_indx = np.flatnonzero(
                            np.abs(cali_list_max_amp - np.mean(cali_list_max_amp)) <= _tmp_std)
                        cali_list_max_amp = cali_list_max_amp[_normal_indx]
                        _tmp_std = np.std(cali_list_min_amp)
                        _normal_indx = np.flatnonzero(
                            np.abs(cali_list_min_amp - np.mean(cali_list_min_amp)) <= _tmp_std)
                        cali_list_min_amp = cali_list_min_amp[_normal_indx]
                        _tmp_std = np.std(cali_list_peak_peak)
                        _normal_indx = np.flatnonzero(
                            np.abs(cali_list_peak_peak - np.mean(cali_list_peak_peak)) <= _tmp_std)
                        cali_list_peak_peak = cali_list_peak_peak[_normal_indx]

                    if np.min((len(cali_list_RR), len(cali_list_W20), len(cali_list_W50), len(cali_list_W80),
                               len(cali_list_max_amp), len(cali_list_min_amp),
                               len(cali_list_peak_peak))) > cf.NUM_ENOUGH_BEAT:
                        # print(i_sample)
                        cali_flag = False
                    break

        local_list_RR = cali_list_RR
        local_list_W20 = cali_list_W20
        local_list_W50 = cali_list_W50
        local_list_W80 = cali_list_W80
        local_list_min_amp = cali_list_min_amp
        local_list_max_amp = cali_list_max_amp
        local_list_peak_peak = cali_list_peak_peak

        cali_list_RR_mean = np.mean(cali_list_RR)
        cali_list_W20_mean = np.mean(cali_list_W20)
        cali_list_W50_mean = np.mean(cali_list_W50)
        cali_list_W80_mean = np.mean(cali_list_W80)
        cali_list_min_amp_mean = np.mean(cali_list_min_amp)
        cali_list_max_amp_mean = np.mean(cali_list_max_amp)
        cali_list_peak_peak_mean = np.mean(cali_list_peak_peak)

        local_list_RR_mean = np.mean(local_list_RR)
        local_list_W20_mean = np.mean(local_list_W20)
        local_list_W50_mean = np.mean(local_list_W50)
        local_list_W80_mean = np.mean(local_list_W80)
        local_list_min_amp_mean = np.mean(local_list_min_amp)
        local_list_max_amp_mean = np.mean(local_list_max_amp)
        local_list_peak_peak_mean = np.mean(local_list_peak_peak)

        ## Load DecisionTree Weight
        svm_symbols = np.asarray(['N'] * len(samples))
        buf_normal_beat = dict()
        list_cali_sample = []
        for i_sample, sample in enumerate(samples):
            sample = calibrate_position_beat(signal, sample, bef_ind, aft_ind)
            list_cali_sample.append(sample)

            if i_sample == len(samples) - 1:
                continue

            (ind_bef_20, ind_aft_20, ind_bef_50, ind_aft_50, ind_bef_80, ind_aft_80,
             width_20, width_50, width_80, max_amp, min_amp, peak_peak) = beat_width(signal, sample, bef_ind, aft_ind)

            qrs = QRSAmplitudeSpecs()
            qrs.w20 = width_20
            qrs.w50 = width_50
            qrs.w80 = width_80
            qrs.min_amp = min_amp
            qrs.max_amp = max_amp
            qrs.peak_peak = peak_peak

            # Using RR to classify N vs SV
            if i_sample > 3:

                qrs_rr = QRSRRSpecs()
                [qrs_rr.RR1, qrs_rr.RR2, qrs_rr.RR3] = np.diff(list_cali_sample[i_sample - 3: i_sample + 1])
                qrs_rr.R12 = qrs_rr.RR1 / qrs_rr.RR2
                qrs_rr.R23 = qrs_rr.RR2 / qrs_rr.RR3
                qrs_rr.RR1_CAL = qrs_rr.RR1 / cali_list_RR_mean
                qrs_rr.RR2_CAL = qrs_rr.RR2 / cali_list_RR_mean
                qrs_rr.RR3_CAL = qrs_rr.RR3 / cali_list_RR_mean
                qrs_rr.RR1_UDT = qrs_rr.RR1 / local_list_RR_mean
                qrs_rr.RR2_UDT = qrs_rr.RR2 / local_list_RR_mean
                qrs_rr.RR3_UDT = qrs_rr.RR3 / local_list_RR_mean

                # Using AMP to classify N vs S vs V
                ratio_to_calibrate = QRSAmplitudeSpecs()
                ratio_to_calibrate.w20 = qrs.w20 / cali_list_W20_mean
                ratio_to_calibrate.w50 = qrs.w50 / cali_list_W50_mean
                ratio_to_calibrate.w80 = qrs.w80 / cali_list_W80_mean
                ratio_to_calibrate.min_amp = qrs.min_amp / cali_list_min_amp_mean
                ratio_to_calibrate.max_amp = qrs.max_amp / cali_list_max_amp_mean
                ratio_to_calibrate.peak_peak = qrs.peak_peak / cali_list_peak_peak_mean

                ratio_to_updated = QRSAmplitudeSpecs()
                ratio_to_updated.w20 = qrs.w20 / local_list_W20_mean
                ratio_to_updated.w50 = qrs.w50 / local_list_W50_mean
                ratio_to_updated.w80 = qrs.w80 / local_list_W80_mean
                ratio_to_updated.min_amp = qrs.min_amp / local_list_min_amp_mean
                ratio_to_updated.max_amp = qrs.max_amp / local_list_max_amp_mean
                ratio_to_updated.peak_peak = qrs.peak_peak / local_list_peak_peak_mean

                # region Using algorithm
                symbol_before = None if i_sample == 0 else svm_symbols[i_sample - 1]

                label = self.decision_tree_func.process(
                    sym_bef=symbol_before,
                    qrs_rr=qrs_rr,
                    qrs_beat=qrs,
                    ratio_to_updated=ratio_to_updated,
                    ratio_to_calibrate=ratio_to_calibrate
                )
                # endregion Using algorithm

                if label != 'N':
                    if label == 'V' and svm_symbols[i_sample - 1] == 'N' and len(buf_normal_beat) > 2:
                        max_amp_ratio = max_amp / buf_normal_beat['max_amp']
                        min_amp_ratio = min_amp / buf_normal_beat['min_amp']
                        w20_ratio = width_20 / buf_normal_beat['w20']
                        w50_ratio = width_50 / buf_normal_beat['w50']
                        w80_ratio = width_80 / buf_normal_beat['w80']

                        amplitude_check = any([
                            max_amp_ratio > cf.THR_AMPMINratio_MAX_LOCAL,
                            max_amp_ratio < cf.THR_AMPMAXratio_MIN_LOCAL,
                            min_amp_ratio > cf.THR_AMPMINratio_MAX_LOCAL,
                            min_amp_ratio < cf.THR_AMPMINratio_MIN_LOCAL
                        ])
                        if amplitude_check:
                            label = 'V'
                        else:
                            label = 'N'

                    svm_symbols[i_sample] = label
                    # print(f'{samples[i_sample - 1]}: {svm_symbols[i_sample - 1]}')
                else:  # S RUN:
                    if (svm_symbols[i_sample - 1] == 'S'
                            and (qrs_rr.R12 < THR_RUN_RR1_MAX)
                            and (qrs_rr.RR2_UDT < THR_RUN_RR_MIN and qrs_rr.RR2_CAL < THR_RUN_RR_MIN)):
                        if qrs_rr.R23 > THR_RUN_RR_MIN:
                            svm_symbols[i_sample] = 'S'
                        else:
                            svm_symbols[i_sample] = 'N'
                    elif (not 'V' in svm_symbols[i_sample - 3: i_sample] and
                          not 'S' in svm_symbols[i_sample - 3: i_sample]):  # and flag_abnormal:

                        local_list_RR = np.concatenate((local_list_RR[1:], [samples[i_sample] - samples[i_sample - 1]]))
                        local_list_W20 = np.concatenate((local_list_W20[1:], [width_20]))
                        local_list_W50 = np.concatenate((local_list_W50[1:], [width_50]))
                        local_list_W80 = np.concatenate((local_list_W80[1:], [width_80]))
                        local_list_min_amp = np.concatenate((local_list_min_amp[1:], [min_amp]))
                        local_list_max_amp = np.concatenate((local_list_max_amp[1:], [max_amp]))
                        local_list_peak_peak = np.concatenate((local_list_peak_peak[1:], [peak_peak]))

                        local_list_RR_mean = np.mean(local_list_RR)
                        local_list_W20_mean = np.mean(local_list_W20)
                        local_list_W50_mean = np.mean(local_list_W50)
                        local_list_W80_mean = np.mean(local_list_W80)
                        local_list_min_amp_mean = np.mean(local_list_min_amp)
                        local_list_max_amp_mean = np.mean(local_list_max_amp)
                        local_list_peak_peak_mean = np.mean(local_list_peak_peak)

                    buf_normal_beat['min_amp'] = min_amp
                    buf_normal_beat['max_amp'] = max_amp
                    buf_normal_beat['peak_peak'] = peak_peak
                    buf_normal_beat['w20'] = width_20
                    buf_normal_beat['w50'] = width_50
                    buf_normal_beat['w80'] = width_80

        symbols_ref = np.asarray(symbols_ref)
        svm_symbols = np.asarray(svm_symbols)

        # View result
        if self.view_signal:
            plt.plot(signal)

            ymax1 = max(signal)
            if max(signal) > 1:
                ymax1 = 1

            ymax2 = max(signal) + 0.5
            if max(signal) > 1.5:
                ymax2 = 1.5

            ymin1 = min(signal)
            if min(signal) < -1:
                ymin1 = -1

            ymin2 = min(signal) - 0.5
            if min(signal) < -1.5:
                ymin2 = -1.5

            # if not symbols_ref is None:
            #     [plt.annotate(symbols_ref[i], (samples_ref[i], ymax1), color='r') for i in range(len(symbols_ref)) if symbols_ref[i] in ["S", "V", "A"]]
            #     [plt.annotate(symbols_ref[i], (samples_ref[i], ymax2), color='g') for i in range(len(symbols_ref))]

            plt.plot(cali_samples, signal[cali_samples], 'ko')
            plt.plot(samples, signal[samples], 'r*')
            [plt.annotate(symbols_ref[i], (samples_ref[i], ymin1), color='g') for i in range(len(symbols_ref)) if
             symbols_ref[i] in ["V", "S"]]
            [plt.annotate(svm_symbols[i], (samples[i], (ymin1 + ymin2) / 2), color='m') for i in range(len(svm_symbols))
             if svm_symbols[i] in ["V", "S"]]
            # [plt.annotate(symbols_predict[i], (sample_predict[i], ymin2), color='b') for i in range(len(symbols_predict))]

            if self.file_name is not None:
                plt.title(str(self.file_name))

            plt.show()
            plt.close()

        return np.asarray(samples), np.asarray(svm_symbols), np.asarray(list_cali_sample)

    def process(self):
        try:
            record = wfdb.rdrecord(self.file_path + str(self.file_name))
            if str(self.file_name) in ['114', '8204']:
                ecg_raw = record.p_signal[:, 1]
            else:
                ecg_raw = record.p_signal[:, 0]

            ann = wfdb.rdann(self.file_path + str(self.file_name), self.ext)
            samples, symbols = beat_annotations(ann)

            result = dict()
            result[self.file_name] = dict()
            result[self.file_name]['Total_beat'] = len(samples)
            result[self.file_name]['Total_PVC'] = len(
                np.flatnonzero((symbols == 'V') | (symbols == 'E') | (symbols == '!') | (symbols == 'r')))
            # print('Total_PVC: ', result[namefile]['Total_PVC'])

            ecg_raw[np.isnan(ecg_raw)] = 0
            if record.fs != self.fs:
                self.ecg, _ = wfdb.resample_sig(ecg_raw, record.fs, self.fs)
                samples = samples * self.fs // record.fs
            else:
                self.ecg = ecg_raw

            # samples_PT, symbols_pt = self.pan_tompkins(self.ecg, fs=self.fs)
            _samples_pt, symbols_pt = self.pan_tompkins_2(self.ecg, self.fs)
            samples_pt, symbols_pt, samples_pt_cali = self.svm_classification(self.ecg, self.fs,
                                                                              np.asarray(_samples_pt),
                                                                              samples, symbols)

            from algs import shannon_rt
            samples_pt_cali, rhythms = (shannon_rt.Shannon(record=self.ecg)
                                        .detect(samples_pt_cali, symbols_pt,
                                                sampling_rate=self.fs, ref=False))

            if record.fs != cf.SAMPLING_RATE:
                samples_PT = samples_pt * record.fs // self.fs
                samples_pt_cali = samples_pt_cali * record.fs // self.fs

            # rhythms = np.asarray(['(AFIB' if i == 1 else '(N' for i in rhythms])

            rhythm = []
            samples_pt_cali_rhythms = []
            symbol_pt_cali_rhythms = []
            for i in range(len(rhythms)):

                if i == 0:
                    # rhythm.append('(AFIB' if rhythms[i] == 1 else '(N')
                    if rhythms[i] == 1:
                        samp = samples_pt_cali[i] - 10 if samples_pt_cali[i] - 10 else samples_pt_cali[i] - 1
                        samples_pt_cali_rhythms.extend([samp, samples_pt_cali[i]])
                        symbol_pt_cali_rhythms.extend(['+', symbols_pt[i]])
                        rhythm.extend(['(AFIB', ''])

                    else:
                        samples_pt_cali_rhythms.append(samples_pt_cali[0])
                        symbol_pt_cali_rhythms.append(symbols_pt[i])
                        rhythm.append('')
                else:
                    # rhythm.append('(AFIB' if rhythms[i] == 1 and rhythms[i - 1] != 1 else '(N' if rhythms[i] == 0 and rhythms[i - 1] != 0 else '')
                    if rhythms[i] == 1 and rhythms[i-1] != 1:
                        samp = samples_pt_cali[i] - 10 if samples_pt_cali[i] - 10 else samples_pt_cali[i] - 1
                        samples_pt_cali_rhythms.extend([samp, samples_pt_cali[i]])
                        symbol_pt_cali_rhythms.extend(['+', symbols_pt[i]])
                        rhythm.extend(['(AFIB', ''])

                    elif rhythms[i] == 0 and rhythms[i - 1] != 0:
                        samples_pt_cali_rhythms.append(samples_pt_cali[i])
                        symbol_pt_cali_rhythms.append(symbols_pt[i])
                        rhythm.append('(N')
                    else:
                        samples_pt_cali_rhythms.append(samples_pt_cali[i])
                        symbol_pt_cali_rhythms.append(symbols_pt[i])
                        rhythm.append('')

            annotation = wfdb.Annotation(
                record_name=str(self.file_name),
                extension='pt',
                sample=np.asarray(samples_pt_cali_rhythms, dtype=int),
                symbol=np.asarray(symbol_pt_cali_rhythms),
                aux_note=np.asarray(rhythm),
                fs=ann.fs
            )
            annotation.wrann(write_fs=True, write_dir=self.file_path)

            annotation = wfdb.Annotation(
                record_name=str(self.file_name),
                extension='ptorg',
                # aux_note=rhythms,
                sample=np.asarray(samples_pt_cali, dtype=int),
                symbol=np.asarray(symbols_pt),
                aux_note=[''] * len(symbols_pt),
                fs=ann.fs
            )
            annotation.wrann(write_fs=True, write_dir=self.file_path)

            aux_note = list(map(lambda x: '(AFIB' if x else '(N', rhythms))

            annotation = wfdb.Annotation(
                record_name=str(self.file_name),
                extension='af',
                aux_note=aux_note,
                sample=np.asarray(samples_pt_cali, dtype=int),
                symbol=np.asarray(['+'] * len(rhythms)),
                # aux_note=[''] * len(symbols_pt),
                fs=ann.fs
            )
            annotation.wrann(write_fs=True, write_dir=self.file_path)

            # annotation = wfdb.Annotation(
            #     record_name=str(self.file_name),
            #     extension='afcali',
            #     # aux_note=rhythms,
            #     sample=np.asarray(samples_pt_cali, dtype=int),
            #     symbol=np.asarray(['+'] * len(symbols_pt)),
            #     aux_note=[''] * len(symbols_pt),
            #     fs=ann.fs
            # )
            # annotation.wrann(write_fs=True, write_dir=self.file_path)

        except (Exception,) as e:
            print(f'{self.file_name}: {e}')
