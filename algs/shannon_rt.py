import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from collections import Counter
from itertools import chain

from scipy.signal import medfilt

from algs import config_qrs as cf


class Shannon:
    def __init__(
            self, record, #annotation, record
    ):
        self.sampling_rate = cf.SAMPLING_RATE

        # self.delay = 62
        self.len_episode = cf.NUM_AFIB_BEAT  # 63  # 127
        self.valid_length = 3 * self.sampling_rate

        self.thresholds = cf.THR_SHANNON #0.51  # 0.416
        # self.annotation = annotation
        self.record = record

    @staticmethod
    def median_filter(
            x,
            k
    ):
        """Apply a length-k median filter to a 1D array x.

        Boundaries are extended by repeating endpoints.

        """

        lim = k // 2
        y = []
        for i in range(lim, len(x)-lim):
            m = np.median(x[i-lim:i+lim])
            y.extend([m])
        y = np.asarray(y)
        return y

    @staticmethod
    def low_reference_filter(
            x,
            filter_type
    ):
        a = np.append(np.array([1, -1]), np.zeros(15))  # mau so
        b = np.append(1, np.zeros(15))  # tu so
        b = np.append(b, -1)

        y = None
        if filter_type == 'lfilter':
            y = signal.lfilter(b, a, x).astype(np.int)
            y = y >> 4
        elif filter_type == 'filtfilt':
            y = signal.filtfilt(b, a, x)

        return y

    @staticmethod
    def high_reference_filter(
            x,
            filter_type
    ):

        a = np.array([1, -2, 1])  # mau so
        b = np.append(1, np.zeros(96))  # tu so
        b[32] = -1
        b[64] = -1
        b[96] = 1

        y = None
        if filter_type == 'lfilter':
            y = signal.lfilter(b, a, x).astype(np.int)
            y = y >> 11
        elif filter_type == 'filtfilt':
            y = signal.filtfilt(b, a, x)

        return y

    @staticmethod
    def symbolic_dynamic(
            hr,
            hr_max=315,
            quanti_step=6
    ):
        sy = hr // quanti_step
        sy[sy >= (hr_max//quanti_step)] = hr_max//quanti_step

        return sy

    @staticmethod
    def word_sequence(
            sy
    ):

        wv = np.convolve(sy, [2**12, 2**6, 1], mode='valid')

        return wv

    @staticmethod
    def shannon_entropy(
            list_seg
    ):
        entropy = np.zeros(list_seg.shape[0])
        N = list_seg.shape[1]
        for i in range(0, list_seg.shape[0]):
            unique = np.unique(list_seg[i].astype(int), return_counts=True)
            k = len(unique[0])
            A = unique[1]

            # k = len(Counter(list_seg[i]))
            # A = np.histogram(list_seg[i], bins=int(k))[0]

            P = A / N
            log2N = np.log2(N)
            if np.isnan(log2N):
                log2N = 1

            entropy[i] = - ((k / (N * log2N)) * np.sum(P * np.nan_to_num(np.log2(P))))

        return entropy

    def post_process(
            self,
            beats,
            prediction
    ):
        # region Remove short rhythm
        group_split = np.split(np.arange(len(prediction)), np.flatnonzero(np.abs(np.diff(prediction)) != 0) + 1)
        short_event = np.array(list(chain.from_iterable(map(
            lambda x: [int(prediction[x[-1]] or beats[x[-1]] - beats[x[0]] <= self.valid_length)] * len(x),
            group_split
        ))))

        group_split = np.split(np.arange(len(short_event)), np.flatnonzero(np.abs(np.diff(short_event)) != 0) + 1)
        prediction = np.array(list(chain.from_iterable(map(
            lambda x: [int(prediction[x[-1]] and beats[x[-1]] - beats[x[0]] >= self.valid_length)] * len(x),
            group_split
        ))))

        return prediction

    def detect(
            self,
            beats: np.ndarray,
            symbols: np.ndarray,
            sampling_rate: np.ndarray = cf.SAMPLING_RATE,
            **kwargs
    ):
        ref = kwargs.get('ref', None)
        title = kwargs.get('title', None)
        record = kwargs.get('record', None)

        _rr_sequence = np.diff(beats)
        # rr_sequence = medfilt(_rr_sequence, 3)
        # rr_sequence = np.convolve(_rr_sequence, np.ones(3)/3, mode='valid')
        # rr_sequence = np.convolve(_rr_sequence, np.ones(2)/2, mode='valid')
        hr_sequence = 60 * self.sampling_rate / _rr_sequence
        symbolic = self.symbolic_dynamic(hr_sequence)
        word_se = self.word_sequence(symbolic)
        frame = np.arange(0, self.len_episode, 1)[None, :] + np.arange(-self.len_episode//2, len(word_se) - self.len_episode//2 - 1, 1)[:, None]
        frame[frame < 0] = 0
        frame[frame >= len(word_se)] = len(word_se) - 1
        list_seg = word_se[frame]
        symbol_frame = symbols[symbols.shape[0] - word_se.shape[0]:][frame]
        sample_frame = beats[symbols.shape[0] - word_se.shape[0]:][frame]

        entropy = self.shannon_entropy(list_seg)
        prediction = np.zeros(len(beats), dtype=int)
        # start = self.len_episode // 2  # // 2
        start = 0  # // 2
        cnt = 0
        for i_indx, i_entropy in enumerate(entropy):
            if i_entropy > self.thresholds:# and len(np.flatnonzero(symbol_frame[i_indx] == 'V')) < 5:
                indx_v = np.flatnonzero(np.in1d(symbol_frame[i_indx],  ['V']) == True)
                indx_v_diff = np.diff(indx_v)
                indx_v_diff_2 = np.flatnonzero(np.in1d(indx_v_diff, [2, 3]) == True)

                # indx_s = np.flatnonzero(np.in1d(symbol_frame[i_indx], ['A', 'S']) == True)
                # indx_s_diff = np.diff(indx_s)
                # indx_s_diff_2 = np.flatnonzero(np.in1d(indx_s_diff, [2, 3]) == True)


                if len(indx_v_diff_2) <= 4:# and len(indx_s_diff_2) <= 6:
                    prediction[frame[i_indx]] |= 1


        prediction_bk_1 = prediction.copy()

        # indx = np.flatnonzero(prediction == 0)
        # indx_diff = np.diff(indx)
        # indx_remove = np.flatnonzero((indx_diff > 1))
        # for i_indx_remove in indx_remove:
        #     if indx_diff[i_indx_remove] < 20:
        #         prediction[indx[i_indx_remove]: indx[i_indx_remove + 1]] = 1

        prediction_bk_2 = prediction.copy()

        # indx = np.flatnonzero(prediction == 1)
        # indx_diff = np.diff(indx)
        # indx_remove = np.flatnonzero((indx_diff > 1))
        # for i_indx_remove in indx_remove:
        #     if indx_diff[i_indx_remove] < 10:
        #         prediction[indx[i_indx_remove]: indx[i_indx_remove + 1]] = 0

        prediction_bk_3 = prediction.copy()

        # indx_wordse = np.flatnonzero(word_se >= cf.THR_SHANNON_WORDSE)
        # indx_shannon = np.flatnonzero(entropy[indx_wordse] >= self.thresholds)
        # if len(indx_shannon) > 0:
        #     prediction[indx_wordse[indx_shannon]] = 1
        #     ind_0 = np.in1d(prediction, [0])
        #     ind_0_true = np.flatnonzero(ind_0==False)
        #     ind_0_true_diff = np.diff(ind_0_true)
        #     ind_0_true_diff_short = np.flatnonzero((ind_0_true_diff < 30) & (ind_0_true_diff > 1))
        #     if len(ind_0_true_diff_short) > 0:
        #         for i in ind_0_true_diff_short:
        #             prediction[ind_0_true[i]:ind_0_true[i+1]] = 1

        ind_0 = np.in1d(prediction, [1])
        ind_0_true = np.flatnonzero(ind_0 == False)
        ind_0_true_diff = np.diff(ind_0_true)
        ind_0_true_diff_short = np.flatnonzero((ind_0_true_diff < 15) & (ind_0_true_diff > 1))
        if len(ind_0_true_diff_short) > 0:
            for i in ind_0_true_diff_short:
                prediction[ind_0_true[i]:ind_0_true[i + 1]] = 0


            # for i in indx_wordse[indx_shannon]:
            #     prediction[frame[i]] = 1
            #     a=10

        # prediction[start:start+len(entropy)] = np.where(entropy >= self.thresholds, 1, 0)
        # prediction[start:start+len(entropy)] = np.where((entropy >= self.thresholds) & (entropy < 0.95), 1, 0)
        # prediction = self.post_process(beats, prediction)

        if ref:
            fig, axis = plt.subplots(5, 1, figsize=(12, 8), sharey=False, sharex='all')

            fig.suptitle(title)
            axis[0].set_title('{} - hr_sequence'.format(cf.FILE_NAME))
            axis[0].plot(beats[:len(hr_sequence)], hr_sequence)
            axis[1].set_title('symbolic')
            axis[1].plot(beats[:len(symbolic)], symbolic)
            axis[2].set_title('word_se')
            axis[2].plot(beats[:len(word_se)], word_se)
            axis[3].set_title('Shannon entropy')
            axis[3].plot(beats[start:start+len(entropy)], entropy)
            axis[3].plot(beats, prediction_bk_1, 'b')
            axis[3].plot(beats, prediction_bk_2 + 0.3, 'm')
            axis[3].plot(beats, prediction_bk_3 + 0.2, 'y')
            axis[3].plot(beats, prediction)
            axis[3].axhline(self.thresholds, color='red', linestyle='-', alpha=1)
            axis[4].plot(self.record[:, 0])
            axis[4].plot(beats, self.record[:, 0][beats], '*r')

            ref_rhythm = np.flatnonzero(np.array(self.annotation.aux_note) == '(AFIB')
            indx_non_afib = np.flatnonzero(np.in1d(np.array(self.annotation.aux_note), ['', '(AFIB']) == False)
            # ref_rhythm_end = [indx_non_afib[np.flatnonzero(indx_non_afib > u)[0]] for u in ref_rhythm if len(np.flatnonzero(indx_non_afib > u)) > 0]
            # ref_rhythm_end.append(len(self.record[:, 0]))

            # plt.show()

            # if len(ref_rhythm) > 0:
            if cf.FILE_NAME != '*':
                import os
                import wfdb as wf
                ann_atr = wf.rdann('/mnt/Dataset/ECG/PhysionetData/mitdb/{}'.format(cf.FILE_NAME), 'atr')
                if ann_atr.fs != cf.SAMPLING_RATE:
                    ann_atr.sample = ann_atr.sample * cf.SAMPLING_RATE // ann_atr.fs

                ref_rhythm = np.flatnonzero(np.array(ann_atr.aux_note) == '(AFIB')
                indx_non_afib = np.flatnonzero(np.in1d(np.array(ann_atr.aux_note), ['', '(AFIB']) == False)
                ref_rhythm_end = [indx_non_afib[np.flatnonzero(indx_non_afib > u)[0]] for u in ref_rhythm if len(np.flatnonzero(indx_non_afib > u)) > 0]
                if len(ref_rhythm) > len(ref_rhythm_end):
                    ref_rhythm_end.append(-1)

                for i, gr in enumerate(ref_rhythm):
                    axis[0].axvspan(xmin=ann_atr.sample[gr], xmax=ann_atr.sample[ref_rhythm_end[i]], color='g', alpha=0.5)
                    axis[1].axvspan(xmin=ann_atr.sample[gr], xmax=ann_atr.sample[ref_rhythm_end[i]], color='g', alpha=0.5)
                    axis[2].axvspan(xmin=ann_atr.sample[gr], xmax=ann_atr.sample[ref_rhythm_end[i]], color='g', alpha=0.5)
                    axis[3].axvspan(xmin=ann_atr.sample[gr], xmax=ann_atr.sample[ref_rhythm_end[i]], color='g', alpha=0.5)

            # if len(ref_rhythm) > 0:
            #     sample = np.array(self.annotation.sample)
            #     group = np.split(ref_rhythm, np.flatnonzero(np.diff(ref_rhythm) != 1) + 1)
            #     group = list(map(
            #         lambda y: [sample[y[0]], sample[y[-1] + 1] if y[-1] + 1 <= len(np.array(self.annotation.aux_note)) - 1 else record.sig_len],
            #         group))
            #     for gr in group:
            #         axis[0].axvspan(xmin=gr[0], xmax=gr[-1], color='k', alpha=0.8)
            #         axis[1].axvspan(xmin=gr[0], xmax=gr[-1], color='k', alpha=0.8)
            #         axis[3].axvspan(xmin=gr[0], xmax=gr[-1], color='k', alpha=0.8)

            plt.show()


        return beats, prediction #, entropy
