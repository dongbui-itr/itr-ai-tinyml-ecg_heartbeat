import re
import operator

import numpy as np


rhythms_datastore = {
    'feature_len': 2560,
    'sampling_rate': 256,
    'rhythm_class': {
        'OTHER': 0,
        'SINUS': 1,
        'AFIB': 2,
        'SVT': 3,
        'VT': 4,
        'AVB1': 5,
        'AVB2': 6,
        'AVB3': 7
    },
    'bwr': False,
    'norm': False,
    'clip01': False
}


def post_processing_beats_and_rhythms(samples, symbols, rhythms, rhythm_classification_datastore=None):
    if rhythm_classification_datastore is None:
        rhythm_classification_datastore = rhythms_datastore
    sampling_rate = rhythm_classification_datastore["sampling_rate"]
    rhythm_classification_class = rhythm_classification_datastore["rhythm_class"]
    rhythm_main = 'OTHER'
    rhythm_sub = ''
    total_rhythm_predict = rhythms
    afib_label = rhythm_classification_datastore['rhythm_class']['AFIB']
    if np.count_nonzero(total_rhythm_predict):
        # region Remove short rhythm
        group_index = np.flatnonzero(np.abs(np.diff(total_rhythm_predict)) != 0) + 1
        group_split = np.split(total_rhythm_predict, group_index)
        valid_length = 3
        pre_long_rhythm = -1
        if len(group_split) >= 2:
            for i in range(len(group_split) + 1):
                if i == len(group_split) or len(group_split[i]) > valid_length:
                    post_long_rhythm = i
                    for j in range(pre_long_rhythm + 1, post_long_rhythm):
                        if pre_long_rhythm == -1:
                            group_split[j][:] = group_split[post_long_rhythm][0]
                        elif post_long_rhythm == len(group_split):
                            group_split[j][:] = group_split[pre_long_rhythm][0]
                        elif len(group_split[pre_long_rhythm]) > len(group_split[post_long_rhythm]):
                            group_split[j][:] = group_split[pre_long_rhythm][0]
                        else:
                            group_split[j][:] = group_split[post_long_rhythm][0]
                    pre_long_rhythm = i
            total_rhythm_predict = np.concatenate(group_split)
        # endregion Remove short rhythm

        # region Finding main rhythm
        group_index = np.flatnonzero(np.abs(np.diff(total_rhythm_predict)) > 0) + 1
        group_rhythm = np.split(total_rhythm_predict, group_index)
        rhythm_invert = {i: k for k, i in rhythm_classification_class.items()}
        rhythm_len = dict()
        for i in range(len(group_rhythm)):
            for k in rhythm_classification_class.keys():
                if k not in rhythm_len and k == rhythm_invert[group_rhythm[i][0]]:
                    rhythm_len[k] = len(group_rhythm[i])
                elif k == rhythm_invert[group_rhythm[i][0]]:
                    rhythm_len[k] += len(group_rhythm[i])
        rhythm_main = max(rhythm_len.items(), key=operator.itemgetter(1))[0]
        # endregion Finding main rhythm

        if afib_label in rhythms:
            # region Event <= 5s between AFIB will be AFIB to make possibly longest AFIB
            if rhythm_main == "AFIB":
                group_index = np.flatnonzero(np.abs(np.diff(total_rhythm_predict == 2)) > 0) + 1
                group_afib = np.split(total_rhythm_predict, group_index)
                if len(group_afib) >= 3:
                    for i in range(1, len(group_afib) - 1):
                        if (
                                len(group_afib[i]) <= 5
                                and group_afib[i - 1][0] == 2
                                and group_afib[i + 1][0] == 2
                        ):
                            group_afib[i][:] = 2

                    total_rhythm_predict = np.concatenate(group_afib)
                    group_index = np.flatnonzero(np.abs(np.diff(total_rhythm_predict)) > 0) + 1
                    group_rhythm = np.split(total_rhythm_predict, group_index)
                    rhythm_len = dict()
                    for i in range(len(group_rhythm)):
                        for k in rhythm_classification_class.keys():
                            if k not in rhythm_len and k == rhythm_invert[group_rhythm[i][0]]:
                                rhythm_len[k] = len(group_rhythm[i])
                            elif k == rhythm_invert[group_rhythm[i][0]]:
                                rhythm_len[k] += len(group_rhythm[i])
            # endregion Event <= 5s between AFIB will be AFIB to make possibly longest AFIB

            # region Remove SVT sticks AFIB
            group_index = np.flatnonzero(np.abs(np.diff(total_rhythm_predict)) != 0) + 1
            group_split = np.split(total_rhythm_predict, group_index)
            if len(group_split) >= 2:
                for i in range(len(group_split)):
                    if group_split[i][0] == 3:
                        if (
                                (i > 0 and group_split[i - 1][0] == 2)
                                or (i < len(group_split) - 1 and group_split[i + 1][0] == 2)
                        ):
                            group_split[i][:] = 2
                total_rhythm_predict = np.concatenate(group_split)
                rhythms = total_rhythm_predict
            # endregion Remove SVT sticks AFIB

        # region Finding sub rhythm
        # OTHER is not sub-rhythm, try to remove OTHER and get the next
        rhythm_main_len = rhythm_len.pop(rhythm_main, None)
        rhythm_len.pop('OTHER', None)
        if len(rhythm_len.keys()) > 0:
            temp_rhythm_sub = max(rhythm_len.items(), key=operator.itemgetter(1))[0]
            if rhythm_main == 'OTHER' and rhythm_len.get(temp_rhythm_sub, 0) == rhythm_main_len:
                rhythm_main = temp_rhythm_sub
                rhythm_len.pop(rhythm_main, None)
                if len(rhythm_len.keys()) > 0:
                    rhythm_sub = max(rhythm_len.items(), key=operator.itemgetter(1))[0]
            else:
                rhythm_sub = temp_rhythm_sub
        # endregion Finding sub rhythm

        # region Check S/V run in SVT/VT
        try:
            sinus_label = rhythm_classification_datastore['rhythm_class']['SINUS']
            tachy_labels = {
                rhythm_classification_datastore['rhythm_class']['SVT']: 'A',
                rhythm_classification_datastore['rhythm_class']['VT']: 'V'
            }
            for tachy_label, symbol in tachy_labels.items():
                pattern_run = '{}{}{}'.format(symbol, symbol, symbol)
                if tachy_label in rhythms:
                    group_index = np.flatnonzero(np.abs(np.diff(rhythms)) != 0) + 1
                    group_split = np.split(rhythms, group_index)
                    accumulate_len = 0
                    for i in range(len(group_split)):
                        if group_split[i][0] == tachy_label:
                            rhythm_samples = (np.arange(len(group_split[i]) * sampling_rate)
                                              + (accumulate_len * sampling_rate))
                            beats_in_group_index = np.flatnonzero(np.isin(samples, rhythm_samples))
                            symbols_in_group = symbols[beats_in_group_index]
                            samples_in_group = samples[beats_in_group_index]
                            string_symbols = ''.join(list(symbols_in_group))
                            index_run = [m.start() for m in re.finditer('(?={})'.format(pattern_run), string_symbols)]
                            if len(index_run) == 0:
                                group_split[i][:] = sinus_label
                            else:
                                index_run_temp = []
                                for index in index_run:
                                    stop_index = string_symbols[index:].find('NNN')
                                    if stop_index == -1:
                                        stop_index = len(string_symbols[index:])
                                    index_run_temp += [index + i for i in range(stop_index)]
                                index_run = list(set(index_run_temp))
                                if index_run:
                                    beats_run = samples_in_group[index_run]
                                    for j in range(len(group_split[i])):
                                        rhythm_samples = (np.arange(sampling_rate)
                                                          + (accumulate_len + j) * sampling_rate)
                                        if not np.count_nonzero(np.isin(beats_run, rhythm_samples)):
                                            group_split[i][j] = sinus_label

                        accumulate_len += len(group_split[i])
                    rhythms = np.concatenate(group_split)
        except Exception as error:
            print('Check S/V run in SVT/VT: {}'.format(error))
        # endregion Check S/V run in SVT/VT

        if afib_label in rhythms:
            # region Check AFib contain S
            try:
                sinus_label = rhythm_classification_datastore['rhythm_class']['SINUS']
                samples_s_index = np.flatnonzero(symbols == 'A')
                samples_s = samples[samples_s_index]
                if afib_label in rhythms:
                    group_index = np.flatnonzero(np.abs(np.diff(rhythms)) != 0) + 1
                    group_split = np.split(rhythms, group_index)
                    accumulate_len = 0
                    for i in range(len(group_split)):
                        if group_split[i][0] == afib_label:
                            rhythm_samples = (np.arange(len(group_split[i]) * sampling_rate)
                                              + (accumulate_len * sampling_rate))
                            if not np.count_nonzero(np.isin(samples_s, rhythm_samples)):
                                group_split[i][:] = sinus_label
                        accumulate_len += len(group_split[i])
            except Exception as error:
                print('AFib contain S: {}'.format(error))
            # endregion Check AFib contain S

            # region Remove S in AFib
            try:
                rhythms_frame = rhythms[:, None] + np.zeros(sampling_rate, dtype=int)
                rhythms_frame = rhythms_frame.flatten()
                valid_index = np.flatnonzero(samples < len(rhythms_frame))
                events = rhythms_frame[samples[valid_index]]

                change_index = np.flatnonzero(np.logical_and(symbols == 'A', events == 2))
                if len(symbols):
                    symbols[change_index] = 'N'
            except Exception as error:
                print('S in AFib: {}'.format(error))
            # endregion Remove S in AFib

    return {
        'samples': samples,
        'symbols': symbols,
        'rhythms': rhythms,
        'main_rhythm': rhythm_main,
        'sub_rhythm': rhythm_sub
    }
