import os
import numpy as np
from datetime import datetime

HES_SAMPLING_RATE = 200
HOUR_IN_DAY = 24
SECOND_IN_MINUTE = 60
MINUTE_IN_HOUR = 60
MINUTE_IN_DAY = HOUR_IN_DAY * MINUTE_IN_HOUR
SECOND_IN_HOUR = SECOND_IN_MINUTE * MINUTE_IN_HOUR
SECOND_IN_DAY = SECOND_IN_MINUTE * MINUTE_IN_DAY
CHANNEL = 1
CHANNEL_COLUMN = ['CH1', 'CH2', 'CH3', 'CH4']
EVENT_LEN = 3
BEATS_EXTEND = 7

NOISE = 0x0001
BRADY = 0x0002
TACHY = 0x0004
AFIB = 0x0800
PAUSE = 0x0100

NOR_TYPE = 1
VES_TYPE = 60
SVES_TYPE = 70

SINGLE_SVE = 1
COUPLET_SVE = 2
SVE_RUN = 4
SVE_BIGEMINY = 15
SVE_TRIGEMINAL = 16

SINGLE_VES = 5
COUPLET_VES = 6
VES_RUN = 8
VES_BIGEMINY = 9
VES_TRIGEMINAL = 10

HOLTER_SINGLE_VES = 1
HOLTER_VES_RUN = 2
HOLTER_VES_COUPLET = 4
HOLTER_VES_BIGEMINY = 8
HOLTER_VES_TRIGEMINAL = 16
HOLTER_SINGLE_SVES = 32
HOLTER_SVES_RUN = 64
HOLTER_SVES_COUPLET = 128
HOLTER_SVES_BIGEMINY = 256
HOLTER_SVES_TRIGEMINAL = 512
NORMAL = 0
HOLTER_SV_EVENT = np.asarray([HOLTER_SINGLE_SVES, HOLTER_SVES_COUPLET, HOLTER_SVES_RUN, HOLTER_SVES_BIGEMINY,
                              HOLTER_SVES_TRIGEMINAL, HOLTER_SINGLE_VES, HOLTER_VES_COUPLET, HOLTER_VES_RUN,
                              HOLTER_VES_BIGEMINY,
                              HOLTER_VES_TRIGEMINAL, NORMAL])
SV_EVENT = np.asarray([SINGLE_SVE, COUPLET_SVE, SVE_RUN, SVE_BIGEMINY, SVE_TRIGEMINAL, SINGLE_VES, COUPLET_VES, VES_RUN,
                       VES_BIGEMINY, VES_TRIGEMINAL, NORMAL])

HOLTER_BRADY = 1024
HOLTER_TACHY = 2048
HOLTER_MAX_HR = 4096
HOLTER_MIN_HR = 8192
HOLTER_LONG_RR = 16384
HOLTER_PAUSE = 32768
HOLTER_AFIB = 65536
HOLTER_ARTIFACT = 131072
HOLTER_USER_EVENTS = 262144
HOLTER_CUSTOM_EVENTS = 524288
HOLTER_SINUS_ARRHYTHMIA = 1048576

HOLTER_EVENT_NAME = np.asarray(["Single_VES", "VES_Run", "VES_Couplet", "VES_Bigeminy", "VES_trigeminal", "Single_SVES",
                                "SVES_run", "SVES_Couplet", "SVES_Bigeminy", "SVES_Trigeminal", "Brady", "Tachy",
                                "Max_HR", "Min_HR", "Long_RR", "PAUSE", "AFIB", "ARTIFACT", "USER_EVENTS",
                                "CUSTOM_EVENTS"])

SV_EVENT_SUM = HOLTER_SINGLE_VES + HOLTER_VES_RUN + HOLTER_VES_COUPLET + HOLTER_VES_BIGEMINY + HOLTER_VES_TRIGEMINAL + \
               HOLTER_SINGLE_SVES + HOLTER_SVES_RUN + HOLTER_SVES_COUPLET + HOLTER_SVES_BIGEMINY + \
               HOLTER_SVES_TRIGEMINAL
HOLTER_EVENT_SUM = SV_EVENT_SUM + HOLTER_BRADY + HOLTER_TACHY + HOLTER_MAX_HR + HOLTER_MIN_HR + HOLTER_LONG_RR + \
            HOLTER_PAUSE + HOLTER_AFIB + HOLTER_ARTIFACT + HOLTER_USER_EVENTS + HOLTER_CUSTOM_EVENTS

EVENT_SUM_EXCLUDE_ARTIFACT = SV_EVENT_SUM + HOLTER_BRADY + HOLTER_TACHY + HOLTER_MAX_HR + HOLTER_MIN_HR + HOLTER_LONG_RR + \
            HOLTER_PAUSE + HOLTER_AFIB + HOLTER_USER_EVENTS + HOLTER_CUSTOM_EVENTS

EVENT_SUM_EXCLUDE_MIN_MAX_LONG_RR = SV_EVENT_SUM + HOLTER_BRADY + HOLTER_TACHY + HOLTER_PAUSE + HOLTER_AFIB + \
                           HOLTER_ARTIFACT + HOLTER_USER_EVENTS + HOLTER_CUSTOM_EVENTS

STUDY_DATA = np.asarray(['EPOCH', 'BEAT', 'BEAT_TYPE', 'EVENT', 'SV_EVENT', 'SV_DURATION', 'QTC', 'ST_LEVEL',
                         'ST_SLOPE', 'QT', 'T_PEAK', 'FILE_INDEX'])

MINUTE_DATA = np.asarray(['HR', 'VES', 'SVES', 'SINGLE_VES', 'VE_COUPLET', 'VE_BIGEMINY', 'VE_TRIGEMINAL', 'VE_RUN',
                          'SINGLE_SVES', 'SVE_COUPLET', 'SVE_BIGEMINY', 'SVE_TRIGEMINAL', 'SVE_RUN', 'LONGRR',
                          'ST_SLOPE', 'QT', 'QTC', 'ST_LEVEL', 'MEANRR', 'HRV'])

HOUR_DATA = np.asarray(['VES_HOUR', 'SVES_HOUR', 'HR_AVG_HOUR', 'HR_MAX_HOUR', 'HR_MIN_HOUR',
                        'HRV_HOUR_SDNN', 'HRV_HOUR_SDANN', 'HRV_HOUR_RMSSD', 'HRV_HOUR_PNN50', 'HRV_HOUR_SDNN_INDX',
                        'HRV_HOUR_MSD', 'HRV_HOUR_MEAN_RR', 'T_PEAK_MAX_HOUR', 'T_PEAK_MEAN_HOUR', 'T_PEAK_MIN_HOUR',
                        'T_BEATS', 'QT_MAX', 'QT_MIN', 'QT_AVG', 'QTC_MAX', 'QTC_MIN', 'QTC_AVG'])

DAY_HOUR = 8
NIGHT_HOUR = 20

NOISE_FLAG = True
RECORD_DIR = '/media/quangpc/AI/portal_data/study_HRV/'
STUDY_NAME = 'study-6554'

MAX_NUM_STRIP_EACH_EVENT = 3
MAX_NUM_STRIP_SINUS_EVENT = 2

def study_data_column(event):
    column = np.where(STUDY_DATA == event)
    if len(column) > 0:
        return column[0][0]
    else:
        return 0


def study_data_holter_column(event):
    column = np.where(STUDY_DATA == event)
    if len(column) > 0:
        return column[0][0]
    else:
        return 0


def minute_data_column(event):
    column = np.where(MINUTE_DATA == event)
    if len(column) > 0:
        return column[0][0]
    else:
        return 0


def hour_data_column(event):
    column = np.where(HOUR_DATA == event)
    if len(column) > 0:
        return column[0][0]
    else:
        return 0


def epoch2datetime(epoch_time, start_time_of_study):
    """

    :param epoch_time:
    :param start_time_of_study:
    :return:
    """
    pass2zero = datetime.fromtimestamp(start_time_of_study).hour * 60 * 60 + \
                datetime.fromtimestamp(start_time_of_study).minute * 60 + \
                datetime.fromtimestamp(start_time_of_study).second
    day = int((epoch_time - (start_time_of_study - pass2zero)) / 86400) + 1
    return '{} ({})'.format(datetime.fromtimestamp(epoch_time).timetz(), day)


def find_index_noise_convert_study_holter(study_data_holter):
    """

    :param study_data_holter:
    :return:
    """
    event_noise = study_data_holter[:, study_data_holter_column('EVENT')] & HOLTER_ARTIFACT
    index_event_noise = np.where(event_noise == HOLTER_ARTIFACT)[0]
    if len(index_event_noise) > 0:
        if index_event_noise[0] == 0:
            index_event_noise[0] = 1
            index_event_noise[1] = 2

        index_event_noise = np.reshape(index_event_noise, (-1, 2))
        re_index_event_noise = index_event_noise[:, 0] - 1
        index_event_noise = np.concatenate((np.reshape(re_index_event_noise, (-1, 1)), index_event_noise), axis=1)
        index_event_noise = index_event_noise.flatten()
    return index_event_noise


def find_index_noise(study_data_holter):
    """

    :param study_data_holter:
    :return:
    """
    event_noise = study_data_holter[:, study_data_holter_column('EVENT')] & HOLTER_ARTIFACT
    index_event_noise = np.where(event_noise == HOLTER_ARTIFACT)[0]
    return index_event_noise


def find_index_file_transfer(study_data_holter):
    """

    :param study_data_holter:
    :return:
    """
    file_index = study_data_holter[:, study_data_holter_column('FILE_INDEX')]
    group_index = np.where(abs(np.diff(file_index)) == 1)[0]
    if len(group_index) > 0:
        group_index = np.concatenate((np.reshape(group_index, (-1, 1)), np.reshape(group_index + 1, (-1, 1))), axis=1)
        group_index = group_index.flatten()
    return group_index


def find_study_holter_data_no_index_noise(study_data):
    if NOISE_FLAG:
        index_study_data = np.arange(0, len(study_data[:, 0] - 1))
        index_event_noise = find_index_noise(study_data)
        index_event_noise_select_rr = np.arange(0, len(index_event_noise), 2)
        index_study_data_no_noise = np.setdiff1d(index_study_data, index_event_noise[index_event_noise_select_rr])
        study_data = study_data[index_study_data_no_noise]
        return study_data
    else:
        return study_data
