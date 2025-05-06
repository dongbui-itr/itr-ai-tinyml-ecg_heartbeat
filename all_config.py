import getpass
import os
from collections import OrderedDict

NUM_PROCESS = os.cpu_count()
DATASET_DAY = "2021-09-30"
DATASET_DISK = "MegaProject"
# DATASET_DISK_SECOND = "MegaProject"
EVENT_TYPE = "RhythmNet"
EVENT_EVAL_TYPE = "RhythmNetEval"
BEAT_TYPE = "BeatNet"
BEAT_EVAL_TYPE = "BeatNetEval"
ONLY_GET_COMMENT_EVENT = True
RHYTHM_TYPE_INDEX = "1"
BEAT_TYPE_INDEX = "2"

EXT_BEAT = "atr"
EXT_BEAT_EVAL = "atr"
EXT_EVENT = "rhy"
EXT_EVENT_EVAL = "rev"

MIN_RR_INTERVAL = 0.2 #sec

PORTAL_DATA = '/media/{}/{}/PortalData/AllEvents'.format(getpass.getuser(),
                                                         DATASET_DISK)

PORTAL_DATA_NO_COMMENT = '/media/{}/{}/PortalData/AllEvents-NoComment'.format(getpass.getuser(),
                                                                              DATASET_DISK)

DATA_PATH_EVENT = '/media/{}/{}/PortalData/{}/{}'.format(getpass.getuser(),
                                                         DATASET_DISK,
                                                         EVENT_TYPE,
                                                         DATASET_DAY)
DATA_PATH_EVENT_EVAL = '/media/{}/{}/PortalData/{}/{}'.format(getpass.getuser(),
                                                              DATASET_DISK,
                                                              EVENT_EVAL_TYPE,
                                                              DATASET_DAY)

DATA_PATH_BEAT = '/media/{}/{}/PortalData/{}/{}'.format(getpass.getuser(),
                                                        DATASET_DISK,
                                                        BEAT_TYPE,
                                                        DATASET_DAY)

DATA_PATH_BEAT_EVAL = '/media/{}/{}/PortalData/{}/{}'.format(getpass.getuser(),
                                                             DATASET_DISK,
                                                             BEAT_EVAL_TYPE,
                                                             DATASET_DAY)

NUM_EVENT_PER_FILE = 100
NUM_EVENT_PER_PERSON = {
    "bs1": 500,
    "bs2": 500,
    "bs3": 500,
}

TYPES_DATA = ['N', 'S', 'V', 'R', 'BRADY', 'TACHY', 'NOISE', 'AFIB']
# TYPES_DATA = ['V']

CLASS_TYPES = ['N', 'V', 'S', 'R', '|']

CLASS_WEIGHTS = {
        0: 1,
        1: 4,
        2: 4,
        3: 2,
    }

CLASS_WEIGHTS_RETRAIN = {
        0: 1,
        1: 3,
        2: 1,
    }

OVERLAB_IN_FILE = 3
OVERLAB = 1
OFFSET_FRAME_BEAT = [0, 5]

PATH_DATA_EC57 = '/mnt/Dataset//ECG/PhysionetData/'
DB_TESTING = [
        # ['mitdb', 'atr', 'atr'],
        ['nstdb', 'atr', 'atr'],
        # ['ahadb', 'atr', 'atr'],
        # ['escdb', 'atr', 'atr'],
        # ['afdb', 'qrs', 'atr'],
    ]


SAMP_FROM = 0
SAMP_TO = 0

FILE_NAME = '118e_6'
DEBUG = True

# FILE_NAME = '*'
# DEBUG = False

