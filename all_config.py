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

LABEL_RHYTHM_TYPES = OrderedDict(
    [
        ("0",
         OrderedDict([
             ("OTHER",
              [
                  "artifact/out_of_list",
              ]),
             ("SINUS",
              [
                  "sinus tachycardia/out_of_list",
                  "sinus bradycardia/out_of_list",
                  "sinus arrhythmia/out_of_list",
                  "sinus rhythm/out_of_list",
              ]),
             ("AFIB",
              [
                  "atrial fibrillation/atrial fibrillation",
                  "atrial fibrillation/atrial flutter",
                  "atrial fibrillation/out_of_list",

                  "atrial flutter/atrial fibrillation",
                  "atrial flutter/atrial flutter",
                  "atrial flutter/out_of_list",
              ]),
             ("SVT",
              [
                  "supraventricular tachycardia/out_of_list",
                  "supraventricular tachycardia/supraventricular tachycardia",
                  "paroxysmal supraventricular tachycardia/out_of_list",
                  "paroxysmal supraventricular tachycardia/paroxysmal supraventricular tachycardia",

                  "sinus tachycardia/supraventricular tachycardia",
                  "sinus tachycardia/paroxysmal supraventricular tachycardia",
                  "sinus bradycardia/supraventricular tachycardia",
                  "sinus bradycardia/paroxysmal supraventricular tachycardia",
                  "sinus arrhythmia/supraventricular tachycardia",
                  "sinus arrhythmia/paroxysmal supraventricular tachycardia",
                  "sinus rhythm/supraventricular tachycardia",
                  "sinus rhythm/paroxysmal supraventricular tachycardia",
              ]),
             ("VT",
              [
                  "ventricular tachycardia/out_of_list",
                  "ventricular tachycardia/ventricular tachycardia",

                  "sinus tachycardia/ventricular tachycardia",
                  "sinus bradycardia/ventricular tachycardia",
                  "sinus arrhythmia/ventricular tachycardia",
                  "sinus rhythm/ventricular tachycardia",
              ]),
             ("AVB1",
              [
                  "1st degree/out_of_list",
                  "1st degree/1st degree",

                  "sinus tachycardia/1st degree",
                  "sinus bradycardia/1st degree",
                  "sinus arrhythmia/1st degree",
                  "sinus rhythm/1st degree",
              ]),
             ("AVB2",
              [
                  "2nd degree/out_of_list",
                  "2nd degree/2nd degree",

                  "sinus tachycardia/2nd degree",
                  "sinus bradycardia/2nd degree",
                  "sinus arrhythmia/2nd degree",
                  "sinus rhythm/2nd degree",
              ]),
             ("AVB3",
              [
                  "3rd degree/out_of_list",
                  "3rd degree/3rd degree",

                  "advanced heart block/out_of_list",
                  "advanced heart block/advanced heart block",

                  "sinus tachycardia/3rd degree",
                  "sinus bradycardia/3rd degree",
                  "sinus arrhythmia/3rd degree",
                  "sinus rhythm/3rd degree",

                  "sinus tachycardia/3rd degree",
                  "sinus bradycardia/3rd degree",
                  "sinus arrhythmia/3rd degree",
                  "sinus rhythm/3rd degree",
              ]),
         ])),
        ("1",
         OrderedDict([
             ("OTHER",
              [
                  "artifact/out_of_list",
              ]),
             ("SINUS",
              [
                  "sinus tachycardia/out_of_list",
                  "sinus bradycardia/out_of_list",
                  "sinus arrhythmia/out_of_list",
                  "sinus rhythm/out_of_list",
              ]),
             ("AFIB",
              [
                  "sinus tachycardia/atrial fibrillation",
                  "sinus bradycardia/atrial fibrillation",
                  "sinus arrhythmia/atrial fibrillation",
                  "sinus rhythm/atrial fibrillation",

                  "sinus tachycardia/atrial flutter",
                  "sinus bradycardia/atrial flutter",
                  "sinus arrhythmia/atrial flutter",
                  "sinus rhythm/atrial flutter",

                  "atrial fibrillation/out_of_list",
                  "atrial flutter/out_of_list",
              ]),
             ("SVT",
              [
                  "sinus tachycardia/supraventricular tachycardia",
                  "sinus bradycardia/supraventricular tachycardia",
                  "sinus arrhythmia/supraventricular tachycardia",
                  "sinus rhythm/supraventricular tachycardia",

                  "sinus tachycardia/paroxysmal supraventricular tachycardia",
                  "sinus bradycardia/paroxysmal supraventricular tachycardia",
                  "sinus arrhythmia/paroxysmal supraventricular tachycardia",
                  "sinus rhythm/paroxysmal supraventricular tachycardia",

                  "supraventricular tachycardia/out_of_list",
                  "paroxysmal supraventricular tachycardia/out_of_list",
              ]),
             ("VT",
              [
                  "sinus tachycardia/ventricular tachycardia",
                  "sinus bradycardia/ventricular tachycardia",
                  "sinus arrhythmia/ventricular tachycardia",
                  "sinus rhythm/ventricular tachycardia",

                  "ventricular tachycardia/out_of_list",
              ]),
             ("AVB1",
              [
                  "sinus tachycardia/1st degree",
                  "sinus bradycardia/1st degree",
                  "sinus arrhythmia/1st degree",
                  "sinus rhythm/1st degree",

                  "1st degree/out_of_list",
                  "1st degree/1st degree",
              ]),
             ("AVB2",
              [
                  "sinus tachycardia/2nd degree",
                  "sinus bradycardia/2nd degree",
                  "sinus arrhythmia/2nd degree",
                  "sinus rhythm/2nd degree",

                  "2nd degree/out_of_list",
                  "2nd degree/2nd degree",
              ]),
             ("AVB3",
              [
                  "sinus tachycardia/3rd degree",
                  "sinus bradycardia/3rd degree",
                  "sinus arrhythmia/3rd degree",
                  "sinus rhythm/3rd degree",

                  "sinus tachycardia/advanced heart block",
                  "sinus bradycardia/advanced heart block",
                  "sinus arrhythmia/advanced heart block",
                  "sinus rhythm/advanced heart block",

                  "3rd degree/out_of_list",
                  "3rd degree/3rd degree",
                  "advanced heart block/out_of_list",
                  "advanced heart block/advanced heart block",
              ]),
         ])),
    ])

LABEL_BEAT_TYPES = OrderedDict(
    [
        ("0", OrderedDict([
            ("NOTABEAT", [
            ]),
            ("N", [
                "sinus tachycardia/out_of_list",
                "sinus arrhythmia/out_of_list",
            ]),
            ("A", [
                "sinus tachycardia/atrial run",
                "sinus tachycardia/atrial couplet",
                "sinus tachycardia/atrial bigeminy",
                "sinus tachycardia/atrial trigeminy",

                "sinus bradycardia/atrial run",
                "sinus bradycardia/atrial couplet",
                "sinus bradycardia/atrial bigeminy",
                "sinus bradycardia/atrial trigeminy",

                "sinus arrhythmia/atrial run",
                "sinus arrhythmia/atrial couplet",
                "sinus arrhythmia/atrial bigeminy",
                "sinus arrhythmia/atrial trigeminy",

                "sinus rhythm/atrial run",
                "sinus rhythm/atrial couplet",
                "sinus rhythm/atrial bigeminy",
                "sinus rhythm/atrial trigeminy",

                "sinus tachycardia/supraventricular tachycardia",
                "sinus bradycardia/supraventricular tachycardia",
                "sinus arrhythmia/supraventricular tachycardia",
                "sinus rhythm/supraventricular tachycardia",

                "sinus tachycardia/paroxysmal supraventricular tachycardia",
                "sinus bradycardia/paroxysmal supraventricular tachycardia",
                "sinus arrhythmia/paroxysmal supraventricular tachycardia",
                "sinus rhythm/paroxysmal supraventricular tachycardia",

                "sinus tachycardia/pac",
                "sinus bradycardia/pac",
                "sinus arrhythmia/pac",
                "sinus rhythm/pac",

                "supraventricular tachycardia/out_of_list"
            ]),
            ("V", [
                "sinus tachycardia/ventricular run",
                "sinus tachycardia/ventricular couplet",
                "sinus tachycardia/ventricular bigeminy",
                "sinus tachycardia/ventricular trigeminy",

                "sinus bradycardia/ventricular run",
                "sinus bradycardia/ventricular couplet",
                "sinus bradycardia/ventricular bigeminy",
                "sinus bradycardia/ventricular trigeminy",

                "sinus arrhythmia/ventricular run",
                "sinus arrhythmia/ventricular couplet",
                "sinus arrhythmia/ventricular bigeminy",
                "sinus arrhythmia/ventricular trigeminy",

                "sinus rhythm/ventricular run",
                "sinus rhythm/ventricular couplet",
                "sinus rhythm/ventricular bigeminy",
                "sinus rhythm/ventricular trigeminy",

                "sinus tachycardia/ventricular tachycardia",
                "sinus bradycardia/ventricular tachycardia",
                "sinus arrhythmia/ventricular tachycardia",
                "sinus rhythm/ventricular tachycardia",

                "sinus tachycardia/pvc",
                "sinus bradycardia/pvc",
                "sinus arrhythmia/pvc",
                "sinus rhythm/pvc",

                "ventricular tachycardia/out_of_list"
            ]),
            ("R", [
                "sinus tachycardia/ivcd",
                "sinus bradycardia/ivcd",
                "sinus arrhythmia/ivcd",
                "sinus rhythm/ivcd",
            ]),
        ])),
        ("1", OrderedDict([
            ("NOTABEAT", [
            ]),
            ("N", [
                "sinus arrhythmia/out_of_list",
                "sinus tachycardia/out_of_list",

                "sinus tachycardia/multi events",
                "sinus bradycardia/multi events",
                "sinus arrhythmia/multi events",
                "sinus rhythm/multi events",
            ]),
            ("A", [
                "sinus tachycardia/atrial run",
                "sinus tachycardia/atrial couplet",
                "sinus tachycardia/atrial bigeminy",
                "sinus tachycardia/atrial trigeminy",

                "sinus bradycardia/atrial run",
                "sinus bradycardia/atrial couplet",
                "sinus bradycardia/atrial bigeminy",
                "sinus bradycardia/atrial trigeminy",

                "sinus arrhythmia/atrial run",
                "sinus arrhythmia/atrial couplet",
                "sinus arrhythmia/atrial bigeminy",
                "sinus arrhythmia/atrial trigeminy",

                "sinus rhythm/atrial run",
                "sinus rhythm/atrial couplet",
                "sinus rhythm/atrial bigeminy",
                "sinus rhythm/atrial trigeminy",

                "sinus tachycardia/supraventricular tachycardia",
                "sinus bradycardia/supraventricular tachycardia",
                "sinus arrhythmia/supraventricular tachycardia",
                "sinus rhythm/supraventricular tachycardia",

                "sinus tachycardia/paroxysmal supraventricular tachycardia",
                "sinus bradycardia/paroxysmal supraventricular tachycardia",
                "sinus arrhythmia/paroxysmal supraventricular tachycardia",
                "sinus rhythm/paroxysmal supraventricular tachycardia",

                "sinus tachycardia/pac",
                "sinus bradycardia/pac",
                "sinus arrhythmia/pac",
                "sinus rhythm/pac",

                "supraventricular tachycardia/out_of_list"
            ]),
            ("V", [
                "sinus tachycardia/ventricular run",
                "sinus tachycardia/ventricular couplet",
                "sinus tachycardia/ventricular bigeminy",
                "sinus tachycardia/ventricular trigeminy",

                "sinus bradycardia/ventricular run",
                "sinus bradycardia/ventricular couplet",
                "sinus bradycardia/ventricular bigeminy",
                "sinus bradycardia/ventricular trigeminy",

                "sinus arrhythmia/ventricular run",
                "sinus arrhythmia/ventricular couplet",
                "sinus arrhythmia/ventricular bigeminy",
                "sinus arrhythmia/ventricular trigeminy",

                "sinus rhythm/ventricular run",
                "sinus rhythm/ventricular couplet",
                "sinus rhythm/ventricular bigeminy",
                "sinus rhythm/ventricular trigeminy",

                "sinus tachycardia/ventricular tachycardia",
                "sinus bradycardia/ventricular tachycardia",
                "sinus arrhythmia/ventricular tachycardia",
                "sinus rhythm/ventricular tachycardia",

                "sinus tachycardia/pvc",
                "sinus bradycardia/pvc",
                "sinus arrhythmia/pvc",
                "sinus rhythm/pvc",

                "ventricular tachycardia/out_of_list"
            ]),
            ("R", [
                "sinus tachycardia/ivcd",
                "sinus bradycardia/ivcd",
                "sinus arrhythmia/ivcd",
                "sinus rhythm/ivcd",
            ]),
            ("Q", [
            ]),
        ])),
        ("2", OrderedDict([
            ("NOTABEAT", [
            ]),
            ("ARTIFACT", [
            ]),
            ("N", [
                "sinus arrhythmia/out_of_list",
                "sinus tachycardia/out_of_list",

                "sinus tachycardia/multi events",
                "sinus bradycardia/multi events",
                "sinus arrhythmia/multi events",
                "sinus rhythm/multi events",
            ]),
            ("A", [
                "sinus tachycardia/atrial run",
                "sinus tachycardia/atrial couplet",
                "sinus tachycardia/atrial bigeminy",
                "sinus tachycardia/atrial trigeminy",

                "sinus bradycardia/atrial run",
                "sinus bradycardia/atrial couplet",
                "sinus bradycardia/atrial bigeminy",
                "sinus bradycardia/atrial trigeminy",

                "sinus arrhythmia/atrial run",
                "sinus arrhythmia/atrial couplet",
                "sinus arrhythmia/atrial bigeminy",
                "sinus arrhythmia/atrial trigeminy",

                "sinus rhythm/atrial run",
                "sinus rhythm/atrial couplet",
                "sinus rhythm/atrial bigeminy",
                "sinus rhythm/atrial trigeminy",

                "sinus tachycardia/supraventricular tachycardia",
                "sinus bradycardia/supraventricular tachycardia",
                "sinus arrhythmia/supraventricular tachycardia",
                "sinus rhythm/supraventricular tachycardia",

                "sinus tachycardia/paroxysmal supraventricular tachycardia",
                "sinus bradycardia/paroxysmal supraventricular tachycardia",
                "sinus arrhythmia/paroxysmal supraventricular tachycardia",
                "sinus rhythm/paroxysmal supraventricular tachycardia",

                "sinus tachycardia/pac",
                "sinus bradycardia/pac",
                "sinus arrhythmia/pac",
                "sinus rhythm/pac",

                "supraventricular tachycardia/out_of_list"
            ]),
            ("V", [
                "sinus tachycardia/ventricular run",
                "sinus tachycardia/ventricular couplet",
                "sinus tachycardia/ventricular bigeminy",
                "sinus tachycardia/ventricular trigeminy",

                "sinus bradycardia/ventricular run",
                "sinus bradycardia/ventricular couplet",
                "sinus bradycardia/ventricular bigeminy",
                "sinus bradycardia/ventricular trigeminy",

                "sinus arrhythmia/ventricular run",
                "sinus arrhythmia/ventricular couplet",
                "sinus arrhythmia/ventricular bigeminy",
                "sinus arrhythmia/ventricular trigeminy",

                "sinus rhythm/ventricular run",
                "sinus rhythm/ventricular couplet",
                "sinus rhythm/ventricular bigeminy",
                "sinus rhythm/ventricular trigeminy",

                "sinus tachycardia/ventricular tachycardia",
                "sinus bradycardia/ventricular tachycardia",
                "sinus arrhythmia/ventricular tachycardia",
                "sinus rhythm/ventricular tachycardia",

                "sinus tachycardia/pvc",
                "sinus bradycardia/pvc",
                "sinus arrhythmia/pvc",
                "sinus rhythm/pvc",

                "ventricular tachycardia/out_of_list"
            ]),
            ("R", [
                "sinus tachycardia/ivcd",
                "sinus bradycardia/ivcd",
                "sinus arrhythmia/ivcd",
                "sinus rhythm/ivcd",
            ]),
            ("Q", [
            ]),
        ])),
        ("3", OrderedDict([
            ("NOTABEAT", []),
            ("N", [
                "sinus arrhythmia/out_of_list",
                "sinus tachycardia/out_of_list",

                "sinus tachycardia/multi events",
                "sinus bradycardia/multi events",
                "sinus arrhythmia/multi events",
                "sinus rhythm/multi events",
            ]),
            ("S", [
                "sinus tachycardia/atrial run",
                "sinus tachycardia/atrial couplet",
                "sinus tachycardia/atrial bigeminy",
                "sinus tachycardia/atrial trigeminy",

                "sinus bradycardia/atrial run",
                "sinus bradycardia/atrial couplet",
                "sinus bradycardia/atrial bigeminy",
                "sinus bradycardia/atrial trigeminy",

                "sinus arrhythmia/atrial run",
                "sinus arrhythmia/atrial couplet",
                "sinus arrhythmia/atrial bigeminy",
                "sinus arrhythmia/atrial trigeminy",

                "sinus rhythm/atrial run",
                "sinus rhythm/atrial couplet",
                "sinus rhythm/atrial bigeminy",
                "sinus rhythm/atrial trigeminy",

                "sinus tachycardia/supraventricular tachycardia",
                "sinus bradycardia/supraventricular tachycardia",
                "sinus arrhythmia/supraventricular tachycardia",
                "sinus rhythm/supraventricular tachycardia",

                "sinus tachycardia/paroxysmal supraventricular tachycardia",
                "sinus bradycardia/paroxysmal supraventricular tachycardia",
                "sinus arrhythmia/paroxysmal supraventricular tachycardia",
                "sinus rhythm/paroxysmal supraventricular tachycardia",

                "sinus tachycardia/pac",
                "sinus bradycardia/pac",
                "sinus arrhythmia/pac",
                "sinus rhythm/pac",

                "supraventricular tachycardia/out_of_list"
            ]),
            ("V", [
                "sinus tachycardia/ventricular run",
                "sinus tachycardia/ventricular couplet",
                "sinus tachycardia/ventricular bigeminy",
                "sinus tachycardia/ventricular trigeminy",

                "sinus bradycardia/ventricular run",
                "sinus bradycardia/ventricular couplet",
                "sinus bradycardia/ventricular bigeminy",
                "sinus bradycardia/ventricular trigeminy",

                "sinus arrhythmia/ventricular run",
                "sinus arrhythmia/ventricular couplet",
                "sinus arrhythmia/ventricular bigeminy",
                "sinus arrhythmia/ventricular trigeminy",

                "sinus rhythm/ventricular run",
                "sinus rhythm/ventricular couplet",
                "sinus rhythm/ventricular bigeminy",
                "sinus rhythm/ventricular trigeminy",

                "sinus tachycardia/ventricular tachycardia",
                "sinus bradycardia/ventricular tachycardia",
                "sinus arrhythmia/ventricular tachycardia",
                "sinus rhythm/ventricular tachycardia",

                "sinus tachycardia/pvc",
                "sinus bradycardia/pvc",
                "sinus arrhythmia/pvc",
                "sinus rhythm/pvc",

                "ventricular tachycardia/out_of_list"
            ]),
        ])),

    ])

TYPES_DATA = ['N', 'S', 'V', 'R', 'BRADY', 'TACHY', 'NOISE', 'AFIB']
# TYPES_DATA = ['V']

CLASS_TYPES = ['N', 'V', 'S']

CLASSES = {
    "NOTABEAT": "Not a Beat",
    "N": "QRS",
    "S": "QRS",
    "V": "QRS",
    "ARTIFACT": "Artifact"
}

REVERSED_CLASSES = {v: k for k, v in CLASSES.items()}
NUM_CLASS = len(REVERSED_CLASSES)

CLASS_WEIGHTS = {
        0: 0.5,
        1: 2,
        2: 2,
        3: 6,
        4: 1
    }

OVERLAB_IN_FILE = 3

PATH_DATA_EC57 = '/media/xuandung-ai/Data_4T1/AI-Database/PhysionetData/'
DB_TESTING = [
        ['mitdb', 'atr', 'atr'],
        ['nstdb', 'atr', 'atr'],
        ['ahadb', 'atr', 'atr'],
        ['escdb', 'atr', 'atr'],
    ]

FILE_NAME = '*'
DEBUG = False
