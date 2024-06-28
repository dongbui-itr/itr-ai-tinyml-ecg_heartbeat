from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import shutil
import datetime
import sys
from multiprocessing import JoinableQueue, Lock, Process

import numpy as np

import wfdb as wf
import tensorflow as tf

from glob import glob
from random import shuffle
from functools import partial
from os.path import basename, dirname
from all_config import *

CLIP_RANGE = [-5.0, 5.0]
BAND_PASS_FILTER = [1.0, 30.0]


def _get_tfrecord_filenames(dir_path, is_training):
    if not os.path.exists(dir_path):
        raise FileNotFoundError("{}; No such file or directory.".format(dir_path))

    filenames = sorted(glob(os.path.join(dir_path, "*.tfrecord")))
    if not filenames:
        raise FileNotFoundError("No TFRecords found in {}".format(dir_path))

    if is_training:
        shuffle(filenames)

    return filenames


def _preprocess_proto(example_proto, feature_len, label_len, class_num):
    """Read sample from protocol buffer."""
    encoding_scheme = {
        'sample': tf.io.FixedLenFeature(shape=[feature_len, ], dtype=tf.float32),
        'label': tf.io.FixedLenFeature(shape=[label_len], dtype=tf.int64),
    }
    proto = tf.io.parse_single_example(example_proto, encoding_scheme)
    sample = proto["sample"]
    label = proto["label"]
    label = tf.one_hot(label, class_num)
    return sample, label
