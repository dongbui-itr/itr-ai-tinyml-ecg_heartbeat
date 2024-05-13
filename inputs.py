import copy
import getpass
import json
import os
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from glob import glob
from multiprocessing import Process, JoinableQueue, Lock
from os.path import basename, dirname
from random import shuffle
import csv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import wfdb as wf
from utils.reprocessing import (
    bwr,
    norm,
    afib_annotations,
    beat_annotations,
    butter_lowpass_filter,
    butter_highpass_filter,
    butter_bandpass_filter)
from wfdb.processing import resample_sig, resample_singlechan
from all_config import EXT_BEAT, EXT_BEAT_EVAL

# SYS
NUM_NORMALIZATION = 0.6
MIN_RR_INTERVAL = 0.15
HES_SAMPLING_RATE = 200
# CONFIG
NEW_MODE = 0
OLD_MODE = 1
MODE = NEW_MODE
OFFSET_FRAME_BEAT = [0, 3, 6, 9, 11]
MAX_CHANNEL = 3
ADD_ARTIFACT = True
BAND_PASS_FILTER = [1.0, 30.0]
RHYTHM_BAND_PASS_FILTER = [1.0, 30.0]
CLIP_RANGE = [-5.0, 5.0]
RHYTHM_CLIP_RANGE = [-5.0, 5.0]
MAX_NUM_IMG_SAVE = 100
MAX_LEN_PLOT = 15
EVENT_LEN_STANDARD = 100
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
           ]),
            ("A", [
           ]),
            ("V", [
           ]),
            ("R", [
           ]),
            ("Q", [
           ]),
       ])),
        ("3", OrderedDict([
            ("NOTABEAT", [
           ]),
            ("N", [
           ]),
            ("S", [
           ]),
            ("V", [
           ]),
            ("R", [
           ]),
       ])),
        ("4", OrderedDict([
            ("NOTABEAT", [
           ]),
            ("N", [
           ]),
       ])),
        ("5", OrderedDict([
            ("NOTABEAT", [
           ]),
            ("N", [
           ]),
            ("ARTIFACT", [
           ]),
       ]))
   ])
LABEL_PHY_BEAT_TYPES = OrderedDict([("NOTABEAT", []),
                                    ("N", ['N']),
                                    ("A", ['A', 'a', 'S']),
                                    ("V", ['V', 'E']),
                                   ])
INV_SYMBOL = {1: "N",
              60: "V",
              70: "A",
              90: "R",
              80: "Q"
              }
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def update(d, other):
    d.update(other)
    return d


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def np_to_tfrecords(sample_buffer, label_buffer, writer):
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))

    assert isinstance(sample_buffer, np.ndarray)
    assert len(sample_buffer.shape) == 2  # If X has a higher rank,
    # it should be rshape before fed to this function.
    assert isinstance(label_buffer, np.ndarray) or label_buffer is None

    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(sample_buffer)
    if label_buffer is not None:
        assert sample_buffer.shape[0] == label_buffer.shape[0]
        assert len(label_buffer.shape) == 2
        dtype_feature_y = _dtype_feature(label_buffer)

    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(sample_buffer.shape[0]):
        x = sample_buffer[idx]
        if label_buffer is not None:
            y = label_buffer[idx]

        d_feature = dict()
        d_feature['sample'] = dtype_feature_x(x)
        if label_buffer is not None:
            d_feature['label'] = dtype_feature_y(y)

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)


def cal_num_process_and_num_shard(files, org_num_processes, org_num_shards):
    if len(files) >= org_num_shards:
        num_processes = org_num_processes
        num_shards = org_num_shards
    else:
        for n_threads in reversed(range(org_num_processes)):
            if len(files) // n_threads >= 1:
                num_processes = n_threads
                num_shards = (len(files) // n_threads) * n_threads
                break
    return num_processes, num_shards


def initiate_process_parameters(res_db_dict, ranges):
    return_dict = dict()
    return_dict['process_res_db_dict'] = [0] * len(ranges)
    return_dict['process_lst_file_to_handle'] = [0] * len(ranges)

    for i in range(len(ranges)):
        return_dict['process_res_db_dict'][i] = copy.deepcopy(res_db_dict)
        return_dict['process_res_db_dict'][i]["eval"]["total_sample"] = 0
        return_dict['process_res_db_dict'][i]["eval"]["total_artifact_sample"] = 0
        return_dict['process_res_db_dict'][i]["eval"]["files"] = dict()
        return_dict['process_res_db_dict'][i]["train"]["total_sample"] = 0
        return_dict['process_res_db_dict'][i]["train"]["total_artifact_sample"] = 0
        return_dict['process_res_db_dict'][i]["train"]["files"] = dict()
        for key in res_db_dict["beat_class"].keys():
            return_dict['process_res_db_dict'][i]["eval"][key] = 0
            return_dict['process_res_db_dict'][i]["train"][key] = 0

        return_dict['process_lst_file_to_handle'][i] = list()

    return return_dict


def get_annotations(label_path, sig_len):
    qa_channel = label_path[:-4][-1]
    beats = []
    symbols = []
    sample_artifact = []
    try:
        ann = wf.rdann(label_path[:-4], label_path[-3:], )
        beats = np.asarray(ann.sample, dtype=int)
        symbols = np.asarray(ann.symbol)
        check = (beats >= 0) == (beats < sig_len)
        beats = beats[check]
        symbols = symbols[check]

    except:
        print(">>>>>>>>>>> ERR in label file: {} <<<<<<<<<<<<<<<<<<,".format(label_path))

    return int(qa_channel), beats, symbols, sample_artifact


def get_studyid(label_path, res=None, ext='atr'):
    studyFid = None
    eventFid = None
    hasComplexBeat = False

    if res is None:
        res = {"N": 0,
               "A": 0,
               "V": 0,
               "R": 0,
               "Q": 0,
               }

    try:
        ann = wf.rdann(label_path[:-4], ext)

        symbols = np.asarray(ann.symbol)
        # symbols[symbols=='S'] ='A'
        # symbols = np.asarray([INV_SYMBOL[s] for s in symbols])
        for key in res:
            res[key] += np.count_nonzero(symbols == key)

        studyFid = label_path.split('export_')[-1].split('/')[1]
        eventFid = label_path.split('export_')[-1].split('/')[2]
    except:
        print(">>>>>>>>>>> ERR in label file: {} <<<<<<<<<<<<<<<<<<,".format(label_path))

    return studyFid, eventFid, hasComplexBeat, res


def _process_sample(use_gpu_index,
                    lock,
                    file_path,
                    buf_em,
                    buf_ma,
                    ds_type,
                    res_db_dict,
                    writer,
                    output_directory,
                    save_image,
                    debug=False):
    file_name = file_path[:-4]
    try:
        # print(basename(file_name))
        # ext_ann = file_path[-3:]
        header = wf.rdheader(file_name)
        fs_origin = header.fs
        dir_img_deb = output_directory + 'img/'
        if not os.path.exists(dir_img_deb) and save_image:
            os.makedirs(dir_img_deb)

        num_block = res_db_dict["num_block"]
        feature_len = res_db_dict["feature_len"]
        sampling_rate = res_db_dict["sampling_rate"]
        beat_class = res_db_dict["beat_class"]
        ebwr = res_db_dict["bwr"]
        enorm = res_db_dict["norm"]
        # region PORTAL
        ss = file_name.split('/')
        main_rhythm_file = ss[-3][-1]
        sub_rhythm_file = ss[-2]
        lock.acquire()
        lock.release()

        qa_channel, beat_true, symbol_true, sample_artifact = get_annotations(file_path,
                                                                              int(EVENT_LEN_STANDARD * fs_origin))
        if 0 == len(beat_true) or len(beat_true) != len(symbol_true):
            return res_db_dict

        if qa_channel >= 0:
            event_channel = qa_channel
        else:
            event_channel = 0

        record = wf.rdsamp(file_name)[0]
        buf_record = np.nan_to_num(record)
        if fs_origin != sampling_rate:
            buf_ecg_org, _ = resample_sig(buf_record[:, event_channel],
                                          fs_origin,
                                          sampling_rate)
            beat_true = (beat_true * sampling_rate) // fs_origin
        else:
            buf_ecg_org = buf_record[:, event_channel].copy()

        len_of_standard = int(EVENT_LEN_STANDARD * sampling_rate)
        len_of_buf = len(buf_ecg_org)
        if len_of_buf < len_of_standard:
            buf_ecg_org = np.concatenate((buf_ecg_org, np.zeros(len_of_standard - len_of_buf)))

        buf_ecg_org = buf_ecg_org[:len_of_standard]

        buf_ecg = butter_bandpass_filter(buf_ecg_org, BAND_PASS_FILTER[0], BAND_PASS_FILTER[1], sampling_rate)
        buf_ecg = np.clip(buf_ecg, CLIP_RANGE[0], CLIP_RANGE[1])

        if ebwr:
            buf_ecg = bwr(buf_ecg, sampling_rate)

        if enorm:
            buf_ecg = norm(buf_ecg, int(NUM_NORMALIZATION * sampling_rate))

        ind = {k: i for i, k in enumerate(beat_class.keys())}
        ind_invert = {i: k for i, k in enumerate(beat_class.keys())}

        if len(buf_ecg) > feature_len:
            buf_ecg = buf_ecg[:feature_len]
            symbol_true = symbol_true[np.flatnonzero(beat_true < feature_len)]
            beat_true = beat_true[np.flatnonzero(beat_true < feature_len)]

        # indx_Q = np.flatnonzero(symbol_true == 'Q')
        # if len(indx_Q) > 0:
        #     symbol_true = np.delete(symbol_true, indx_Q)
        #     beat_true = np.delete(beat_true, indx_Q)

        if debug:
            plt.plot(buf_ecg)
            plt.plot(beat_true, buf_ecg[beat_true], 'r*')
            [plt.annotate(symbol_true[i], (beat_true[i], max(buf_ecg))) for i in range(len(symbol_true))]
            plt.show()
            plt.close()

        indx_N = np.flatnonzero(symbol_true != 'Q')
        if len(indx_N) > 0:
            symbol_true[indx_N] = 'N'

        indx_Q = np.flatnonzero(symbol_true == 'Q')
        if len(indx_Q) > 0:
            _symbol_true = []
            for i in range(len(symbol_true)):
                if i in indx_Q:
                    _symbol_true.append('ARTIFACT')
                else:
                    _symbol_true.append(symbol_true[i])

            symbol_true = np.asarray(_symbol_true)

        data_len = len(buf_ecg)
        symbol_true = [ind[s] for s in symbol_true]
        beat_true = np.asarray(beat_true, dtype=int)
        symbol_true = np.asarray(symbol_true, dtype=int)

        label_len = feature_len // num_block
        data_index = np.arange(feature_len)[None, :] + \
                     np.arange(0, data_len, feature_len)[:, None]
        label_index = np.arange(label_len)[None, None, :] + \
                      np.arange(0, feature_len, label_len)[None, :, None] + \
                      np.arange(0, data_len, feature_len)[:, None, None]

        lbl_samp = np.full(data_len, ind["NOTABEAT"], dtype=int)
        if len(sample_artifact) > 0 and "ARTIFACT" in beat_class.keys():
            for a in sample_artifact:
                a = (a * sampling_rate) // fs_origin
                lbl_samp[a] = ind["ARTIFACT"]

        lbl_samp[beat_true] = symbol_true
        process_data = buf_ecg[data_index]
        lbl_samp_frame = lbl_samp[label_index]

        process_label_symbol = np.asarray([np.max(lbl, axis=1) for lbl in lbl_samp_frame], dtype=int).flatten()
        res_db_dict[ds_type]["total_sample"] += len(process_data)
        for key in beat_class.keys():
            lbl_int = int(ind[key])
            lbl_pos = np.where(process_label_symbol == lbl_int)[0]
            if len(lbl_pos) > 0:
                res_db_dict[ds_type][key] += len(lbl_pos)

        # if save_image:
        #     sub_save_image = dir_img_deb + main_rhythm_file + "/" + sub_rhythm_file
        #     if not os.path.exists(sub_save_image):
        #         os.makedirs(sub_save_image)
        #         file_count = 0
        #     else:
        #         _, _, files = next(os.walk(sub_save_image))
        #         file_count = len(files)
        #
        #     if file_count < MAX_NUM_IMG_SAVE:
        #         note = ""
        #         for n in ind_invert.keys():
        #             note += "{}: {}\n".format(n, ind_invert[n])
        #
        #         buf_frame = buf_ecg.copy()
        #         buf_lbl = np.asarray([np.full(label_len, l) for l in process_label_symbol]).flatten()
        #         buf_mark = np.zeros(data_len)
        #         buf_mark[_from_event: _to_event] = max(buf_lbl)
        #         plot_len = data_len // 3
        #         fig, axx = plt.subplots(nrows=3, ncols=1, figsize=(19.20, 10.80))
        #         fig.suptitle('main: {}; sub {}; Id: {}'.format(
        #             main_rhythm_file,
        #             sub_rhythm_file,
        #             basename(file_name)), fontsize=11)
        #         for i, ax in enumerate(axx):
        #             t = np.arange(i * plot_len, (i + 1) * plot_len, 1) / sampling_rate
        #             ax.text(0, 0, technician_comment)
        #             ax.text(0, np.mean(buf_ecg[i * plot_len: (i + 1) * plot_len]), note, ha='left', rotation=0,
        #                     wrap=True)
        #             ax.plot(t, buf_frame[i * plot_len: (i + 1) * plot_len], label="buf")
        #             ax.plot(t, buf_lbl[i * plot_len: (i + 1) * plot_len], label="type")
        #             ax.plot(t, buf_mark[i * plot_len: (i + 1) * plot_len], label="mark")
        #             major_ticks = np.arange(i * plot_len, (i + 1) * plot_len, sampling_rate) / sampling_rate
        #             minor_ticks = np.arange(i * plot_len, (i + 1) * plot_len, label_len) / sampling_rate
        #             ax.set_xticks(major_ticks)
        #             ax.set_xticks(minor_ticks, minor=True)
        #             ax.set_yticks(
        #                 np.arange(round(np.min(buf_frame), 0), round(min(np.max(buf_lbl), 5), 0) + 1, 1))
        #             ax.grid(which='major', color='#CCCCCC', linestyle='--')
        #             ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        #             ax.legend()
        #
        #         DEBUG_IMG = False
        #         if not DEBUG_IMG:
        #             img_name = sub_save_image + "/" + basename(file_name) + "_" + str(event_channel)
        #             fig.savefig(img_name + ".svg", format='svg', dpi=1200)
        #             plt.close(fig)
        #         else:
        #             print(basename(file_name))
        #             plt.show()
        # print(file_name.split('export_')[-1].split('/')[0])
        # print(process_label_symbol)

        np_to_tfrecords(sample_buffer=np.reshape(process_data, (-1, feature_len)),
                        label_buffer=np.reshape(process_label_symbol, (-1, num_block)),
                        writer=writer)

        np_to_tfrecords(sample_buffer=np.reshape(process_data/3, (-1, feature_len)),
                        label_buffer=np.reshape(process_label_symbol, (-1, num_block)),
                        writer=writer)

        np_to_tfrecords(sample_buffer=np.reshape(process_data/5, (-1, feature_len)),
                        label_buffer=np.reshape(process_label_symbol, (-1, num_block)),
                        writer=writer)

        np_to_tfrecords(sample_buffer=np.reshape(process_data/9, (-1, feature_len)),
                        label_buffer=np.reshape(process_label_symbol, (-1, num_block)),
                        writer=writer)

        # endregion PORTAL

    except Exception as err:
        print("input get err: {} at {}".format(err, basename(file_name)))

    return res_db_dict


def _process_files_batch(use_gpu_index,
                         queue,
                         lock,
                         process_index,
                         ranges,
                         file_names,
                         buf_em,
                         buf_ma,
                         output_directory,
                         num_shards,
                         total_shards,
                         save_image):
    """

     :param process_index:
     :param ranges:
     :param file_names:
     :param output_train_directory:
     :param output_eval_directory:
     :param num_shards:
     :param ann_ext_all:
     :return:
     """
    global process_res_db_dict

    num_processes = len(ranges)
    assert not num_shards % num_processes
    num_shards_per_batch = int(num_shards / num_processes)
    # np.random.seed((process_index + 1) * 1000)
    shard_ranges = np.linspace(ranges[process_index][0],
                               ranges[process_index][1],
                               num_shards_per_batch + 1).astype(int)
    # num_files_in_thread = ranges[process_index][1] - ranges[process_index][0]

    # Initial parameter for each process
    res_db_dict = process_res_db_dict[process_index]

    counter = 0
    ds_type = "train" if 'train' in basename(dirname(output_directory)) else "eval"
    for s in range(num_shards_per_batch):

        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = process_index * num_shards_per_batch + s
        output_filename = '%s_%.5d-of-%.5d.tfrecord' % (ds_type,
                                                        res_db_dict['previous_shards'][ds_type] + shard + 1,
                                                        total_shards)
        output_file = os.path.join(output_directory, output_filename)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        writer = tf.io.TFRecordWriter(output_file)
        for i in files_in_shard:
            file_name = file_names[i]
            try:
                # if not 'event-mark-12-31-22-23-27-19-32-0-1' in file_name:
                #     continue

                res_db_dict = _process_sample(use_gpu_index=use_gpu_index,
                                              lock=lock,
                                              file_path=file_name,
                                              buf_em=buf_em,
                                              buf_ma=buf_ma,
                                              ds_type=ds_type,
                                              res_db_dict=res_db_dict,
                                              writer=writer,
                                              output_directory=output_directory,
                                              save_image=save_image)

                process_res_db_dict[process_index] = res_db_dict
            except Exception as e:
                lock.acquire()
                try:
                    print(e)
                    print('SKIPPED: Unexpected error while decoding %s.' % file_name)
                finally:
                    lock.release()
                continue

            shard_counter += 1
            counter += 1

            # if not counter % 1000:
            #     lock.acquire()
            #     try:
            #         print('%s [processor %d]: Processed %d of %d files in processing batch.' %
            #               (datetime.now(), process_index, counter, num_files_in_thread))
            #         sys.stdout.flush()
            #     finally:
            #         lock.release()

        writer.close()

        lock.acquire()
        # try:
        #     print('%s [processor %d]: Wrote %d files to %s' %
        #           (datetime.now(), process_index, shard_counter, output_file))
        #
        #     sys.stdout.flush()
        # finally:
        #     lock.release()

        shard_counter = 0

    queue.put({'process_res_db_dict': process_res_db_dict[process_index]})
    lock.acquire()
    # try:
    #     print('%s [processor %d]: Wrote %d files to %d shards.' %
    #           (datetime.now(), process_index, counter, num_files_in_thread))
    #     sys.stdout.flush()
    # finally:
    #     lock.release()

    queue.task_done()


def build_tfrecord(use_gpu_index,
                   db_process_info,
                   total_shards,
                   datastore_dict,
                   all_file,
                   output_directory,
                   save_image):
    """

    :param use_gpu_index:
    :param db_process_info:
    :param total_shards:
    :param datastore_dict:
    :param lst_file:
    :param output_directory:
    :param save_image:
    :return:
    """
    global process_res_db_dict

    num_processes = db_process_info['processors']
    num_shards = db_process_info['shards']
    sampling_rate = datastore_dict["sampling_rate"]
    print(output_directory)
    spacing = np.linspace(0, len(all_file), num_processes + 1).astype(np.int64)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a processor for each batch.
    # print('Launching %d processors for spacings: %s' % (num_processes, ranges))
    sys.stdout.flush()
    # Create a mechanism for monitoring when all processors are finished.
    coord = tf.train.Coordinator()

    # Initiate parameter for each process
    process_data_dict = initiate_process_parameters(datastore_dict, ranges)
    process_res_db_dict = process_data_dict['process_res_db_dict']
    num_shards = min(num_shards, len(all_file))
    processors = list()
    process_queue = [list() for _ in range(num_processes)]
    process_lock = [list() for _ in range(num_processes)]

    for process_index in range(len(ranges)):
        process_queue[process_index] = JoinableQueue()
        process_lock[process_index] = Lock()
        args = (use_gpu_index,
                process_queue[process_index],
                process_lock[process_index],
                process_index,
                ranges,
                all_file,
                None,
                None,
                output_directory,
                num_shards,
                total_shards,
                save_image)

        # _process_files_batch(use_gpu_index,
        #                      process_queue[process_index],
        #                      process_lock[process_index],
        #                      process_index,
        #                      ranges,
        #                      all_file,
        #                      None,
        #                      None,
        #                      output_directory,
        #                      num_shards,
        #                      total_shards,
        #                      save_image)

        t = Process(target=_process_files_batch, args=args)
        t.start()
        processors.append(t)

    # Get output of processes
    for process_index in range(len(ranges)):
        process_returned_data = process_queue[process_index].get()
        process_res_db_dict[process_index] = process_returned_data['process_res_db_dict']
        processors[process_index].terminate()

    # Wait for all the processors to terminate.
    coord.join(processors)

    # Concatenate processes returned output !!!
    ds_type = "train" if "train" in basename(dirname(output_directory)) else "eval"
    datastore_dict['previous_shards'][ds_type] += num_shards

    datastore_dict[ds_type]["total_sample"] += int(
        np.asarray([t[ds_type]["total_sample"] for t in process_res_db_dict]).sum())

    datastore_dict[ds_type]["total_artifact_sample"] += int(
        np.asarray([t[ds_type]["total_artifact_sample"] for t in process_res_db_dict]).sum())

    for t in process_res_db_dict:
        for tfb in t[ds_type]["files"].keys():
            datastore_dict[ds_type]["files"][tfb] = t[ds_type]["files"][tfb]

    for key in datastore_dict["beat_class"].keys():
        datastore_dict[ds_type][key] += int(np.asarray([t[ds_type][key] for t in process_res_db_dict]).sum())

    # print('%s: Finished writing all %d files in data set.' % (datetime.now(), len(all_file)))
    sys.stdout.flush()
    return datastore_dict


def create_tfrecord_from_portal_event(data_model_dir,
                                      data_dir,
                                      save_image=False,
                                      org_num_processes=os.cpu_count(),
                                      org_num_shards=os.cpu_count(),
                                      over_write=False):
    """

    """
    data_info = basename(dirname(data_model_dir))
    if not os.path.exists(data_model_dir):
        os.makedirs(data_model_dir)
    elif not over_write and os.path.exists(data_model_dir + 'datastore.txt'):
        print("{} exist!".format(data_model_dir + 'datastore.txt'))
        return
    elif over_write:
        shutil.rmtree(data_model_dir)
        os.makedirs(data_model_dir)

    sampling_rate = int(data_info.split('_')[0])
    feature_len = int(float(data_info.split('_')[1]) * sampling_rate)
    num_block = int(data_info.split('_')[2])
    block_len = int(feature_len // num_block)

    assert (feature_len % num_block) == 0, print('feature_len not mod num_block')
    tmp = feature_len
    step = 0
    while tmp % 2 == 0 and tmp > num_block:
        tmp = tmp / 2
        step += 1

    assert (int(tmp) == num_block), print('feature_len and num_block do not match')

    ebwr = bool(int(data_info.split('_')[3]))
    enorm = bool(int(data_info.split('_')[4]))
    overlap = int(data_info.split('_')[5])
    class_index = int(data_info.split('_')[6])
    add_artifact = (int(data_info.split('_')[7]) == 1)
    percent_train = float(data_info.split('_')[8])

    datastore_dict = dict()
    datastore_dict["train"] = dict()
    datastore_dict["eval"] = dict()

    datastore_dict["data_model_dir"] = data_info
    datastore_dict["data_dir"] = data_dir
    datastore_dict["sampling_rate"] = sampling_rate
    datastore_dict["num_block"] = num_block
    datastore_dict["block_len"] = block_len
    datastore_dict["compression_ratio"] = step
    datastore_dict["feature_len"] = feature_len
    datastore_dict["input_len"] = feature_len
    datastore_dict["beat_class"] = LABEL_BEAT_TYPES[str(class_index)]
    datastore_dict["val_class"] = LABEL_BEAT_TYPES[str(class_index)]
    datastore_dict["case_label_process"] = class_index
    datastore_dict["bwr"] = ebwr
    datastore_dict["norm"] = enorm
    datastore_dict["overlap"] = overlap
    datastore_dict["percent_train"] = percent_train
    datastore_dict["add_artifact"] = add_artifact
    datastore_dict["BAND_PASS_FILTER"] = BAND_PASS_FILTER
    datastore_dict["CLIP_RANGE"] = CLIP_RANGE
    datastore_dict["train"]["total_sample"] = 0
    datastore_dict["train"]["total_artifact_sample"] = 0
    datastore_dict["eval"]["total_sample"] = 0
    datastore_dict["eval"]["total_artifact_sample"] = 0
    for key in datastore_dict["beat_class"].keys():
        datastore_dict["eval"][key] = 0
        datastore_dict["train"][key] = 0

    datastore_dict['previous_shards'] = dict()

    fstatus = open(data_model_dir + '/start.txt', 'w')
    fstatus.writelines(str(datetime.now()))
    fstatus.close()

    ds_file = dict()
    ds_file["train"] = {"file": [], "studyFid": [], "eventFid": []}
    ds_file["eval"] = {"file": [], "studyFid": [], "eventFid": []}

    #region TRAIN DATA
    ds_study_info = dict()
    all_beat_type = dict()
    train_beat_type = dict()
    all_file = glob(data_dir + '/*/*/*.{}'.format(EXT_BEAT))
    all_file += glob(data_dir + '/*/*/*.{}'.format(EXT_BEAT_EVAL))
    for label in datastore_dict["beat_class"].keys():
        all_beat_type[label] = 0
        train_beat_type[label] = 0

    shuffle(all_file)
    for f in all_file:
        studyFid, eventFid, hasComplexBeat, num_beat_type = get_studyid(f)
        if studyFid not in ds_study_info.keys():
            ds_study_info[studyFid] = dict()
            ds_study_info[studyFid]["hasComplexBeat"] = []
            ds_study_info[studyFid]["eventFid"] = []
            ds_study_info[studyFid]["file"] = []
            for label in datastore_dict["beat_class"].keys():
                ds_study_info[studyFid][label] = 0

        ds_study_info[studyFid]["eventFid"].append(eventFid)
        ds_study_info[studyFid]["file"].append(f)
        ds_study_info[studyFid]["hasComplexBeat"].append(int(hasComplexBeat))
        for label in datastore_dict["beat_class"].keys():
            if label in num_beat_type:
                ds_study_info[studyFid][label] += num_beat_type[label]
                all_beat_type[label] += num_beat_type[label]
            else:
                ds_study_info[studyFid][label] = 0

    list_study_id = list(ds_study_info.keys())
    repair_data = True
    list_get = []
    while repair_data:
        train_beat_type["R"] = 0
        train_beat_type["V"] = 0
        train_beat_type["A"] = 0
        train_beat_type["N"] = 0
        list_get = []
        shuffle(list_study_id)
        for studyFid in list_study_id:
            if studyFid in list_get:
                continue

            if train_beat_type["R"] >= all_beat_type["R"] * percent_train:
                break

            lst = np.asarray([ds_study_info[studyFid]["R"],
                              ds_study_info[studyFid]["V"],
                              ds_study_info[studyFid]["A"]])

            if np.argmax(lst) == 0:
                train_beat_type["R"] += ds_study_info[studyFid]["R"]
                train_beat_type["V"] += ds_study_info[studyFid]["V"]
                train_beat_type["A"] += ds_study_info[studyFid]["A"]
                train_beat_type["N"] += ds_study_info[studyFid]["N"]

                ds_file["train"]["studyFid"].append(studyFid)
                ds_file["train"]["eventFid"] += ds_study_info[studyFid]["eventFid"]
                ds_file["train"]["file"] += ds_study_info[studyFid]["file"]
                list_get.append(studyFid)

        for studyFid in list_study_id:
            if studyFid in list_get:
                continue

            if train_beat_type["V"] >= all_beat_type["V"] * percent_train:
                break

            lst = np.asarray([ds_study_info[studyFid]["R"],
                              ds_study_info[studyFid]["V"],
                              ds_study_info[studyFid]["A"]])

            if np.argmax(lst) == 1:
                train_beat_type["R"] += ds_study_info[studyFid]["R"]
                train_beat_type["V"] += ds_study_info[studyFid]["V"]
                train_beat_type["A"] += ds_study_info[studyFid]["A"]
                train_beat_type["N"] += ds_study_info[studyFid]["N"]

                ds_file["train"]["studyFid"].append(studyFid)
                ds_file["train"]["eventFid"] += ds_study_info[studyFid]["eventFid"]
                ds_file["train"]["file"] += ds_study_info[studyFid]["file"]
                list_get.append(studyFid)

        for studyFid in list_study_id:
            if studyFid in list_get:
                continue

            if train_beat_type["A"] >= all_beat_type["A"] * percent_train:
                break

            lst = np.asarray([ds_study_info[studyFid]["R"],
                              ds_study_info[studyFid]["V"],
                              ds_study_info[studyFid]["A"]])

            if np.argmax(lst) == 2:
                train_beat_type["R"] += ds_study_info[studyFid]["R"]
                train_beat_type["V"] += ds_study_info[studyFid]["V"]
                train_beat_type["A"] += ds_study_info[studyFid]["A"]
                train_beat_type["N"] += ds_study_info[studyFid]["N"]

                ds_file["train"]["studyFid"].append(studyFid)
                ds_file["train"]["eventFid"] += ds_study_info[studyFid]["eventFid"]
                ds_file["train"]["file"] += ds_study_info[studyFid]["file"]
                list_get.append(studyFid)

        if train_beat_type["N"] >= all_beat_type["N"] * percent_train:
            break

    for studyFid in list_study_id:
        if studyFid in list_get:
            continue

        ds_file["eval"]["studyFid"].append(studyFid)
        ds_file["eval"]["eventFid"] += ds_study_info[studyFid]["eventFid"]
        ds_file["eval"]["file"] += ds_study_info[studyFid]["file"]

    for t in ["train", "eval"]:
        datastore_dict['previous_shards'][t] = 0
        out_dir = '{}{}/'.format(data_model_dir, t)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)

        db_process_info = dict()
        try:
            all_file = ds_file[t]["file"].copy()
            shuffle(all_file)
            num_processes, num_shards = cal_num_process_and_num_shard(all_file, org_num_processes, org_num_shards)
            db_process_info['files'] = all_file
            db_process_info['processors'] = num_processes
            db_process_info['shards'] = num_shards
            total_shards = np.asarray([db_process_info['shards']]).sum()

            datastore_dict = build_tfrecord(use_gpu_index=0,
                                            db_process_info=db_process_info,
                                            total_shards=total_shards,
                                            datastore_dict=datastore_dict,
                                            all_file=all_file,
                                            output_directory=out_dir,
                                            save_image=save_image)

            print('num_{}_samples = {}'.format(t, datastore_dict[t]['total_sample']))
        except Exception as e:
            print(e)

    write_log(data_model_dir,
              ds_file,
              datastore_dict,
              ds_study_info,
              all_beat_type,
              train_beat_type)

    fstatus = open(data_model_dir + '/finish.txt', 'w')
    fstatus.writelines(str(datetime.now()))
    fstatus.close()


def create_tfrecord_from_portal_event2(data_model_dir,
                                       data_dir,
                                       media_dir,
                                       save_image=False,
                                       org_num_processes=os.cpu_count(),
                                       org_num_shards=os.cpu_count(),
                                       over_write=False):
    """

    """
    data_info = basename(dirname(data_model_dir))
    if not os.path.exists(data_model_dir):
        os.makedirs(data_model_dir)
    elif not over_write and os.path.exists(data_model_dir + 'datastore.txt'):
        print("{} exist!".format(data_model_dir + 'datastore.txt'))
        return
    elif over_write:
        shutil.rmtree(data_model_dir)
        os.makedirs(data_model_dir)

    sampling_rate = int(data_info.split('_')[0])
    feature_len = int(float(data_info.split('_')[1]) * sampling_rate)
    num_block = int(data_info.split('_')[2])
    block_len = int(feature_len // num_block)

    assert (feature_len % num_block) == 0, print('feature_len not mod num_block')
    tmp = feature_len
    step = 0
    while tmp % 2 == 0 and tmp > num_block:
        tmp = tmp / 2
        step += 1

    # assert (int(tmp) == num_block), print('feature_len and num_block do not match')

    ebwr = bool(int(data_info.split('_')[3]))
    enorm = bool(int(data_info.split('_')[4]))
    overlap = int(data_info.split('_')[5])
    class_index = int(data_info.split('_')[6])
    add_artifact = (int(data_info.split('_')[7]) == 1)
    percent_train = float(data_info.split('_')[8])

    datastore_dict = dict()
    datastore_dict["train"] = dict()
    datastore_dict["eval"] = dict()

    datastore_dict["data_model_dir"] = data_info
    datastore_dict["data_dir"] = data_dir
    datastore_dict["eval_dir"] = data_dir
    datastore_dict["sampling_rate"] = sampling_rate
    datastore_dict["num_block"] = num_block
    datastore_dict["block_len"] = block_len
    datastore_dict["compression_ratio"] = step
    datastore_dict["feature_len"] = feature_len
    datastore_dict["input_len"] = feature_len
    datastore_dict["beat_class"] = LABEL_BEAT_TYPES[str(class_index)]
    datastore_dict["val_class"] = LABEL_BEAT_TYPES[str(class_index)]
    datastore_dict["case_label_process"] = class_index
    datastore_dict["bwr"] = ebwr
    datastore_dict["norm"] = enorm
    datastore_dict["overlap"] = overlap
    datastore_dict["percent_train"] = percent_train
    datastore_dict["add_artifact"] = add_artifact
    datastore_dict["BAND_PASS_FILTER"] = BAND_PASS_FILTER
    datastore_dict["CLIP_RANGE"] = CLIP_RANGE
    datastore_dict["train"]["total_sample"] = 0
    datastore_dict["train"]["total_artifact_sample"] = 0
    datastore_dict["eval"]["total_sample"] = 0
    datastore_dict["eval"]["total_artifact_sample"] = 0
    for key in datastore_dict["beat_class"].keys():
        datastore_dict["eval"][key] = 0
        datastore_dict["train"][key] = 0

    datastore_dict['previous_shards'] = dict()

    fstatus = open(data_model_dir + '/start.txt', 'w')
    fstatus.writelines(str(datetime.now()))
    fstatus.close()

    ds_file = dict()
    ds_file["train"] = {"file": [], "studyFid": [], "eventFid": []}
    ds_file["eval"] = {"file": [], "studyFid": [], "eventFid": []}

    # region TRAIN DATA
    ds_train_study_info = dict()
    all_train_beat_type = dict()
    train_beat_type = dict()

    #open log_data.json
    with open(media_dir + 'log_data_noise.json', 'r') as fp:
        list_data = json.load(fp)

    # all_file_train = glob(data_dir + '/*/*/*.{}'.format(EXT_BEAT))
    # all_file_train += glob(data_dir + '/*/*/*.{}'.format(EXT_BEAT_EVAL))
    all_file_train = []
    res = {}
    for label in datastore_dict["beat_class"].keys():
        if not 'NOTABEAT' in label:
            if label == 'N':
                list_events = []
                for _label in ['N', 'S', 'V', 'R', 'BRADY', 'TACHY']:
                    _list_events = np.asarray(list_data['Train'][f'{_label}_study'])
                    list_events.extend(_list_events)
                    for studyid in _list_events:
                        all_file_train.extend(glob(data_dir + '/export_{}/{}/*.{}'.format(_label, studyid, EXT_BEAT)))

            else:
                _label = 'noise'
                _list_events = np.asarray(list_data['Train'][f'{_label.upper()}_study'])
                list_events.extend(_list_events)
                for studyid in _list_events:
                    all_file_train.extend(glob(data_dir + '/export_{}/*/{}/*.{}'.format(_label, studyid, EXT_BEAT)))

            all_train_beat_type[label] = 0
            train_beat_type[label] = 0
            res[label] = 0

    shuffle(all_file_train)
    for f in all_file_train:
        studyFid, eventFid, hasComplexBeat, num_beat_type = get_studyid(f, res.copy())
        if studyFid not in ds_train_study_info.keys():
            ds_train_study_info[studyFid] = dict()
            ds_train_study_info[studyFid]["hasComplexBeat"] = []
            ds_train_study_info[studyFid]["eventFid"] = []
            ds_train_study_info[studyFid]["file"] = []
            for label in datastore_dict["beat_class"].keys():
                ds_train_study_info[studyFid][label] = 0

        ds_train_study_info[studyFid]["eventFid"].append(eventFid)
        ds_train_study_info[studyFid]["file"].append(f)
        ds_train_study_info[studyFid]["hasComplexBeat"].append(int(hasComplexBeat))
        for label in datastore_dict["beat_class"].keys():
            if label in num_beat_type:
                ds_train_study_info[studyFid][label] += num_beat_type[label]
                all_train_beat_type[label] += num_beat_type[label]
            else:
                ds_train_study_info[studyFid][label] = 0

    list_study_id = list(ds_train_study_info.keys())
    list_get = []
    shuffle(list_study_id)
    for studyFid in list_study_id:
        if studyFid in list_get:
            continue

        for label in datastore_dict["beat_class"].keys():
            if not 'NOTABEAT' in label :
                train_beat_type[label] += ds_train_study_info[studyFid][label]

        ds_file["train"]["studyFid"].append(studyFid)
        ds_file["train"]["eventFid"] += ds_train_study_info[studyFid]["eventFid"]
        ds_file["train"]["file"] += ds_train_study_info[studyFid]["file"]
        list_get.append(studyFid)

    # if train_beat_type["N"] >= all_train_beat_type["N"] * percent_train:
    #     break
    # endregion TRAIN DATA

    # region EVAL DATA
    ds_eval_study_info = dict()
    eval_beat_type = dict()
    all_eval_beat_type = dict()
    all_file_eval = []
    for label in datastore_dict["beat_class"].keys():
        if not 'NOTABEAT' in label :
            if label == 'N':
                list_events = []
                for _label in ['N', 'S', 'V', 'R', 'BRADY', 'TACHY']:
                    _list_events = np.asarray(list_data['Eval'][f'{_label}_study'])
                    list_events.extend(_list_events)
                    for studyid in _list_events:
                        all_file_eval.extend(glob(data_dir + '/export_{}/{}/*.{}'.format(label, studyid, EXT_BEAT)))
        else:
            _label = 'noise'
            _list_events = np.asarray(list_data['Eval'][f'{_label.upper()}_study'])
            list_events.extend(_list_events)
            for studyid in _list_events:
                all_file_eval.extend(glob(data_dir + '/export_{}/*/{}/*.{}'.format(_label, studyid, EXT_BEAT)))

        all_eval_beat_type[label] = 0
        eval_beat_type[label] = 0
        res[label] = 0

    shuffle(all_file_eval)
    for f in all_file_eval:
        studyFid, eventFid, hasComplexBeat, num_beat_type = get_studyid(f, res.copy())
        if studyFid not in ds_eval_study_info.keys():
            ds_eval_study_info[studyFid] = dict()
            ds_eval_study_info[studyFid]["hasComplexBeat"] = []
            ds_eval_study_info[studyFid]["eventFid"] = []
            ds_eval_study_info[studyFid]["file"] = []
            for label in datastore_dict["beat_class"].keys():
                ds_eval_study_info[studyFid][label] = 0

        ds_eval_study_info[studyFid]["eventFid"].append(eventFid)
        ds_eval_study_info[studyFid]["file"].append(f)
        ds_eval_study_info[studyFid]["hasComplexBeat"].append(int(hasComplexBeat))
        for label in datastore_dict["beat_class"].keys():
            if label in num_beat_type:
                ds_eval_study_info[studyFid][label] += num_beat_type[label]
                all_eval_beat_type[label] += num_beat_type[label]
            else:
                ds_eval_study_info[studyFid][label] = 0

    list_study_id = list(ds_eval_study_info.keys())
    list_get = []
    shuffle(list_study_id)
    for studyFid in list_study_id:
        if studyFid in list_get:
            continue

        for label in datastore_dict["beat_class"].keys():
            if not 'NOTABEAT' in label :
                eval_beat_type[label] += ds_eval_study_info[studyFid][label]

        ds_file["eval"]["studyFid"].append(studyFid)
        ds_file["eval"]["eventFid"] += ds_eval_study_info[studyFid]["eventFid"]
        ds_file["eval"]["file"] += ds_eval_study_info[studyFid]["file"]
        list_get.append(studyFid)

    # endregion EVAL DATA

    for t in ["train", "eval"]:
        datastore_dict['previous_shards'][t] = 0
        out_dir = '{}{}/'.format(data_model_dir, t)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)

        db_process_info = dict()
        try:
            all_file = ds_file[t]["file"].copy()
            shuffle(all_file)
            num_processes, num_shards = cal_num_process_and_num_shard(all_file, org_num_processes, org_num_shards)
            db_process_info['files'] = all_file
            db_process_info['processors'] = num_processes
            db_process_info['shards'] = num_shards
            total_shards = np.asarray([db_process_info['shards']]).sum()

            datastore_dict = build_tfrecord(use_gpu_index=0,
                                            db_process_info=db_process_info,
                                            total_shards=total_shards,
                                            datastore_dict=datastore_dict,
                                            all_file=all_file,
                                            output_directory=out_dir,
                                            save_image=save_image)

            print('num_{}_samples = {}'.format(t, datastore_dict[t]['total_sample']))
        except Exception as e:
            print(e)

    write_log2(data_model_dir,
               ds_file,
               datastore_dict,
               ds_train_study_info,
               ds_eval_study_info,
               all_train_beat_type,
               all_eval_beat_type)

    fstatus = open(data_model_dir + '/finish.txt', 'w')
    fstatus.writelines(str(datetime.now()))
    fstatus.close()


def write_log(log_path, ds_file, datastore_dict, ds_study_info, all_beat_type, train_beat_type):
    log_datastore = open(log_path + 'datastore.txt', 'w')
    json.dump(datastore_dict, log_datastore)
    log_datastore.close()

    log_datastore = open(log_path + 'ds_study_info.txt', 'w')
    json.dump(ds_study_info, log_datastore)
    log_datastore.close()

    log_datastore = open(log_path + 'all_beat_type.txt', 'w')
    json.dump(all_beat_type, log_datastore)
    log_datastore.close()

    log_datastore = open(log_path + 'train_beat_type.txt', 'w')
    json.dump(train_beat_type, log_datastore)
    log_datastore.close()

    write_csv = open(log_path + '/ds_eval.txt', 'w')
    fields = ["id", "path", "check"]
    write = csv.writer(write_csv)
    write.writerow(fields)
    for pathSave in ds_file["eval"]["file"]:
        write.writerow([basename(pathSave)[:-4], pathSave, False])

    write_csv.close()

    write_csv = open(log_path + '/ds_train.txt', 'w')
    fields = ["id", "path", "check"]
    write = csv.writer(write_csv)
    write.writerow(fields)
    for pathSave in ds_file["train"]["file"]:
        write.writerow([basename(pathSave)[:-4], pathSave, False])

    write_csv.close()

    fsds_file = open(log_path + '/ds_study_train.txt', 'w')
    for fids in ds_file["train"]["studyFid"]:
        fids = np.asarray(fids)
        fid = np.unique(fids)
        fsds_file.write(str(fid[0]) + "\n")

    fsds_file.close()

    fsds_file = open(log_path + '/ds_study_eval.txt', 'w')
    for fids in ds_file["eval"]["studyFid"]:
        fids = np.asarray(fids)
        fid = np.unique(fids)
        fsds_file.write(str(fid[0]) + "\n")

    fsds_file.close()


def write_log2(log_path, ds_file, datastore_dict, ds_train_study_info, ds_eval_study_info, all_train_beat_type,
               all_eval_beat_type):
    log_datastore = open(log_path + 'datastore.txt', 'w')
    json.dump(datastore_dict, log_datastore)
    log_datastore.close()

    log_datastore = open(log_path + 'ds_train_study_info.txt', 'w')
    json.dump(ds_train_study_info, log_datastore)
    log_datastore.close()

    log_datastore = open(log_path + 'ds_eval_study_info.txt', 'w')
    json.dump(ds_eval_study_info, log_datastore)
    log_datastore.close()

    log_datastore = open(log_path + 'all_train_beat_type.txt', 'w')
    json.dump(all_train_beat_type, log_datastore)
    log_datastore.close()

    log_datastore = open(log_path + 'all_eval_beat_type.txt', 'w')
    json.dump(all_eval_beat_type, log_datastore)
    log_datastore.close()

    write_csv = open(log_path + '/ds_eval.txt', 'w')
    fields = ["id", "path", "check"]
    write = csv.writer(write_csv)
    write.writerow(fields)
    for pathSave in ds_file["eval"]["file"]:
        write.writerow([basename(pathSave)[:-4], pathSave, False])

    write_csv.close()

    write_csv = open(log_path + '/ds_train.txt', 'w')
    fields = ["id", "path", "check"]
    write = csv.writer(write_csv)
    write.writerow(fields)
    for pathSave in ds_file["train"]["file"]:
        write.writerow([basename(pathSave)[:-4], pathSave, False])

    write_csv.close()

    fsds_file = open(log_path + '/ds_study_train.txt', 'w')
    for fids in ds_file["train"]["studyFid"]:
        fids = np.asarray(fids)
        fid = np.unique(fids)
        fsds_file.write(str(fid[0]) + "\n")

    fsds_file.close()

    fsds_file = open(log_path + '/ds_study_eval.txt', 'w')
    for fids in ds_file["eval"]["studyFid"]:
        fids = np.asarray(fids)
        fid = np.unique(fids)
        fsds_file.write(str(fid[0]) + "\n")

    fsds_file.close()


def main(argv=None):
    MEDIA_PATH = '/media/{}/MegaProject/AIAnnotation/BeatNet/211014-IMG'.format(getpass.getuser())
    DATA_PATH = '/media/{}/MegaProject/PortalData/BeatNet/2021-09-30/'.format(getpass.getuser())
    # sampling_rate
    # feature_len
    # num_block
    # bwr
    # norm
    # overlap
    # class_index
    # add_artifact
    # percent_train

    datasets = [
        '256_60.0_960_1_1_0_2_0_0.7',
   ]

    for d in datasets:
        print("\n>>>>>>>>>>> {} <<<<<<<<<<<\n".format(d))
        data_model_dir = MEDIA_PATH + '/{}/'.format(d)
        create_tfrecord_from_portal_event(data_model_dir=data_model_dir,
                                          data_dir=DATA_PATH,
                                          save_image=True,
                                          org_num_processes=1,
                                          org_num_shards=1,
                                          over_write=False,
                                          )


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.app.run()
