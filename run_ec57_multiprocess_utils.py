from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import glob
import os
import time
from collections import defaultdict
import operator
from os.path import basename, dirname

import keras.models
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import inputs as data_model
import model as model
import model_old as model_old
import wfdb as wf
from utils.reprocessing import (
    bwr,
    norm,
    beat_cluster,
    beat_select,
    remove_short_event,
    get_beat2d,
    smooth,
    butter_bandpass_filter)
from wfdb.processing import resample_sig
from collections import Counter
from post_processing import post_processing_beats_and_rhythms
from shutil import copyfile
from all_config import EXT_BEAT_EVAL, MIN_RR_INTERVAL


def bxb(predict_sample, predict_symbol, ref_sample, ref_symbol, epsilon):
    # Find true sample
    matrix = ref_sample[:, None] - predict_sample
    matrix = np.abs(matrix)
    compare_matrix = np.less_equal(matrix, epsilon)
    # Find true symbols of true sample
    true_sample_index = np.nonzero(compare_matrix)
    compare_symbol = np.array(['+'] * len(ref_symbol))
    compare_symbol[true_sample_index[0]] = predict_symbol[true_sample_index[1]]

    return list(ref_symbol), list(compare_symbol)


def beat_classification_2(model,
                        file_name,
                        channel_ecg,
                        beat_datastore,
                        overlap=None,
                        img_directory=None):
    """

    """
    print(file_name)
    sampling_rate = beat_datastore["sampling_rate"]
    beat_class = beat_datastore["beat_class"]
    beat_num_block = beat_datastore["num_block"]
    beat_feature_len = beat_datastore["feature_len"]
    beat_ebwr = beat_datastore["bwr"]
    beat_enorm = beat_datastore["norm"]
    if "BAND_PASS_FILTER" in beat_datastore.keys():
        beat_filter = beat_datastore["BAND_PASS_FILTER"]
        beat_clip = beat_datastore["CLIP_RANGE"]
    else:
        beat_filter = [1.0, 30.0]
        beat_clip = None

    header = wf.rdheader(file_name)
    fs_origin = header.fs
    samp_to = 0
    samp_from = 0
    total_peak = []
    total_label = []
    event_len = (data_model.EVENT_LEN_STANDARD * fs_origin)
    beat_inv = {i: k for i, k in enumerate(beat_class.keys())}
    beat_ind = {k: i for i, k in enumerate(beat_class.keys())}
    while samp_to <= header.sig_len - 1:
        try:
            if header.sig_len - samp_from <= 0:
                break

            # region Process
            samp_len = min(event_len, (header.sig_len - samp_from))
            samp_to = min(samp_from + samp_len, header.sig_len)

            record = wf.rdsamp(file_name, sampfrom=samp_from, sampto=samp_to, channels=[channel_ecg])
            # Avoid cases where the value is NaN
            buf_record = np.nan_to_num(record[0][:, 0])
            fs_origin = record[1].get('fs')

            if fs_origin != sampling_rate:
                buf_ecg_org, _ = resample_sig(buf_record, fs_origin, sampling_rate)
            else:
                buf_ecg_org = buf_record.copy()

            len_of_standard = int(data_model.EVENT_LEN_STANDARD * sampling_rate)

            len_of_buf = len(buf_ecg_org)
            if len_of_buf < len_of_standard:
                buf_ecg_org = np.concatenate((buf_ecg_org, np.full(len_of_standard - len_of_buf, buf_ecg_org[-1])))

            buf_ecg = butter_bandpass_filter(buf_ecg_org,
                                             beat_filter[0],
                                             beat_filter[1],
                                             sampling_rate)
            if beat_clip is not None:
                buf_ecg = np.clip(buf_ecg,
                                  beat_clip[0],
                                  beat_clip[1])

            buf_bwr_ecg = bwr(buf_ecg, sampling_rate)
            if beat_ebwr:
                buf_ecg = bwr(buf_ecg, sampling_rate)

            if beat_enorm:
                buf_ecg = norm(buf_ecg, int(data_model.NUM_NORMALIZATION * sampling_rate))

            data_len = len(buf_ecg)
            beat_label_len = beat_feature_len // beat_num_block
            data_index = np.arange(beat_feature_len)[None, :] + \
                         np.arange(0, data_len - beat_feature_len // 2, beat_feature_len - beat_label_len * 2)[:, None]

            _samp_from = (samp_from * sampling_rate) // fs_origin
            _samp_to = (samp_to * sampling_rate) // fs_origin
            if data_model.MODE == 0:
                buf_frame = []
                data_index_frame = []
                for fr in data_model.OFFSET_FRAME_BEAT:
                    if len(buf_frame) == 0:
                        buf_frame = np.concatenate((buf_ecg[fr:], np.full(fr, 0)))[data_index]
                        data_index_frame = data_index
                    else:
                        buf_frame = np.concatenate(
                            (buf_frame, np.concatenate((buf_ecg[fr:], np.full(fr, 0)))[data_index]))
                        data_index_frame = np.concatenate((data_index_frame, data_index + fr))

                buf_frame = np.asarray(buf_frame)
                data_index_frame = np.asarray(data_index_frame)

                group_beat_prob = model.predict(buf_frame)
                group_beat_candidate = np.argmax(group_beat_prob, axis=-1)

                # if True:
                #     for i, i_frame in enumerate(buf_frame):
                #         indx = np.flatnonzero(group_beat_candidate[i] == 1) * 32
                #
                #         plt.plot(i_frame)
                #         plt.plot(indx, i_frame[indx], 'r*')
                #         plt.show()

                label_index = np.arange(beat_label_len)[None, :] + \
                              np.arange(0, beat_feature_len, beat_label_len)[:, None]

                group_bwr_frame = buf_bwr_ecg[data_index_frame]
                beats = []
                symbols = []
                amps = []

                for i_cnt, beat_candidate in enumerate(group_beat_candidate):
                    beat_candidate = np.asarray(beat_candidate).reshape((-1, beat_num_block))
                    _beats = []
                    group_bwr_buff = group_bwr_frame[i_cnt]
                    group_offset = data_index_frame[i_cnt]
                    group_beat = beat_candidate[0]
                    _group_offset = group_offset[label_index]
                    _index = np.flatnonzero(abs(np.diff(group_beat)) > 0) + 1
                    _group_beat = np.split(group_beat, _index)
                    _group_offset = np.split(_group_offset, _index)
                    for gbeat, goffset in zip(_group_beat, _group_offset):
                        if np.max(gbeat) > beat_ind["NOTABEAT"]:
                            goffset = np.asarray(goffset).flatten()
                            index_ext = goffset.copy()
                            if (goffset[0] - beat_label_len) >= 0:
                                index_ext = np.concatenate((np.arange((goffset[0] - beat_label_len),
                                                                      goffset[0]), index_ext))

                            if (goffset[-1] + beat_label_len) < beat_feature_len:
                                index_ext = np.concatenate((index_ext,
                                                            np.arange(goffset[-1], (goffset[-1] + beat_label_len))))

                            gbuff = np.asarray(group_bwr_buff[index_ext - group_offset[0]]).flatten()
                            flip_g = gbuff * -1.0
                            peaks_up = np.argmax(gbuff)
                            peaks_down = np.argmax(flip_g)
                            if abs(gbuff[peaks_up]) > abs(flip_g[peaks_down]):
                                ma = abs(gbuff[peaks_up])
                                peaks = peaks_up
                            else:
                                ma = abs(flip_g[peaks_down])
                                peaks = peaks_down

                            amps.append(ma)
                            beats.append(peaks + index_ext[0])
                            _beats.append(peaks + index_ext[0])
                            qr_count = Counter(gbeat)
                            qr = qr_count.most_common(1)[0][0]
                            symbols.append(beat_inv[qr])

                    # plt.plot(_beats - group_offset[0], buf_frame[i_cnt][_beats - group_offset[0]], 'r*')
                    # plt.show()
                    # a=10
                    # if len(np.flatnonzero(np.diff(np.asarray(beats) * 360 // 128) > 1000)) > 0:
                    #     a=10

                if len(beats) > 0:
                    symbols = [x for _, x in sorted(zip(beats, symbols))]
                    amps = [x for _, x in sorted(zip(beats, amps))]
                    beats = sorted(beats)

                    beats = np.asarray(beats, dtype=int)
                    symbols = np.asarray(symbols)
                    amps = np.asarray(amps)
                    index_artifact = symbols == 'ARTIFACT'
                    sample_artifact = []
                    if np.count_nonzero(index_artifact) > 0:
                        sample_artifact = np.zeros(data_len, dtype=int)
                        sample_artifact[beats[index_artifact]] = 1
                        label_index_artifact = np.arange(sampling_rate)[None, :] + \
                                               np.arange(0, data_len, sampling_rate)[:, None]
                        sample_artifact = sample_artifact[label_index_artifact]
                        sample_artifact = np.asarray([np.max(lbl) for lbl in sample_artifact], dtype=int)

                        index_del = np.where(symbols == 'ARTIFACT')[0]
                        symbols = np.delete(symbols, index_del)
                        beats = np.delete(beats, index_del)
                        amps = np.delete(amps, index_del)

                    if len(beats) > 0:
                        min_rr = data_model.MIN_RR_INTERVAL * sampling_rate
                        try:
                            group_beats, group_symbols, group_amps, group_len = beat_cluster(beats,
                                                                                             symbols,
                                                                                             amps,
                                                                                             min_rr)
                        except Exception as err:
                            print(err)

                        beats = []
                        symbols = []
                        amps = []
                        for _beat, _symbol, _amp in zip(group_beats, group_symbols, group_amps):
                            qr_count = Counter(_symbol)
                            qr = qr_count.most_common(1)[0][0]
                            symbols.append(qr)

                            p = np.argmax(_amp)
                            amps.append(max(_amp))
                            beats.append(_beat[p])
                            if len(np.flatnonzero(np.diff(beats) > 1000)) > 0:
                                a = 10

                        beats = np.asarray(beats)
                        symbols = np.asarray(symbols)
                        amps = np.asarray(amps)
                        # try:
                        #     beats, symbols, amps = beat_select(beats, symbols, amps, buf_bwr_ecg, sampling_rate)
                        # except Exception as err:
                        #     print(err)

                        # # region debug
                        # t = np.arange(_samp_from, _samp_from + len(buf_ecg), 1) / sampling_rate
                        # plt.plot(t, buf_ecg)
                        # # plt.plot(t, lbl_draw)
                        # for b, s in zip(beats, symbols):
                        #     plt.annotate(s, xy=(t[b], buf_ecg[b]))
                        #
                        # plt.show()
                        # # endregion debug
            else:
                frame = buf_ecg[data_index]
                buf_ecg1 = np.concatenate((buf_ecg[3:], np.zeros(3)))
                frame2 = buf_ecg1[data_index]
                buf_ecg3 = np.concatenate((buf_ecg[5:], np.zeros(5)))
                frame3 = buf_ecg3[data_index]
                lenOfframe = len(frame)
                frame = np.concatenate((frame, frame2, frame3))
                group_beat_prob = model.predict(frame)
                group_beat_candidate = np.argmax(group_beat_prob, axis=-1)
                beat_candidate = group_beat_candidate[:lenOfframe, :]
                beat_candidate2 = group_beat_candidate[lenOfframe: lenOfframe * 2, :]
                beat_candidate3 = group_beat_candidate[lenOfframe * 2:, :]

                label_index = np.arange(beat_label_len)[None, :] + \
                              np.arange(0, beat_feature_len, beat_label_len)[:, None]
                beats = []
                symbols = []
                amps = []
                for beats_group, beats_group2, beats_group3, buf_group, index_group in zip(beat_candidate,
                                                                                           beat_candidate2,
                                                                                           beat_candidate3, frame,
                                                                                           data_index):
                    buf = buf_group[label_index]
                    idx = index_group[label_index]
                    for qr0, qr2, qr3, gr, id in zip(beats_group, beats_group2, beats_group3, buf, idx):
                        # qr_count = Counter([qr0, qr2, qr3])
                        # qr = qr_count.most_common(1)[0][0]
                        qr = max([qr0, qr2, qr3])
                        if beat_ind["NOTABEAT"] < qr:
                            flip_g = gr * -1.0
                            peaks = np.argmax(gr)
                            flip_peaks = np.argmax(flip_g)
                            peaks = np.concatenate(([peaks], [flip_peaks]))
                            ma = np.abs(gr[peaks] - np.mean(gr))
                            mb = np.argmax(ma)
                            amps.append(max(ma))
                            beats.append(peaks[mb] + id[0])
                            symbols.append(beat_inv[qr])

                beats = np.asarray(beats)
                symbols = np.asarray(symbols)
                index_artifact = symbols == 'ARTIFACT'
                sample_artifact = []
                if np.count_nonzero(index_artifact) > 0:
                    sample_artifact = np.zeros(data_len, dtype=int)
                    sample_artifact[beats[index_artifact]] = 1
                    label_index_artifact = np.arange(sampling_rate)[None, :] + \
                                           np.arange(0, data_len, sampling_rate)[:, None]
                    sample_artifact = sample_artifact[label_index_artifact]
                    sample_artifact = np.asarray([np.max(lbl) for lbl in sample_artifact], dtype=int)

                    index_del = np.where(symbols == 'ARTIFACT')[0]
                    symbols = np.delete(symbols, index_del)
                    beats = np.delete(beats, index_del)

                if len(beats) > 0:
                    symbols = np.asarray([x for _, x in sorted(zip(beats, symbols))])
                    amps = np.asarray([x for _, x in sorted(zip(beats, amps))])
                    beats = np.sort(beats)

                    group_index = np.where(abs(np.diff(beats)) > (data_model.MIN_RR_INTERVAL * sampling_rate))[0] + 1
                    group_beats = np.split(beats, group_index)
                    group_symbols = np.split(symbols, group_index)
                    group_amps = np.split(amps, group_index)
                    beats = []
                    symbols = []
                    for _beat, _symbol, _amp in zip(group_beats, group_symbols, group_amps):
                        # occurence_count = Counter(_symbol)
                        # sym = occurence_count.most_common(1)[0][0]
                        iamp = np.argmax(_amp)
                        symbols.append(_symbol[iamp])
                        beats.append(_beat[iamp])

                    beats = np.asarray(beats)
                    symbols = np.asarray(symbols)

                    # # region debug
                    # t = np.arange(_samp_from, _samp_from + len(buf_ecg), 1) / sampling_rate
                    # plt.plot(t, buf_ecg)
                    # # plt.plot(t, lbl_draw)
                    # for b, s in zip(beats, symbols):
                    #     plt.annotate(s, xy=(t[b], buf_ecg[b]))
                    #
                    # plt.show()
                    # # endregion debug

            if img_directory is not None:
                # if True:
                #     img_directory = os.path.dirname(beat_datastore)
                fig, axx = plt.subplots(nrows=data_model.EVENT_LEN_STANDARD // data_model.MAX_LEN_PLOT, ncols=1,
                                        figsize=(19.20, 10.80))
                plt.subplots_adjust(
                    hspace=0,
                    wspace=0.04,
                    left=0.04,  # the left side of the subplots of the figure
                    right=0.98,  # the right side of the subplots of the figure
                    bottom=0.2,  # the bottom of the subplots of the figure
                    top=0.88
                )
                rhythm_main = basename(dirname(file_name))
                sub_save_image = img_directory + "/" + rhythm_main

                file_count = 0
                if not os.path.exists(sub_save_image):
                    os.makedirs(sub_save_image)
                else:
                    for root, dirs, files in os.walk(sub_save_image):
                        file_count += len(files)

                sub_save_image = img_directory + "/" + rhythm_main + "/{}".format(file_count // 500)
                if not os.path.exists(sub_save_image):
                    os.makedirs(sub_save_image)

                fig.suptitle('Channel: {}; Sub Rhythm: {}; Id: {}'.format(
                    channel_ecg,
                    rhythm_main,
                    basename(file_name)), fontsize=11)

                if len(beats) > 0:
                    draw_index = np.arange(beat_label_len)[None, None, :] + \
                                 np.arange(0, beat_feature_len, beat_label_len)[None, :, None] + \
                                 np.arange(0, data_len, beat_feature_len)[:, None, None]
                    draw_beat_label = []
                    for beat_candidate in group_beat_candidate:
                        beat_candidate = np.asarray(beat_candidate).reshape((-1, beat_num_block))
                        _draw_beat_label = np.zeros(data_len)[draw_index]
                        for s in range(len(beat_candidate)):
                            for l in range(len(beat_candidate[s])):
                                _draw_beat_label[s][l] = np.full(len(_draw_beat_label[s][l]), beat_candidate[s][l])

                        draw_beat_label.append(_draw_beat_label.flatten())

                for i, ax in enumerate(axx):
                    start_buf = i * data_model.MAX_LEN_PLOT * sampling_rate
                    stop_buf = min((i + 1) * data_model.MAX_LEN_PLOT * sampling_rate, data_len)

                    plot_len = (data_model.MAX_LEN_PLOT * sampling_rate)
                    t = np.arange(start_buf, start_buf + plot_len, 1) / sampling_rate
                    buf_ecg_draw = buf_ecg_org.copy()
                    buf_ecg_draw = butter_bandpass_filter(buf_ecg_draw, 0.5, 40.0, sampling_rate)
                    buf_ecg_draw = np.clip(buf_ecg_draw, -5.0, 5.0)
                    # buf_ecg_draw = bwr(buf_ecg_draw, sampling_rate)
                    buf_frame = buf_ecg_draw[start_buf: stop_buf]
                    plot_len = len(buf_frame)

                    ax.plot(t, buf_frame, linestyle='-', linewidth=2.0)
                    if len(beats) > 0:
                        for d, _draw_beat_label in enumerate(draw_beat_label):
                            ax.plot(t, (_draw_beat_label[start_buf: stop_buf]) + 0.1 * d, label="BEAT-{}".format(d))

                        total_beat_draw = beats.copy()
                        index_draw = (total_beat_draw >= start_buf) == (total_beat_draw < stop_buf)
                        beats_draw = total_beat_draw[index_draw] - start_buf
                        symbols_draw = symbols[index_draw]
                        amps_draw = amps[index_draw]
                        for b, s, a in zip(beats_draw, symbols_draw, amps_draw):
                            if s == "V":
                                ax.annotate('{}\n{:.3f}'.format(s, a), xy=(t[b], buf_frame[b]), color='red',
                                            fontsize=8, fontweight='bold')
                            elif s == "A":
                                ax.annotate('{}\n{:.3f}'.format(s, a), xy=(t[b], buf_frame[b]), color='blue',
                                            fontsize=8, fontweight='bold')
                            elif s == "R":
                                ax.annotate('{}\n{:.3f}'.format(s, a), xy=(t[b], buf_frame[b]), color='green',
                                            fontsize=8, fontweight='bold')
                            else:
                                ax.annotate('{}\n{:.3f}'.format(s, a), xy=(t[b], buf_frame[b]), color='black',
                                            fontsize=8, fontweight='bold')

                    major_ticks = np.arange(start_buf, start_buf + plot_len,
                                            sampling_rate) / sampling_rate
                    minor_ticks = np.arange(start_buf, start_buf + plot_len,
                                            sampling_rate // 4) / sampling_rate
                    ax.set_xticks(major_ticks)
                    ax.set_xticks(minor_ticks, minor=True)
                    ax.grid(which='major', color='#CCCCCC', linestyle='--')
                    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
                    ax.legend(loc="lower right", prop={'size': 6})

                    if stop_buf == data_len:
                        break

                DEBUG_IMG = True
                if not DEBUG_IMG:
                    img_name = sub_save_image + "/{}".format(basename(file_name))

                    fig.savefig(img_name + ".svg", format='svg', dpi=1200)
                    plt.close(fig)
                else:
                    plt.show()

            if len(beats) > 0:
                beats = (beats * fs_origin) // sampling_rate
                beats += samp_from

                if len(total_label) == 0:
                    total_label = symbols
                    total_peak = beats
                else:
                    try:
                        indx = np.flatnonzero((beats[0] - total_peak) < MIN_RR_INTERVAL * fs_origin)[0]
                        total_label = np.concatenate((total_label[:indx], symbols), axis=0)
                        total_peak = np.concatenate((total_peak[:indx], beats), axis=0)
                    except:
                        total_label = np.concatenate((total_label, symbols), axis=0)
                        total_peak = np.concatenate((total_peak, beats), axis=0)

            if overlap is None:
                samp_from = samp_to
            else:
                samp_from = samp_to - overlap * fs_origin

        except Exception as e:
            print("process_sample {}: {}".format(file_name, e))
            pass

    if len(total_peak) == 0:
        return np.asarray([0], dtype=int), np.asarray([1]), fs_origin

    return np.asarray(total_peak, dtype=int), np.asarray(total_label), fs_origin


def beat_rhythm_classification(beat_model,
                               beat_datastore,
                               rhythm_model,
                               rhythm_datastore,
                               file_name,
                               channel_ecg=0,
                               img_directory=None):
    """

    """
    header = wf.rdheader(file_name)
    fs_origin = header.fs
    samp_to = 0
    samp_from = 0
    total_beat = []
    total_symbol = []

    total_sample = []
    total_rhythm = []
    event_len = (data_model.EVENT_LEN_STANDARD * fs_origin)

    beat_class = beat_datastore["beat_class"]
    beat_num_block = beat_datastore["num_block"]
    beat_feature_len = beat_datastore["feature_len"]
    beat_ebwr = beat_datastore["bwr"]
    beat_enorm = beat_datastore["norm"]
    if "BAND_PASS_FILTER" in beat_datastore.keys():
        beat_filter = beat_datastore["BAND_PASS_FILTER"]
        beat_clip = beat_datastore["CLIP_RANGE"]
    else:
        beat_filter = [1.0, 30.0]
        beat_clip = None

    rhythm_class = rhythm_datastore["rhythm_class"]
    rhythm_block_len = rhythm_datastore["block_len"]
    rhythm_feature_len = rhythm_datastore["feature_len"]
    sampling_rate = rhythm_datastore["sampling_rate"]
    rhythm_ebwr = rhythm_datastore["bwr"]
    rhythm_enorm = rhythm_datastore["norm"]
    if "BAND_PASS_FILTER" in rhythm_datastore.keys():
        rhythm_filter = rhythm_datastore["BAND_PASS_FILTER"]
        rhythm_clip = rhythm_datastore["CLIP_RANGE"]
    else:
        rhythm_filter = [1.0, 30.0]
        rhythm_clip = None

    beat_inv = {i: k for i, k in enumerate(beat_class.keys())}
    beat_ind = {k: i for i, k in enumerate(beat_class.keys())}
    rhythm_inv = {i: k for i, k in enumerate(rhythm_class.keys())}
    rhythm_ind = {k: i for i, k in enumerate(rhythm_class.keys())}

    while samp_to <= header.sig_len:
        try:
            if header.sig_len - samp_from <= 0:
                break

            # region Process
            samp_len = min(event_len, (header.sig_len - samp_from))
            samp_to = samp_from + samp_len
            record = wf.rdsamp(file_name, sampfrom=samp_from, sampto=samp_to, channels=[channel_ecg])
            # Avoid cases where the value is NaN
            buf_record = np.nan_to_num(record[0][:, 0])
            fs_origin = record[1].get('fs')

            if fs_origin != sampling_rate:
                buf_ecg_org, _ = resample_sig(buf_record, fs_origin, sampling_rate)
            else:
                buf_ecg_org = buf_record.copy()

            len_of_standard = int(data_model.EVENT_LEN_STANDARD * sampling_rate)

            len_of_buf = len(buf_ecg_org)
            if len_of_buf < len_of_standard:
                buf_ecg_org = np.concatenate((buf_ecg_org, np.full(len_of_standard - len_of_buf, buf_ecg_org[-1])))

            # endregion

            # region BEAT
            buf_ecg = butter_bandpass_filter(buf_ecg_org,
                                             beat_filter[0],
                                             beat_filter[1],
                                             sampling_rate)
            if beat_clip is not None:
                buf_ecg = np.clip(buf_ecg,
                                  beat_clip[0],
                                  beat_clip[1])

            buf_bwr_ecg = bwr(buf_ecg_org, sampling_rate)
            if beat_ebwr:
                buf_ecg = bwr(buf_ecg_org, sampling_rate)

            if beat_enorm:
                buf_ecg = norm(buf_ecg, int(data_model.NUM_NORMALIZATION * sampling_rate))

            data_len = len(buf_ecg)
            beat_label_len = beat_feature_len // beat_num_block
            data_index = np.arange(beat_feature_len)[None, :] + \
                         np.arange(0, data_len, beat_feature_len)[:, None]

            _samp_from = (samp_from * sampling_rate) // fs_origin
            _samp_to = (samp_to * sampling_rate) // fs_origin

            if data_model.MODE == 0:  # New mode
                buf_frame = []
                for fr in data_model.OFFSET_FRAME_BEAT:
                    if len(buf_frame) == 0:
                        buf_frame = np.concatenate((buf_ecg[fr:], np.full(fr, 0)))[data_index]
                    else:
                        buf_frame = np.concatenate(
                            (buf_frame, np.concatenate((buf_ecg[fr:], np.full(fr, 0)))[data_index]))

                buf_frame = np.asarray(buf_frame)
                group_beat_prob = beat_model.predict(buf_frame)
                group_beat_candidate = np.argmax(group_beat_prob, axis=-1)
                group_beat_candidate = group_beat_candidate.reshape((-1, len(data_index), beat_num_block))
                label_index = np.arange(beat_label_len)[None, :] + \
                              np.arange(0, beat_feature_len, beat_label_len)[:, None]

                group_bwr_frame = buf_bwr_ecg[data_index]
                beats = []
                symbols = []
                amps = []

                for beat_candidate in group_beat_candidate:
                    # beat_candidate = np.asarray(beat_candidate).reshape((-1, beat_num_block))
                    for group_beat, group_bwr_buff, group_offset in zip(beat_candidate,
                                                                        group_bwr_frame,
                                                                        data_index):
                        _group_offset = group_offset[label_index]
                        _index = np.where(abs(np.diff(group_beat)) > 0)[0] + 1
                        _group_beat = np.split(group_beat, _index)
                        _group_offset = np.split(_group_offset, _index)
                        for gbeat, goffset in zip(_group_beat, _group_offset):
                            if np.max(gbeat) > beat_ind["NOTABEAT"]:
                                goffset = np.asarray(goffset).flatten()
                                index_ext = goffset.copy()
                                if (goffset[0] - beat_label_len) >= 0:
                                    index_ext = np.concatenate((np.arange((goffset[0] - beat_label_len),
                                                                          goffset[0]), index_ext))

                                if (goffset[-1] + beat_label_len) < beat_feature_len:
                                    index_ext = np.concatenate((index_ext,
                                                                np.arange(goffset[-1], (goffset[-1] + beat_label_len))))

                                gbuff = np.asarray(group_bwr_buff[index_ext]).flatten()
                                flip_g = gbuff * -1.0
                                peaks_up = np.argmax(gbuff)
                                peaks_down = np.argmax(flip_g)
                                if abs(gbuff[peaks_up]) > abs(flip_g[peaks_down]):
                                    ma = abs(gbuff[peaks_up])
                                    peaks = peaks_up
                                else:
                                    ma = abs(flip_g[peaks_down])
                                    peaks = peaks_down

                                amps.append(ma)
                                beats.append(peaks + index_ext[0])
                                qr_count = Counter(gbeat)
                                qr = qr_count.most_common(1)[0][0]
                                symbols.append(beat_inv[qr])

                if len(beats) > 0:
                    symbols = [x for _, x in sorted(zip(beats, symbols))]
                    amps = [x for _, x in sorted(zip(beats, amps))]
                    beats = sorted(beats)

                    beats = np.asarray(beats, dtype=int)
                    symbols = np.asarray(symbols)
                    amps = np.asarray(amps)
                    index_artifact = symbols == 'ARTIFACT'
                    sample_artifact = []
                    if np.count_nonzero(index_artifact) > 0:
                        sample_artifact = np.zeros(data_len, dtype=int)
                        sample_artifact[beats[index_artifact]] = 1
                        label_index_artifact = np.arange(sampling_rate)[None, :] + \
                                               np.arange(0, data_len, sampling_rate)[:, None]
                        sample_artifact = sample_artifact[label_index_artifact]
                        sample_artifact = np.asarray([np.max(lbl) for lbl in sample_artifact], dtype=int)

                        index_del = np.where(symbols == 'ARTIFACT')[0]
                        symbols = np.delete(symbols, index_del)
                        beats = np.delete(beats, index_del)
                        amps = np.delete(amps, index_del)

                    if len(beats) > 0:
                        min_rr = data_model.MIN_RR_INTERVAL * sampling_rate
                        group_beats, group_symbols, group_amps, group_len = beat_cluster(beats,
                                                                                         symbols,
                                                                                         amps,
                                                                                         min_rr)
                        beats = []
                        symbols = []
                        amps = []
                        for _beat, _symbol, _amp in zip(group_beats, group_symbols, group_amps):
                            qr_count = Counter(_symbol)
                            qr = qr_count.most_common(1)[0][0]
                            symbols.append(qr)

                            p = np.argmax(_amp)
                            amps.append(max(_amp))
                            beats.append(_beat[p])

                        beats = np.asarray(beats)
                        symbols = np.asarray(symbols)
                        amps = np.asarray(amps)

                        beats, symbols, amps = beat_select(beats, symbols, amps, buf_bwr_ecg, sampling_rate)
            else:  # Old mode
                frame = buf_ecg[data_index]
                buf_ecg1 = np.concatenate((buf_ecg[3:], np.zeros(3)))
                frame2 = buf_ecg1[data_index]
                buf_ecg3 = np.concatenate((buf_ecg[5:], np.zeros(5)))
                frame3 = buf_ecg3[data_index]
                lenOfframe = len(frame)
                frame = np.concatenate((frame, frame2, frame3))
                group_beat_prob = beat_model.predict(frame)
                group_beat_candidate = np.argmax(group_beat_prob, axis=-1)
                beat_candidate = group_beat_candidate[:lenOfframe, :]
                beat_candidate2 = group_beat_candidate[lenOfframe: lenOfframe * 2, :]
                beat_candidate3 = group_beat_candidate[lenOfframe * 2:, :]

                label_index = np.arange(beat_label_len)[None, :] + \
                              np.arange(0, beat_feature_len, beat_label_len)[:, None]
                beats = []
                symbols = []
                amps = []
                for beats_group, beats_group2, beats_group3, buf_group, index_group in zip(beat_candidate, beat_candidate2,
                                                                                           beat_candidate3, frame,
                                                                                           data_index):
                    buf = buf_group[label_index]
                    idx = index_group[label_index]
                    for qr0, qr2, qr3, gr, id in zip(beats_group, beats_group2, beats_group3, buf, idx):
                        # qr_count = Counter([qr0, qr2, qr3])
                        # qr = qr_count.most_common(1)[0][0]
                        qr = max([qr0, qr2, qr3])
                        if beat_ind["NOTABEAT"] < qr:
                            flip_g = gr * -1.0
                            peaks = np.argmax(gr)
                            flip_peaks = np.argmax(flip_g)
                            peaks = np.concatenate(([peaks], [flip_peaks]))
                            ma = np.abs(gr[peaks] - np.mean(gr))
                            mb = np.argmax(ma)
                            amps.append(max(ma))
                            beats.append(peaks[mb] + id[0])
                            symbols.append(beat_inv[qr])

                beats = np.asarray(beats)
                symbols = np.asarray(symbols)
                index_artifact = symbols == 'ARTIFACT'
                sample_artifact = []
                if np.count_nonzero(index_artifact) > 0:
                    sample_artifact = np.zeros(data_len, dtype=int)
                    sample_artifact[beats[index_artifact]] = 1
                    label_index_artifact = np.arange(sampling_rate)[None, :] + \
                                           np.arange(0, data_len, sampling_rate)[:, None]
                    sample_artifact = sample_artifact[label_index_artifact]
                    sample_artifact = np.asarray([np.max(lbl) for lbl in sample_artifact], dtype=int)

                    index_del = np.where(symbols == 'ARTIFACT')[0]
                    symbols = np.delete(symbols, index_del)
                    beats = np.delete(beats, index_del)

                if len(beats) > 0:
                    symbols = np.asarray([x for _, x in sorted(zip(beats, symbols))])
                    amps = np.asarray([x for _, x in sorted(zip(beats, amps))])
                    beats = np.sort(beats)

                    group_index = np.where(abs(np.diff(beats)) > (data_model.MIN_RR_INTERVAL * sampling_rate))[0] + 1
                    group_beats = np.split(beats, group_index)
                    group_symbols = np.split(symbols, group_index)
                    group_amps = np.split(amps, group_index)
                    beats = []
                    symbols = []
                    amps = []
                    for _beat, _symbol, _amp in zip(group_beats, group_symbols, group_amps):
                        # occurence_count = Counter(_symbol)
                        # sym = occurence_count.most_common(1)[0][0]
                        iamp = np.argmax(_amp)
                        symbols.append(_symbol[iamp])
                        beats.append(_beat[iamp])
                        amps.append(np.max(_amp))

                    beats = np.asarray(beats)
                    symbols = np.asarray(symbols)
                    amps = np.asarray(amps)

            # endregion BEAT

            # region RHYTHM

            buf_ecg = butter_bandpass_filter(buf_ecg_org,
                                             rhythm_filter[0],
                                             rhythm_filter[1],
                                             sampling_rate)
            if rhythm_clip is not None:
                buf_ecg = np.clip(buf_ecg,
                                  rhythm_clip[0],
                                  rhythm_clip[1])

            if rhythm_ebwr:
                buf_ecg = bwr(buf_ecg, sampling_rate)

            if rhythm_enorm:
                buf_ecg = norm(buf_ecg, int(data_model.NUM_NORMALIZATION * sampling_rate))

            data_index = np.arange(rhythm_feature_len)[None, :] + \
                         np.arange(0, data_len, rhythm_feature_len)[:, None]
            frame = buf_ecg[data_index]
            rhythm_prob = rhythm_model.predict(frame)
            rhythm_candidate = np.argmax(rhythm_prob, axis=-1)
            rhythm_candidate = rhythm_candidate.flatten()
            if len(sample_artifact) > 0:
                rhythm_candidate[sample_artifact == 1] = rhythm_ind["OTHER"]

            # group_index = np.flatnonzero(np.abs(np.diff(rhythm_candidate)) != 0) + 1
            # group_split = np.split(rhythm_candidate, group_index)
            # valid_length = 9
            # most_rhythm = 1
            # if len(group_split) >= 2:
            #     for i in range(len(group_split)):
            #         if len(group_split[i]) > valid_length:
            #             most_rhythm = group_split[i][0]
            #             break
            #     for i in range(len(group_split)):
            #         if len(group_split[i]) <= valid_length:
            #             group_split[i][:] = most_rhythm
            #         else:
            #             most_rhythm = group_split[i][0]
            #         rhythm_candidate = np.concatenate(group_split)

            sample = np.arange(0, data_len, rhythm_block_len)
            # endregion RHYTHM

            post_processing_result = post_processing_beats_and_rhythms(samples=beats,
                                                                       symbols=symbols,
                                                                       rhythms=rhythm_candidate)
            beats = post_processing_result['samples']
            symbols = post_processing_result['symbols']
            rhythm_candidate = post_processing_result['rhythms']
            sample = (sample * fs_origin) // sampling_rate
            sample += samp_from

            rhythm_candidate_pred_draw = np.zeros(data_len)
            rhythm_candidate_pred_draw = rhythm_candidate_pred_draw[label_index]
            for i, p in enumerate(rhythm_candidate):
                rhythm_candidate_pred_draw[i] = p

            rhythm_candidate_pred_draw = rhythm_candidate_pred_draw.flatten()

            if img_directory is not None:
                fig, axx = plt.subplots(nrows=data_model.EVENT_LEN_STANDARD // data_model.MAX_LEN_PLOT, ncols=1,
                                        figsize=(19.20, 10.80))
                plt.subplots_adjust(
                    hspace=0,
                    wspace=0.04,
                    left=0.04,  # the left side of the subplots of the figure
                    right=0.98,  # the right side of the subplots of the figure
                    bottom=0.2,  # the bottom of the subplots of the figure
                    top=0.88
                )
                rhythm_main = post_processing_result["main_rhythm"]
                rhythm_sub = post_processing_result["sub_rhythm"]

                if len(rhythm_sub) == 0:
                    rhythm_sub = "out_of_list"

                sub_save_image = img_directory + "/" + rhythm_main + "/" + rhythm_sub

                file_count = 0
                if not os.path.exists(sub_save_image):
                    os.makedirs(sub_save_image)
                else:
                    for root, dirs, files in os.walk(sub_save_image):
                        file_count += len(files)

                sub_save_image = img_directory + "/" + rhythm_main + "/" + rhythm_sub + "/{}".format(file_count // 500)
                if not os.path.exists(sub_save_image):
                    os.makedirs(sub_save_image)

                fig.suptitle('Channel: {}; Main Rhythm: {}; Sub Rhythm: {}; Id: {}'.format(
                    channel_ecg,
                    rhythm_main,
                    rhythm_sub,
                    basename(file_name)), fontsize=11)

                note = ""
                for n in rhythm_inv.keys():
                    note += "{}: {}\n".format(n, rhythm_inv[n])

                if len(total_beat) > 0:
                    draw_index = np.arange(beat_label_len)[None, None, :] + \
                                 np.arange(0, beat_feature_len, beat_label_len)[None, :, None] + \
                                 np.arange(0, data_len, beat_feature_len)[:, None, None]
                    draw_beat_label = []
                    for beat_candidate in group_beat_candidate:
                        beat_candidate = np.asarray(beat_candidate).reshape((-1, beat_num_block))
                        _draw_beat_label = np.zeros(data_len)[draw_index]
                        for s in range(len(beat_candidate)):
                            for l in range(len(beat_candidate[s])):
                                _draw_beat_label[s][l] = np.full(len(_draw_beat_label[s][l]), beat_candidate[s][l])

                        draw_beat_label.append(_draw_beat_label.flatten())

                for i, ax in enumerate(axx):
                    start_buf = i * data_model.MAX_LEN_PLOT * sampling_rate
                    stop_buf = min((i + 1) * data_model.MAX_LEN_PLOT * sampling_rate, data_len)
                    if i == 0:
                        ax.text(0, np.mean(buf_ecg_org), note, ha='left', rotation=0, wrap=True)

                    plot_len = (data_model.MAX_LEN_PLOT * sampling_rate)
                    t = np.arange(start_buf, start_buf + plot_len, 1) / sampling_rate
                    if len(total_rhythm) > 0:
                        ax.plot(t, rhythm_candidate_pred_draw[start_buf: stop_buf], label=rhythm_main, linestyle='--')

                    buf_ecg_draw = buf_ecg_org.copy()
                    buf_ecg_draw = butter_bandpass_filter(buf_ecg_draw, 0.5, 40.0, sampling_rate)
                    buf_ecg_draw = np.clip(buf_ecg_draw, -5.0, 5.0)
                    # buf_ecg_draw = bwr(buf_ecg_draw, sampling_rate)
                    buf_frame = buf_ecg_draw[start_buf: stop_buf]
                    plot_len = len(buf_frame)

                    ax.plot(t, buf_frame, linestyle='-', linewidth=2.0)
                    if len(beats) > 0:
                        for d, _draw_beat_label in enumerate(draw_beat_label):
                            ax.plot(t, (_draw_beat_label[start_buf: stop_buf]) + 0.1 * d, label="BEAT-{}".format(d))

                        total_beat_draw = beats.copy()
                        index_draw = (total_beat_draw >= start_buf) == (total_beat_draw < stop_buf)
                        beats_draw = total_beat_draw[index_draw] - start_buf
                        symbols_draw = total_symbol[index_draw]
                        amps_draw = amps[index_draw]
                        for b, s, a in zip(beats_draw, symbols_draw, amps_draw):
                            if s == "V":
                                ax.annotate('{}\n{:.3f}'.format(s, a), xy=(t[b], buf_frame[b]), color='red',
                                            fontsize=8, fontweight='bold')
                            elif s == "A":
                                ax.annotate('{}\n{:.3f}'.format(s, a), xy=(t[b], buf_frame[b]), color='blue',
                                            fontsize=8, fontweight='bold')
                            elif s == "R":
                                ax.annotate('{}\n{:.3f}'.format(s, a), xy=(t[b], buf_frame[b]), color='green',
                                            fontsize=8, fontweight='bold')
                            else:
                                ax.annotate('{}\n{:.3f}'.format(s, a), xy=(t[b], buf_frame[b]), color='black',
                                            fontsize=8, fontweight='bold')

                    major_ticks = np.arange(start_buf, start_buf + plot_len,
                                            sampling_rate) / sampling_rate
                    minor_ticks = np.arange(start_buf, start_buf + plot_len,
                                            sampling_rate // 4) / sampling_rate
                    ax.set_xticks(major_ticks)
                    ax.set_xticks(minor_ticks, minor=True)
                    ax.grid(which='major', color='#CCCCCC', linestyle='--')
                    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
                    ax.legend(loc="lower right", prop={'size': 6})

                    if stop_buf == data_len:
                        break

                DEBUG_IMG = True
                if not DEBUG_IMG:
                    img_name = sub_save_image + "/{}".format(basename(file_name))

                    fig.savefig(img_name + ".svg", format='svg', dpi=1200)
                    plt.close(fig)
                else:
                    plt.show()

            if len(total_rhythm) == 0:
                total_rhythm = rhythm_candidate
                total_sample = sample
            else:
                total_rhythm = np.concatenate((total_rhythm, rhythm_candidate), axis=0)
                total_sample = np.concatenate((total_sample, sample), axis=0)

            if len(beats) > 0:
                beats = (beats * fs_origin) // sampling_rate
                beats += samp_from
                if len(total_symbol) == 0:
                    total_symbol = symbols
                    total_beat = beats
                else:
                    total_symbol = np.concatenate((total_symbol, symbols), axis=0)
                    total_beat = np.concatenate((total_beat, beats), axis=0)

            samp_from = samp_to

        except Exception as e:
            print("process_sample {}: {}".format(file_name, e))
            break

    total_beat = np.asarray(total_beat, dtype=int)
    total_symbol = np.asarray(total_symbol)

    group_index = np.where(abs(np.diff(total_rhythm)) > 0)[0] + 1
    group_rhythm = np.split(total_rhythm, group_index)
    group_sample = np.split(total_sample, group_index)
    total_sample = []
    total_rhythm = []
    for i in range(len(group_rhythm)):
        if len(group_rhythm[i]) > 0:
            sym = group_rhythm[i][0]
            sam = group_sample[i][0]
            if len(total_rhythm) == 0 or (len(total_rhythm) > 0 and total_rhythm[-1] != sym):
                total_rhythm.append(sym)
                total_sample.append(sam)

    # region MIX

    for i in range(len(total_sample)):
        st = total_sample[i]
        if (i + 1) >= len(total_sample):
            sp = header.sig_len
        else:
            sp = total_sample[i + 1]

        if rhythm_inv[total_rhythm[i]] == "AFIB":
            idx = (total_beat >= st) == (total_beat <= sp)
            if np.count_nonzero(idx):
                for d, b in enumerate(idx):
                    if b:
                        total_symbol[d] = 'N' if total_symbol[d] == 'A' else total_symbol[d]

        # elif rhythm_inv[total_rhythm[i]] == "OTHER":
        #     idx = (total_beat >= st) == (total_beat <= sp)
        #     if np.count_nonzero(idx):
        #         for d, b in enumerate(idx):
        #             if b:
        #                 total_symbol[d] = "Q"

        # elif rhythm_inv[total_rhythm[i]] == "VT":
        #     idx = (total_beat >= st) == (total_beat <= sp)
        #     if np.count_nonzero(idx):
        #         for d, b in enumerate(idx):
        #             if b:
        #                 total_symbol[d] = "V"
        #
        # elif rhythm_inv[total_rhythm[i]] == "SVT":
        #     idx = (total_beat >= st) == (total_beat <= sp)
        #     if np.count_nonzero(idx):
        #         for d, b in enumerate(idx):
        #             if b:
        #                 total_symbol[d] = "A"

    # endregion MIX

    return total_beat, total_symbol, total_sample, total_rhythm, fs_origin


def beat_classification(beat_model,
                        file_name,
                        channel_ecg,
                        beat_datastore,
                        overlap,
                        img_directory=None):
    """

    """
    header = wf.rdheader(file_name)
    fs_origin = header.fs
    samp_to = 0
    samp_from = 0
    total_beat = []
    total_symbol = []

    total_sample = []
    total_rhythm = []
    event_len = (data_model.EVENT_LEN_STANDARD * fs_origin)

    sampling_rate = beat_datastore['sampling_rate']
    beat_class = beat_datastore["beat_class"]
    beat_num_block = beat_datastore["num_block"]
    beat_feature_len = beat_datastore["feature_len"]
    beat_ebwr = beat_datastore["bwr"]
    beat_enorm = beat_datastore["norm"]
    if "BAND_PASS_FILTER" in beat_datastore.keys():
        beat_filter = beat_datastore["BAND_PASS_FILTER"]
        beat_clip = beat_datastore["CLIP_RANGE"]
    else:
        beat_filter = [1.0, 30.0]
        beat_clip = None

    beat_inv = {i: k for i, k in enumerate(beat_class.keys())}
    beat_ind = {k: i for i, k in enumerate(beat_class.keys())}

    while samp_to <= header.sig_len:
        try:
            if header.sig_len - samp_from <= 0:
                break

            # region Process
            samp_len = min(event_len, (header.sig_len - samp_from))
            samp_to = samp_from + samp_len
            record = wf.rdsamp(file_name, sampfrom=samp_from, sampto=samp_to, channels=[channel_ecg])
            # Avoid cases where the value is NaN
            buf_record = np.nan_to_num(record[0][:, 0])
            fs_origin = record[1].get('fs')

            if fs_origin != sampling_rate:
                buf_ecg_org, _ = resample_sig(buf_record, fs_origin, sampling_rate)
            else:
                buf_ecg_org = buf_record.copy()

            len_of_standard = int(data_model.EVENT_LEN_STANDARD * sampling_rate)

            len_of_buf = len(buf_ecg_org)
            if len_of_buf < len_of_standard:
                buf_ecg_org = np.concatenate((buf_ecg_org, np.full(len_of_standard - len_of_buf, buf_ecg_org[-1])))

            # endregion

            # region BEAT
            buf_ecg = butter_bandpass_filter(buf_ecg_org,
                                             beat_filter[0],
                                             beat_filter[1],
                                             sampling_rate)
            if beat_clip is not None:
                buf_ecg = np.clip(buf_ecg,
                                  beat_clip[0],
                                  beat_clip[1])

            buf_bwr_ecg = bwr(buf_ecg_org, sampling_rate)
            if beat_ebwr:
                buf_ecg = bwr(buf_ecg_org, sampling_rate)

            if beat_enorm:
                buf_ecg = norm(buf_ecg, int(data_model.NUM_NORMALIZATION * sampling_rate))

            data_len = len(buf_ecg)
            beat_label_len = beat_feature_len // beat_num_block
            data_index = np.arange(beat_feature_len)[None, :] + \
                         np.arange(0, data_len, beat_feature_len)[:, None]

            _samp_from = (samp_from * sampling_rate) // fs_origin
            _samp_to = (samp_to * sampling_rate) // fs_origin

            if data_model.MODE == 0:  # New mode
                buf_frame = []
                for fr in data_model.OFFSET_FRAME_BEAT:
                    if len(buf_frame) == 0:
                        buf_frame = np.concatenate((buf_ecg[fr:], np.full(fr, 0)))[data_index]
                    else:
                        buf_frame = np.concatenate(
                            (buf_frame, np.concatenate((buf_ecg[fr:], np.full(fr, 0)))[data_index]))

                buf_frame = np.asarray(buf_frame)
                group_beat_prob = beat_model.predict(buf_frame)
                group_beat_candidate = np.argmax(group_beat_prob, axis=-1)
                group_beat_candidate = group_beat_candidate.reshape((-1, len(data_index), beat_num_block))
                label_index = np.arange(beat_label_len)[None, :] + \
                              np.arange(0, beat_feature_len, beat_label_len)[:, None]

                group_bwr_frame = buf_bwr_ecg[data_index]
                beats = []
                symbols = []
                amps = []

                for beat_candidate in group_beat_candidate:
                    # beat_candidate = np.asarray(beat_candidate).reshape((-1, beat_num_block))
                    for group_beat, group_bwr_buff, group_offset in zip(beat_candidate,
                                                                        group_bwr_frame,
                                                                        data_index):
                        _group_offset = group_offset[label_index]
                        _index = np.where(abs(np.diff(group_beat)) > 0)[0] + 1
                        _group_beat = np.split(group_beat, _index)
                        _group_offset = np.split(_group_offset, _index)
                        for gbeat, goffset in zip(_group_beat, _group_offset):
                            if np.max(gbeat) > beat_ind["NOTABEAT"]:
                                goffset = np.asarray(goffset).flatten()
                                index_ext = goffset.copy()
                                if (goffset[0] - beat_label_len) >= 0:
                                    index_ext = np.concatenate((np.arange((goffset[0] - beat_label_len),
                                                                          goffset[0]), index_ext))

                                if (goffset[-1] + beat_label_len) < beat_feature_len:
                                    index_ext = np.concatenate((index_ext,
                                                                np.arange(goffset[-1], (goffset[-1] + beat_label_len))))

                                gbuff = np.asarray(group_bwr_buff[index_ext]).flatten()
                                flip_g = gbuff * -1.0
                                peaks_up = np.argmax(gbuff)
                                peaks_down = np.argmax(flip_g)
                                if abs(gbuff[peaks_up]) > abs(flip_g[peaks_down]):
                                    ma = abs(gbuff[peaks_up])
                                    peaks = peaks_up
                                else:
                                    ma = abs(flip_g[peaks_down])
                                    peaks = peaks_down

                                amps.append(ma)
                                beats.append(peaks + index_ext[0])
                                qr_count = Counter(gbeat)
                                qr = qr_count.most_common(1)[0][0]
                                symbols.append(beat_inv[qr])

                if len(beats) > 0:
                    symbols = [x for _, x in sorted(zip(beats, symbols))]
                    amps = [x for _, x in sorted(zip(beats, amps))]
                    beats = sorted(beats)

                    beats = np.asarray(beats, dtype=int)
                    symbols = np.asarray(symbols)
                    amps = np.asarray(amps)
                    index_artifact = symbols == 'ARTIFACT'
                    sample_artifact = []
                    if np.count_nonzero(index_artifact) > 0:
                        sample_artifact = np.zeros(data_len, dtype=int)
                        sample_artifact[beats[index_artifact]] = 1
                        label_index_artifact = np.arange(sampling_rate)[None, :] + \
                                               np.arange(0, data_len, sampling_rate)[:, None]
                        sample_artifact = sample_artifact[label_index_artifact]
                        sample_artifact = np.asarray([np.max(lbl) for lbl in sample_artifact], dtype=int)

                        index_del = np.where(symbols == 'ARTIFACT')[0]
                        symbols = np.delete(symbols, index_del)
                        beats = np.delete(beats, index_del)
                        amps = np.delete(amps, index_del)

                    if len(beats) > 0:
                        min_rr = data_model.MIN_RR_INTERVAL * sampling_rate
                        group_beats, group_symbols, group_amps, group_len = beat_cluster(beats,
                                                                                         symbols,
                                                                                         amps,
                                                                                         min_rr)
                        beats = []
                        symbols = []
                        amps = []
                        for _beat, _symbol, _amp in zip(group_beats, group_symbols, group_amps):
                            qr_count = Counter(_symbol)
                            qr = qr_count.most_common(1)[0][0]
                            symbols.append(qr)

                            p = np.argmax(_amp)
                            amps.append(max(_amp))
                            beats.append(_beat[p])

                        beats = np.asarray(beats)
                        symbols = np.asarray(symbols)
                        amps = np.asarray(amps)

                        beats, symbols, amps = beat_select(beats, symbols, amps, buf_bwr_ecg, sampling_rate)
            else:  # Old mode
                frame = buf_ecg[data_index]
                buf_ecg1 = np.concatenate((buf_ecg[3:], np.zeros(3)))
                frame2 = buf_ecg1[data_index]
                buf_ecg3 = np.concatenate((buf_ecg[5:], np.zeros(5)))
                frame3 = buf_ecg3[data_index]
                lenOfframe = len(frame)
                frame = np.concatenate((frame, frame2, frame3))
                group_beat_prob = beat_model.predict(frame)
                group_beat_candidate = np.argmax(group_beat_prob, axis=-1)
                beat_candidate = group_beat_candidate[:lenOfframe, :]
                beat_candidate2 = group_beat_candidate[lenOfframe: lenOfframe * 2, :]
                beat_candidate3 = group_beat_candidate[lenOfframe * 2:, :]

                label_index = np.arange(beat_label_len)[None, :] + \
                              np.arange(0, beat_feature_len, beat_label_len)[:, None]
                beats = []
                symbols = []
                amps = []
                for beats_group, beats_group2, beats_group3, buf_group, index_group in zip(beat_candidate, beat_candidate2,
                                                                                           beat_candidate3, frame,
                                                                                           data_index):
                    buf = buf_group[label_index]
                    idx = index_group[label_index]
                    for qr0, qr2, qr3, gr, id in zip(beats_group, beats_group2, beats_group3, buf, idx):
                        # qr_count = Counter([qr0, qr2, qr3])
                        # qr = qr_count.most_common(1)[0][0]
                        qr = max([qr0, qr2, qr3])
                        if beat_ind["NOTABEAT"] < qr:
                            flip_g = gr * -1.0
                            peaks = np.argmax(gr)
                            flip_peaks = np.argmax(flip_g)
                            peaks = np.concatenate(([peaks], [flip_peaks]))
                            ma = np.abs(gr[peaks] - np.mean(gr))
                            mb = np.argmax(ma)
                            amps.append(max(ma))
                            beats.append(peaks[mb] + id[0])
                            symbols.append(beat_inv[qr])

                beats = np.asarray(beats)
                symbols = np.asarray(symbols)
                index_artifact = symbols == 'ARTIFACT'
                sample_artifact = []
                if np.count_nonzero(index_artifact) > 0:
                    sample_artifact = np.zeros(data_len, dtype=int)
                    sample_artifact[beats[index_artifact]] = 1
                    label_index_artifact = np.arange(sampling_rate)[None, :] + \
                                           np.arange(0, data_len, sampling_rate)[:, None]
                    sample_artifact = sample_artifact[label_index_artifact]
                    sample_artifact = np.asarray([np.max(lbl) for lbl in sample_artifact], dtype=int)

                    index_del = np.where(symbols == 'ARTIFACT')[0]
                    symbols = np.delete(symbols, index_del)
                    beats = np.delete(beats, index_del)

                if len(beats) > 0:
                    symbols = np.asarray([x for _, x in sorted(zip(beats, symbols))])
                    amps = np.asarray([x for _, x in sorted(zip(beats, amps))])
                    beats = np.sort(beats)

                    group_index = np.where(abs(np.diff(beats)) > (data_model.MIN_RR_INTERVAL * sampling_rate))[0] + 1
                    group_beats = np.split(beats, group_index)
                    group_symbols = np.split(symbols, group_index)
                    group_amps = np.split(amps, group_index)
                    beats = []
                    symbols = []
                    amps = []
                    for _beat, _symbol, _amp in zip(group_beats, group_symbols, group_amps):
                        # occurence_count = Counter(_symbol)
                        # sym = occurence_count.most_common(1)[0][0]
                        iamp = np.argmax(_amp)
                        symbols.append(_symbol[iamp])
                        beats.append(_beat[iamp])
                        amps.append(np.max(_amp))

                    beats = np.asarray(beats)
                    symbols = np.asarray(symbols)
                    amps = np.asarray(amps)

            # endregion BEAT

            sample = (sample * fs_origin) // sampling_rate
            sample += samp_from

            rhythm_candidate_pred_draw = np.zeros(data_len)
            rhythm_candidate_pred_draw = rhythm_candidate_pred_draw[label_index]

            rhythm_candidate_pred_draw = rhythm_candidate_pred_draw.flatten()

            if len(beats) > 0:
                beats = (beats * fs_origin) // sampling_rate
                beats += samp_from
                if len(total_symbol) == 0:
                    total_symbol = symbols
                    total_beat = beats
                else:
                    total_symbol = np.concatenate((total_symbol, symbols), axis=0)
                    total_beat = np.concatenate((total_beat, beats), axis=0)

            samp_from = samp_to

        except Exception as e:
            print("process_sample {}: {}".format(file_name, e))
            break

    total_beat = np.asarray(total_beat, dtype=int)
    total_symbol = np.asarray(total_symbol)
    return total_beat, total_symbol, fs_origin


def process_beat_classification(process_index,
                                use_gpu_index,
                                file_list,
                                model_name,
                                checkpoint_dir,
                                datastore_dict,
                                ext_ai,
                                write_mit_annotation,
                                write_hes_annotation,
                                channel=0,
                                overlap=0,
                                memory=1024,
                                dir_image=None):
    """

    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(use_gpu_index)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
    tf.autograph.set_verbosity(1)

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)]
        )
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

    feature_len = datastore_dict["feature_len"]
    beat_class = datastore_dict["beat_class"]

    _qrs_model_path = model_name.split('_')
    func = ""
    m = 0
    for m in range(len(_qrs_model_path)):
        if _qrs_model_path[m].isnumeric():
            break
        else:
            func += _qrs_model_path[m] + "_"

    func = func[:-1]

    if 'beat' in func:
        tmp = checkpoint_dir.split('/')
        for i in tmp:
            try:
                day_export = datetime.datetime.strptime(i, '%y%m%d')
                break
            except:
                continue

        # day_export = checkpoint_dir.split('/')[6]
        # day_export = datetime.datetime.strptime(day_export, '%y%m%d')

        num_loop = int(_qrs_model_path[m])
        num_filters = np.asarray([int(i) for i in _qrs_model_path[m + 1].split('.')], dtype=int)
        try:
            from_logits = bool(int(_qrs_model_path[m + 2]))
        except:
            from_logits = False
        if day_export < datetime.datetime(2021, 12, 1):
            beat_model = getattr(model_old, func)(feature_len,
                                                  len(beat_class),
                                                  from_logits,
                                                  num_filters,
                                                  num_loop,
                                                  0.5,
                                                  False)
        else:
            beat_model = getattr(model, func)(feature_len,
                                              len(beat_class),
                                              from_logits,
                                              num_filters,
                                              num_loop,
                                              0.5,
                                              False)
            beat_model.summary()
            import glob
            checkpoint_dir = glob.glob(checkpoint_dir + '/*.h5')[0]
            # beat_model = keras.models.load_model(ckt)

        # beat_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        beat_model.load_weights(checkpoint_dir)
    else:
        return ""

    log_lines = []

    for file_name in file_list:

        start = time.perf_counter()
        if basename(file_name) == '114':
            channel_ecg = 1
        else:
            channel_ecg = channel

        # if basename(file_name) != "100":
        #     continue

        total_peak, total_symbol, fs_origin = beat_classification(beat_model,
                                                                  file_name,
                                                                  channel_ecg,
                                                                  datastore_dict,
                                                                  overlap,
                                                                  dir_image)
        if fs_origin > 0:
            if len(total_peak) == 0:
                total_peak = [0]
                total_symbol = ['N']

            str_log = '{} with {} events take {} seconds'.format(
                basename(dirname(file_name)) + '/' + basename(file_name),
                len(total_peak),
                time.perf_counter() - start)
            print(str_log)
            log_lines.append(str_log)
            dir_test = os.path.dirname(file_name)
            if write_mit_annotation:
                curr_dir = os.getcwd()
                os.chdir(dir_test + '/')
                annotation2 = wf.Annotation(record_name=basename(file_name),
                                            extension=ext_ai,
                                            sample=np.asarray(total_peak),
                                            symbol=np.asarray(total_symbol),
                                            fs=fs_origin)
                annotation2.wrann(write_fs=True)
                os.chdir(curr_dir)

    return log_lines


def process_beat_rhythm_classification(process_index,
                                       use_gpu_index,
                                       file_list,
                                       beat_datastore,
                                       beat_checkpoint,
                                       rhythm_datastore,
                                       rhythm_checkpoint,
                                       beat_ext_tech,
                                       beat_ext_ai,
                                       rhythm_ext_ai,
                                       tmp_directory,
                                       write_mit_annotation,
                                       write_hes_annotation,
                                       memory=1024):
    """

    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(use_gpu_index)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
    tf.autograph.set_verbosity(1)

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)]
        )
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

    # region BEAT
    beat_feature_len = beat_datastore["feature_len"]
    beat_class = beat_datastore["beat_class"]
    beat_model_name = beat_checkpoint.split('/')[-2]
    _model_path = beat_model_name.split('_')
    func = ""
    m = 0
    for m in range(len(_model_path)):
        if _model_path[m].isnumeric():
            break
        else:
            func += _model_path[m] + "_"

    func = func[:-1]

    if 'beat' in func:
        day_export = beat_checkpoint.split('/')[-6]
        day_export = datetime.datetime.strptime(day_export, '%y%m%d')

        num_loop = int(_model_path[m])
        num_filters = np.asarray([int(i) for i in _model_path[m + 1].split('.')], dtype=int)
        try:
            from_logits = bool(int(_model_path[m + 2]))
        except:
            from_logits = False

        if day_export < datetime.datetime(2021, 12, 1):
            beat_model = getattr(model_old, func)(beat_feature_len,
                                                  len(beat_class),
                                                  from_logits,
                                                  num_filters,
                                                  num_loop,
                                                  0.5,
                                                  False)
        else:
            beat_model = getattr(model, func)(beat_feature_len,
                                              len(beat_class),
                                              from_logits,
                                              num_filters,
                                              num_loop,
                                              0.5,
                                              False)

        beat_model.load_weights(tf.train.latest_checkpoint(beat_checkpoint)).expect_partial()
    else:
        return None

    # endregion BEAT

    # region RHYTHM
    rhythm_feature_len = rhythm_datastore["feature_len"]
    rhythm_class = rhythm_datastore["rhythm_class"]
    rhythm_model_name = rhythm_checkpoint.split('/')[-2]
    _model_path = rhythm_model_name.split('_')
    func = ""
    m = 0
    for m in range(len(_model_path)):
        if _model_path[m].isnumeric():
            break
        else:
            func += _model_path[m] + "_"

    func = func[:-1]

    if 'rhythm' in func:
        day_export = beat_checkpoint.split('/')[-6]
        day_export = datetime.datetime.strptime(day_export, '%y%m%d')

        num_loop = int(_model_path[m])
        num_filters = np.asarray([int(i) for i in _model_path[m + 1].split('.')], dtype=int)
        try:
            from_logits = bool(int(_model_path[m + 2]))
        except:
            from_logits = False

        if day_export < datetime.datetime(2021, 12, 1):
            rhythm_model = getattr(model_old, func)(rhythm_feature_len,
                                                    len(rhythm_class),
                                                    from_logits,
                                                    num_filters,
                                                    num_loop)
        else:
            rhythm_model = getattr(model, func)(rhythm_feature_len,
                                                len(rhythm_class),
                                                from_logits,
                                                num_filters,
                                                num_loop)

        rhythm_model.load_weights(tf.train.latest_checkpoint(rhythm_checkpoint)).expect_partial()
    else:
        return None

    # endregion RHYTHM

    log_lines = []
    lst_file_name = []
    lst_symbol_true = []
    lst_symbol_pred = []
    for file_name in file_list:
        if beat_ext_tech is not None:
            studyFid, eventFid, hasComplexBeat, _ = data_model.get_studyid(file_name + ".{}".format(EXT_BEAT_EVAL))
            if hasComplexBeat:
                print(file_name)
                continue

            header = wf.rdheader(file_name)
            comments = header.comments
            lst_channel_event = []
            lst_from_event = []
            lst_to_event = []
            lst_channel_calipers = []
            for i, e in enumerate(comments):
                if "Mark_Channel_" in e:
                    split_txt = e.split(':')
                    if len(split_txt) > 1 and len(split_txt[1]) > 0:
                        lst_channel_event.append(int(split_txt[1]))

                if "Mark_From_" in e:
                    split_txt = e.split(':')
                    if len(split_txt) > 1 and len(split_txt[1]) > 0:
                        lst_from_event.append(int(split_txt[1]))

                if "Mark_To_" in e:
                    split_txt = e.split(':')
                    if len(split_txt) > 1 and len(split_txt[1]) > 0:
                        lst_to_event.append(int(split_txt[1]))

                if "Calipers_Channel_" in e:
                    split_txt = e.split(':')
                    if len(split_txt) > 1 and len(split_txt[1]) > 0:
                        lst_channel_calipers.append(int(split_txt[1]))

                if "Technician_Comment" in e:
                    split_txt = e.split(':')
                    if len(split_txt) > 1:
                        technician_comment = split_txt[1]

            event_channel = -1
            from_event = -1
            to_event = -1
            if len(lst_channel_event) > 1:
                for ch, fr, to in zip(lst_channel_event, lst_from_event, lst_to_event):
                    for cp in lst_channel_calipers:
                        if ch == cp:
                            event_channel = ch
                            from_event = fr
                            to_event = to
                            break

                    if event_channel >= 0:
                        break

                if event_channel < 0:
                    event_channel = lst_channel_event[0]
                    from_event = lst_from_event[0]
                    to_event = lst_to_event[0]
            else:
                event_channel = lst_channel_event[0]
                from_event = lst_from_event[0]
                to_event = lst_to_event[0]

            qa_channel, beat_true, symbol_true, _ = \
                data_model.get_annotations(file_name + ".{}".format(EXT_BEAT_EVAL),
                                           int(data_model.EVENT_LEN_STANDARD * header.fs))

            if qa_channel >= 0:
                event_channel = qa_channel
            elif event_channel < 0:
                continue
        else:
            event_channel = 0

        start = time.perf_counter()
        total_peak, total_symbol, total_sample, total_rhythm, fs_origin = beat_rhythm_classification(beat_model,
                                                                                                     beat_datastore,
                                                                                                     rhythm_model,
                                                                                                     rhythm_datastore,
                                                                                                     file_name,
                                                                                                     event_channel)
        if len(total_peak) == 0:
            total_peak = [0]
            total_symbol = ['N']

        str_log = '{} with {} beats take {} seconds'.format(
            basename(dirname(file_name)) + '/' + basename(file_name),
            len(total_peak),
            time.perf_counter() - start)
        print(str_log)
        log_lines.append(str_log)
        if tmp_directory is None:
            dir_test = os.path.dirname(file_name)
        else:
            dir_test = tmp_directory

        if write_mit_annotation:
            curr_dir = os.getcwd()
            os.chdir(dir_test + '/')
            if beat_ext_tech is not None:
                copyfile(file_name + ".hea", basename(file_name) + ".hea")
                annotation1 = wf.Annotation(record_name=basename(file_name),
                                            extension=beat_ext_tech,
                                            sample=np.asarray(beat_true),
                                            symbol=np.asarray(symbol_true),
                                            fs=fs_origin)
                annotation1.wrann(write_fs=True)

            annotation2 = wf.Annotation(record_name=basename(file_name),
                                        extension=beat_ext_ai,
                                        sample=np.asarray(total_peak),
                                        symbol=np.asarray(total_symbol),
                                        fs=fs_origin)
            annotation2.wrann(write_fs=True)

            # physionet_annotation = ['(NOISE', '(N', '(AFIB', '(SVTA', '(VT', '(BI', '(BII', '(BIII']
            # annotation3 = wf.Annotation(record_name=basename(file_name),
            #                             extension=rhythm_ext_ai,
            #                             sample=np.asarray(total_sample),
            #                             symbol=np.asarray(["+" for _ in range(len(total_sample))]),
            #                             aux_note=np.array([physionet_annotation[e] for e in total_rhythm]),
            #                             fs=fs_origin)
            # annotation3.wrann(write_fs=True)
            os.chdir(curr_dir)
        if beat_ext_tech is not None:
            ref_symbol, compare_symbol = bxb(predict_sample=np.asarray(total_peak),
                                             predict_symbol=np.asarray(total_symbol),
                                             ref_sample=np.asarray(beat_true),
                                             ref_symbol=np.asarray(symbol_true),
                                             epsilon=fs_origin * 0.15)
            if len(ref_symbol) != len(compare_symbol):
                print("ERR in bxb: len(ref_symbol) != len(compare_symbol)")

            lst_file_name.append(basename(file_name))
            lst_symbol_true += ref_symbol
            lst_symbol_pred += compare_symbol

    return log_lines, lst_file_name, lst_symbol_true, lst_symbol_pred


def process_beat_classification_2(process_index,
                                use_gpu_index,
                                file_list,
                                beat_datastore,
                                beat_checkpoint,
                                beat_ext_tech,
                                beat_ext_ai,
                                write_mit_annotation,
                                memory=2048):
    """

    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(use_gpu_index)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
    tf.autograph.set_verbosity(1)

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)]
        )
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

    # region BEAT
    beat_feature_len = beat_datastore["feature_len"]
    beat_class = beat_datastore["beat_class"]
    beat_model_name = beat_checkpoint.split('/')[-2]
    _model_path = beat_model_name.split('_')
    func = ""
    m = 0
    for m in range(len(_model_path)):
        if _model_path[m].isnumeric():
            break
        else:
            func += _model_path[m] + "_"

    func = func[:-1]

    if 'beat' in func:
        day_export = beat_checkpoint.split('/')[-6]
        day_export = datetime.datetime.strptime(day_export, '%y%m%d')

        num_loop = int(_model_path[m])
        num_filters = np.asarray([int(i) for i in _model_path[m + 1].split('.')], dtype=int)
        try:
            from_logits = bool(int(_model_path[m + 2]))
        except:
            from_logits = False

        if day_export < datetime.datetime(2021, 12, 1):
            beat_model = getattr(model_old, func)(beat_feature_len,
                                                  len(beat_class),
                                                  from_logits,
                                                  num_filters,
                                                  num_loop,
                                                  0.5,
                                                  False)
        else:
            beat_model = getattr(model, func)(beat_feature_len,
                                              len(beat_class),
                                              from_logits,
                                              num_filters,
                                              num_loop,
                                              0.5,
                                              False)
            beat_model.summary()
            import glob
            checkpoint_dir = glob.glob(beat_checkpoint + '/*.h5')[0]

        beat_model.load_weights(checkpoint_dir)
        # beat_model.load_weights(tf.train.latest_checkpoint(beat_checkpoint)).expect_partial()
    else:
        return None

    # endregion BEAT

    log_lines = []
    lst_file_name = []
    lst_symbol_true = []
    lst_symbol_pred = []
    for file_name in file_list:
        event_channel = 0
        start = time.perf_counter()
        total_peak, total_symbol, fs_origin = beat_classification_2(beat_model,
                                                                    file_name,
                                                                    event_channel,
                                                                    beat_datastore
                                                                    )
        print("No. beats of file_name: {}".format(len(total_peak)))
        if len(total_peak) == 0:
            total_peak = [0]
            total_symbol = ['N']

        if 'export_noise' in file_name:
            total_symbol = ['Q'] * len(total_peak)

        str_log = '{} with {} beats take {} seconds'.format(
            basename(dirname(file_name)) + '/' + basename(file_name),
            len(total_peak),
            time.perf_counter() - start)

        if write_mit_annotation:
            annotation2 = wf.Annotation(record_name=basename(file_name),
                                        extension=beat_ext_ai,
                                        sample=np.asarray(total_peak),
                                        symbol=np.asarray(total_symbol),
                                        fs=fs_origin)
            annotation2.wrann(write_fs=True, write_dir=dirname(file_name))

    return log_lines, lst_file_name, lst_symbol_true, lst_symbol_pred
