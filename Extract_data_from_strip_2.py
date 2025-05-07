import os

import numpy as np
import wfdb as wf

from matplotlib import pyplot as plt
from glob import glob
from collections import Counter
from utils.reprocessing import butter_bandpass_filter

DATA_PATH = "/mnt/4T_DATA/LLM/include-strip2/"
OUTPUT_PATH = "/mnt/4T_DATA/DATA_4TINYML/tiny_hb3_data_strip_2/"
DEBUG = False
LEN = 5 #sec

def extract_data(data_path=DATA_PATH,
                   output_data_path=OUTPUT_PATH,
                   output_info_path=OUTPUT_PATH):
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

    if not os.path.exists(output_info_path):
        os.mkdir(output_info_path)

    folders = os.listdir(data_path)
    fp_log = open(f"{output_data_path}/log_err.csv", "w")
    sta_beat = dict()
    sta_beat["Total"] = dict()
    """
    /mnt/4T_DATA/LLM/include-strip2//397447/67651668798462b95825c4cb/event-strip-captured-2024-12-19-08-41-40-utc-07
    ('File formats must be valid WFDB dat formats:', ['8', '16', '32', '61', '80', '160', '212', '310', '311', '24', '508', '516', '524'])
    """
    for k, i_folder in enumerate(folders):
        print(f"{k}/{len(folders)}: {i_folder}")
        files = glob(f"{DATA_PATH}/{i_folder}/*/*.hea")
        # files = glob(f"{DATA_PATH}/377796/*/*.hea")
        # files = glob(f"/mnt/4T_DATA/LLM/include-strip2/388215/6736f6b690e7350001219f47/event-strip-captured-2024-11-15-00-59-00-utc-06.hea")
        for no, file in enumerate(files):
            # print(f"{no}/{len(files)}")
            try:
                file = file[:-4]
                try:
                    header = wf.rdheader(file)
                    # continue
                except:
                    print(f"{k}/{len(folders)}: {i_folder}")

                if header.fs != 250:
                    continue

                startSample = 0
                stop_sampe = 0
                for line in header.comments:
                    if "startSample" in line:
                        startSample = int(line.split(":")[-1])
                    elif "stopSample" in line:
                        stopSample = int(line.split(":")[-1])
                    elif "channel" in line:
                        channel = int(line.split(":")[-1])
                    elif "comment" in line:
                        type = line.split(":")[-1]
                    elif "eventType":
                        eventType = line.split(":")[-1]
                        eventType = eventType.replace(" ", "")
                if startSample < 0:
                    continue

                signal, _ = wf.rdsamp(file)
                ann = wf.rdann(file, 'atr')
                symbols = np.asarray(ann.symbol)
                samples = np.asarray(ann.sample)
                name = file.replace(DATA_PATH, '')
                if DEBUG:
                    plt.figure(1)
                    plt.title(f"{name}\n{eventType}")
                    for ch in range(len(header.sig_name)):
                        if ch == 0:
                            ax = plt.subplot(len(header.sig_name) * 100 + 11 + ch)
                        else:
                            plt.subplot(len(header.sig_name) * 100 + 11 + ch, sharex=ax, sharey=ax)

                        plt.plot(butter_bandpass_filter(signal[:, ch], 1, 40, 250))
                        plt.plot(samples, signal[:, ch][samples], '*r')
                        [plt.annotate(symbols[i], (samples[i], 0.5)) for i, symbol in enumerate(symbols)]
                        plt.axvspan(startSample, stopSample, alpha=0.5)

                    # plt.show()

                if (stopSample - startSample) < (LEN * header.fs):
                    startSample -= (LEN*header.fs - (stopSample - startSample))//2

                flag = True
                start = startSample
                cnt_segment = 0
                while flag:
                    if start > stopSample:
                        start = stopSample - LEN * header.fs

                    if start+LEN*header.fs > len(signal):
                        break

                    signal_segment = signal[start:start+LEN*header.fs, :]
                    samples_segment = samples[np.flatnonzero((samples >= start) & (samples <= start + LEN * header.fs))] - start
                    symbols_segment = symbols[np.flatnonzero((samples >= start) & (samples <= start + LEN * header.fs))]
                    tmp = name.split("/")

                    if DEBUG:
                        plt.figure(2)
                        plt.title(f"{name}")
                        for ch in range(len(header.sig_name)):
                            if ch == 0:
                                ax = plt.subplot(len(header.sig_name) * 100 + 11 + ch)
                            else:
                                plt.subplot(len(header.sig_name) * 100 + 11 + ch, sharex=ax, sharey=ax)

                            plt.plot(butter_bandpass_filter(signal_segment[:, ch], 1, 40, 250))
                            plt.plot(samples_segment, butter_bandpass_filter(signal_segment[:, ch], 1, 40, 250)[samples_segment], '*r')
                            [plt.annotate(symbols_segment[i], (samples_segment[i], 0.5)) for i, symbol in enumerate(symbols_segment)]
                        plt.show()


                    output_dir = f"{output_data_path}/{eventType}/{tmp[0]}/{tmp[1]}"
                    output_dir = output_dir.replace(" ", "")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    sta_segment = dict(Counter(symbols_segment))
                    try:
                        if  not eventType in list(sta_beat.keys()):
                            sta_beat[eventType]= dict()
                            sta_beat[eventType]["studies"] = []
                    except:
                        pass

                    if not tmp[1] in sta_beat[eventType]["studies"]:
                        sta_beat[eventType]["studies"].append(tmp[1])

                    for key in sta_segment.keys():
                        try:
                            sta_beat[eventType][key] += sta_segment[key]
                        except:
                            sta_beat[eventType][key] = sta_segment[key]

                        try:
                            sta_beat["Total"][key] += sta_segment[key]
                        except:
                            sta_beat["Total"][key] = sta_segment[key]

                    if len(symbols_segment) == 0:
                        symbols_segment = np.asarray(["+"])
                        samples_segment = np.asarray([0])

                    if not os.path.exists(f"{output_data_path}/{tmp[-1]}_{cnt_segment}"):
                        annotations = wf.Annotation(record_name=f"{tmp[-1]}_{cnt_segment}",
                                                    symbol=symbols_segment,
                                                    sample=samples_segment,
                                                    extension='atr',
                                                    fs=header.fs,
                                                    )
                        annotations.wrann(write_dir=output_dir, write_fs=True)
                        wf.wrsamp(record_name=f"{tmp[-1]}_{cnt_segment}",
                                  p_signal=signal_segment,
                                  fs=header.fs,
                                  units=header.units,
                                  sig_name=header.sig_name,
                                  adc_gain=header.adc_gain,
                                  fmt=header.fmt,
                                  baseline=header.baseline,
                                  comments=[f"from: {name}",
                                            f"eventType: {eventType}",
                                            f"channel: {channel}",
                                            ],
                                  write_dir=output_dir
                                  )

                    if start + LEN * header.fs >= stopSample:
                        flag =False
                    else:
                        start += LEN * LEN * header.fs
                        cnt_segment += 1

                    plt.close(1)
                    plt.close(2)

            except Exception as err:
                print(f"{file}\n{err}")
                fp_log.writelines(f"{file}, {err}\n")
                pass

    import json
    fp = open(output_info_path + '/log_info_data.json', 'w')
    fp.write(json.dumps(sta_beat, indent=4))
    fp.close()


if __name__ == '__main__':
    extract_data()