from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import datetime
import json
import math
import os
import re
import sys
from functools import partial
from glob import glob
from random import shuffle
import numpy as np

import model as beat_model
import inputs as dat_model
from utils.logging import TextLogging

import keras


def _calc_num_steps(num_samples, batch_size):
    return (num_samples + batch_size - 1) // batch_size


def _get_tfrecord_filenames(dir_path, is_training):
    if not os.path.exists(dir_path):
        raise FileNotFoundError("{}; No such file or directory.".format(dir_path))

    filenames = sorted(glob(os.path.join(dir_path, "*.tfrecord")))
    if not filenames:
        raise FileNotFoundError("No TFRecords found in {}".format(dir_path))

    if is_training:
        shuffle(filenames)

    return filenames


def format_metrics(metrics, sep="; "):
    return sep.join("{}: {:.6f}".format(k, metrics[k]) for k in sorted(metrics.keys()))


def print_cm(cm, labels, normalize=False, hide_zeroes=False, hide_diagonal=False,
             hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [12])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    cm_str = "    " + empty_cell + " "
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        cm_str += "%{0}s".format(columnwidth) % label + " "
    print()
    cm_str += '\n'
    if normalize:
        cm = cm / np.reshape(cm.astype(np.float).sum(axis=1), (-1, 1))
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        cm_str += "    %{0}s".format(columnwidth) % label1 + " "
        for j in range(len(labels)):
            cell = "%{0}.3f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
            cm_str += cell + " "
        print()
        cm_str += '\n'

    return cm_str


def train_beat_classification(use_gpu_index,
                              model_name,
                              log_dir,
                              model_dir,
                              datastore_dict,
                              resume_from,
                              train_directory,
                              eval_directory,
                              batch_size,
                              valid_freq,
                              patience,
                              epoch_num):
    """

    :param use_gpu_index:
    :param model_name:
    :param log_dir:
    :param model_dir:
    :param datastore_dict:
    :param resume_from:
    :param train_directory:
    :param eval_directory:
    :param batch_size:
    :param valid_freq:
    :param patience:
    :param epoch_num:
    :return:
    """
    print('model_dir: {}\n'.format(model_dir))
    feature_len = datastore_dict["feature_len"]
    beat_class = datastore_dict["beat_class"]
    num_block = datastore_dict["num_block"]
    if "val_class" in datastore_dict.keys():
        val_class = datastore_dict["val_class"]
    else:
        val_class = datastore_dict["beat_class"]

    last_checkpoint_dir = model_dir + '/last'
    best_squared_error_checkpoint_dir = model_dir + '/best_squared_error_metric'
    best_loss_checkpoint_dir = model_dir + '/best_loss'
    best_f1_checkpoint_dir = model_dir + '/best_f1'
    for i in [last_checkpoint_dir, best_squared_error_checkpoint_dir,
              best_loss_checkpoint_dir, best_f1_checkpoint_dir]:
        if not os.path.exists(i):
            os.makedirs(i)

    bk_metric = None
    if os.path.exists('{}/{}_bk_metric.txt'.format(log_dir, model_name)):
        with open('{}/{}_bk_metric.txt'.format(log_dir, model_name), 'r') as json_file:
            bk_metric = json.load(json_file)
            try:
                if bk_metric["stop_train"]:
                    return True
            except:
                bk_metric["stop_train"] = False

    fieldnames = ['epoch',
                  'accuracy_train', 'loss_train', 'precision_train', 'recall_train',
                  'accuracy_eval', 'loss_eval', 'precision_eval', 'recall_eval',
                  'squared_error_metrics_train', 'squared_error_metrics_eval',
                  'f1_score_metrics_train', 'f1_score_metrics_eval']

    if not os.path.exists(log_dir + '/{}_log.csv'.format(model_name)):
        with open(log_dir + '/{}_log.csv'.format(model_name), mode='a+') as report_file:
            report_writer = csv.DictWriter(report_file, fieldnames=fieldnames)
            report_writer.writeheader()

    log_train = TextLogging(log_dir + '/{}_training_log.txt'.format(model_name), 'a+')

    log_train.write_mylines('Begin : {}\n'.format(str(datetime.datetime.now())))
    log_train.write_mylines('Batch size : {}\n'.format(batch_size))
    log_train.write_mylines('Epoch : {}\n'.format(epoch_num))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(use_gpu_index)
    import tensorflow as tf
    import logging
    from tensorflow.python.autograph.core import ag_ctx
    from tensorflow.python.autograph.impl import api as autograph
    from tensorflow.python.keras.utils import losses_utils
    from tensorflow.python.keras.utils import metrics_utils
    from tensorflow.python.ops import math_ops
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(1)

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print('Use GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        print('Use CPU')

    class ConfusionMatrix(keras.metrics.Metric):
        def __init__(self, classes, name='confusion_matrix'):
            super(ConfusionMatrix, self).__init__(name=name)
            self.save_matrix = self.add_weight(shape=(classes, classes), name='cm',
                                               initializer='zeros', dtype=tf.int32)
            self.num_of_class = classes

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)
            # y_true = keras.backend.flatten(y_true)
            # y_pred = keras.backend.flatten(y_pred)
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])

            confusion_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_pred,
                                                        num_classes=self.num_of_class)
            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, self.dtype)
                sample_weight = tf.broadcast_to(sample_weight, confusion_matrix.shape)
                confusion_matrix = tf.multiply(confusion_matrix, sample_weight)
            self.save_matrix.assign_add(confusion_matrix)

        def result(self):
            return self.save_matrix

        def reset_states(self):
            keras.backend.set_value(self.save_matrix, np.zeros((self.num_of_class, self.num_of_class)))

    class CustomCallback(keras.callbacks.Callback):
        def __init__(self,
                     model_name,
                     log_train,
                     stopped_epoch,
                     last_checkpoint_dir,
                     best_loss_checkpoint_dir,
                     best_squared_error_checkpoint_dir,
                     best_f1_checkpoint_dir,
                     bk_metric,
                     lbl_train,
                     lbl_val,
                     log_dir,
                     field_name,
                     best_loss=-1,
                     best_squared_error_metrics=-1,
                     best_f1_score_metrics=-1,
                     valid_freq=6,
                     patience=3,
                     length_train=None,
                     length_valid=None,
                     tensorboard_dir=None
                     ):
            """

            :param model_name:
            :param log_train:
            :param stopped_epoch:
            :param last_checkpoint_dir:
            :param best_loss_checkpoint_dir:
            :param best_squared_error_checkpoint_dir:
            :param best_f1_checkpoint_dir:
            :param bk_metric:
            :param lbl_train:
            :param log_dir:
            :param field_name:
            :param best_loss:
            :param best_squared_error_metrics:
            :param best_f1_score_metrics:
            :param valid_freq:
            :param length_train:
            :param length_valid:
            :param tensorboard_dir:
            """
            super(CustomCallback, self).__init__()
            self.bk_metric = bk_metric
            self.log_train = log_train
            self.last_checkpoint_dir = last_checkpoint_dir
            self.best_loss_checkpoint_dir = best_loss_checkpoint_dir
            self.best_squared_error_checkpoint_dir = best_squared_error_checkpoint_dir
            self.best_f1_checkpoint_dir = best_f1_checkpoint_dir
            self.model_name = model_name
            self.stopped_epoch = stopped_epoch
            self.lbl_train = lbl_train
            self.lbl_val = lbl_val
            self.best_loss = best_loss
            self.best_squared_error_metrics = best_squared_error_metrics
            self.best_f1_score_metrics = best_f1_score_metrics
            self.log_dir = log_dir
            self.fieldnames = field_name
            self.train_progressbar = None
            self.length_train = length_train
            self.length_valid = length_valid
            self.progress = 0
            self.valid_freq = valid_freq
            self.patience = patience
            self.wait = 0
            self.epoch_early_stopping = 0
            self.tensorboard_dir = tensorboard_dir
            self.best_weights = None
            self.last_weights = None

        def on_test_batch_end(self, batch, logs=None):
            self.progress += 1
            if self.length_train is not None and self.length_valid is not None:
                self.train_progressbar.update(self.progress)

        def on_train_batch_end(self, batch, logs=None):
            self.progress += 1
            if self.length_train is not None:
                self.train_progressbar.update(self.progress)

        def on_epoch_begin(self, epoch, logs=None):
            epoch += 1
            print("Epoch {}/{}".format(epoch, self.stopped_epoch))
            self.log_train.write_mylines("Epoch {}/{}\n".format(epoch, self.stopped_epoch))
            self.progress = 0
            if epoch >= self.valid_freq and epoch % self.valid_freq == 0:
                if self.length_train is not None and self.length_valid is not None:
                    self.train_progressbar = keras.utils.Progbar(
                        self.length_train + self.length_valid)
            else:
                if self.length_train is not None:
                    self.train_progressbar = keras.utils.Progbar(self.length_train)

        def on_epoch_end(self, epoch, logs=None):
            epoch += 1
            report_row = dict()
            report_row['epoch'] = epoch
            report_row['accuracy_train'] = logs['accuracy']
            report_row['loss_train'] = logs['loss']
            report_row['precision_train'] = logs['precision']
            report_row['recall_train'] = logs['recall']

            train_metrics = {
                'accuracy': logs['accuracy'],
                'loss': logs['loss'],
                'precision': logs['precision'],
                'recall': logs['recall'],
            }
            print("Training " + format_metrics(train_metrics))
            self.log_train.write_mylines("Training " + format_metrics(train_metrics) + '\n')

            confusion_matrix = logs['confusion_matrix']
            cm_str = print_cm(confusion_matrix, self.lbl_train.keys(), False)
            self.log_train.write_mylines("Confusion \n" + cm_str + '\n')

            for f in os.listdir(self.last_checkpoint_dir):
                os.remove(os.path.join(self.last_checkpoint_dir, f))

            self.model.save_weights(
                os.path.join(self.last_checkpoint_dir, self.model_name + "-epoch-{}.weights.h5".format(epoch)))

            confusion_matrix = np.asarray(confusion_matrix)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TPR = TP / (TP + FN)
            PPV = TP / (TP + FP)
            squared_error_metrics_train = 0
            f1_score_metrics_train = 0
            for i, c in enumerate(self.lbl_train.keys()):
                if (1 - TPR[i]) > 0 and (1 - PPV[i]) > 0:
                    squared_error_metrics_train += (1 - TPR[i]) * (1 - TPR[i]) + (1 - PPV[i]) * (1 - PPV[i])
                    f1_score_metrics_train += 2 * TPR[i] * PPV[i] / (TPR[i] + PPV[i])
                    print('{} - se: {}; p+: {}; f1: {}'.format(c, TPR[i], PPV[i],
                                                               2 * TPR[i] * PPV[i] / (TPR[i] + PPV[i])))
                    self.log_train.write_mylines(
                        '{} - se: {}; p+: {}; f1: {}'.format(c, TPR[i], PPV[i], 2 * TPR[i] * PPV[i]
                                                             / (TPR[i] + PPV[i])) + '\n')
                else:
                    print('{} - se: {}; p+: {}; f1: {}'.format(c, TPR[i], PPV[i],
                                                               2 * TPR[i] * PPV[i] / (TPR[i] + PPV[i])))
                    self.log_train.write_mylines(
                        '{} - se: {}; p+: {}; f1: {}'.format(c, TPR[i], PPV[i], 2 * TPR[i] * PPV[i]
                                                             / (TPR[i] + PPV[i])) + '\n')
                    squared_error_metrics_train = np.nan
                    f1_score_metrics_train = np.nan
                    break

            f1_score_metrics_train = f1_score_metrics_train / len(self.lbl_train.keys())
            if self.tensorboard_dir is not None:
                with tf.summary.create_file_writer(self.tensorboard_dir + '/train').as_default():
                    tf.summary.scalar('squared_error', squared_error_metrics_train, step=epoch - 1)
                    tf.summary.scalar('f1_score', f1_score_metrics_train, step=epoch - 1)

            report_row['squared_error_metrics_train'] = squared_error_metrics_train
            report_row['f1_score_metrics_train'] = f1_score_metrics_train

            print('squared_error_metrics_train :{}'.format(squared_error_metrics_train))
            print('f1_score_metrics_train :{}'.format(f1_score_metrics_train))
            self.log_train.write_mylines(
                'squared_error_metrics_train :{}'.format(squared_error_metrics_train) + '\n')
            self.log_train.write_mylines(
                'f1_score_metrics_train :{}'.format(f1_score_metrics_train) + '\n')
            # Eval region
            if epoch >= self.valid_freq and epoch % self.valid_freq == 0:
                val_metrics = {
                    'accuracy': logs['val_accuracy'],
                    'loss': logs['val_loss'],
                    'precision': logs['val_precision'],
                    'recall': logs['val_recall']
                }
                print("Validation " + format_metrics(val_metrics))
                str_val_metrics = format_metrics(val_metrics)
                num_val_metrics = re.findall(r'\d+\.\d+', str_val_metrics)
                self.log_train.write_mylines("Validation " + format_metrics(val_metrics) + '\n')

                confusion_matrix = logs['val_confusion_matrix']
                cm_str = print_cm(confusion_matrix, self.lbl_train.keys(), False)
                self.log_train.write_mylines("Confusion \n" + cm_str + '\n')

                report_row['accuracy_eval'] = val_metrics['accuracy']
                report_row['recall_eval'] = val_metrics['recall']
                report_row['precision_eval'] = val_metrics['precision']
                report_row['loss_eval'] = val_metrics['loss']

                confusion_matrix = np.asarray(confusion_matrix)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TPR = TP / (TP + FN)
                PPV = TP / (TP + FP)
                squared_error_metrics_eval = 0
                f1_score_metrics_eval = 0
                for i, c in enumerate(self.lbl_val.keys()):
                    if (1 - TPR[i]) > 0 and (1 - PPV[i]) > 0:
                        squared_error_metrics_eval += (1 - TPR[i]) * (1 - TPR[i]) + (1 - PPV[i]) * (
                                1 - PPV[i])
                        f1_score_metrics_eval += 2 * TPR[i] * PPV[i] / (TPR[i] + PPV[i])
                        print('{} - se: {}; p+: {}; f1: {}'.format(c, TPR[i], PPV[i],
                                                                   2 * TPR[i] * PPV[i] / (
                                                                           TPR[i] + PPV[i])))
                        self.log_train.write_mylines(
                            '{} - se: {}; p+: {}; f1: {}'.format(c, TPR[i], PPV[i], 2 * TPR[i] * PPV[i]
                                                                 / (TPR[i] + PPV[i])) + '\n')
                    else:
                        print('{} - se: {}; p+: {}; f1: {}'.format(c, TPR[i], PPV[i],
                                                                   2 * TPR[i] * PPV[i] / (
                                                                           TPR[i] + PPV[i])))
                        self.log_train.write_mylines('{} - se: {}; p+: {}; f1: {}'
                                                     .format(c, TPR[i], PPV[i],
                                                             2 * TPR[i] * PPV[i] / (
                                                                     TPR[i] + PPV[i])) + '\n')
                        squared_error_metrics_eval = np.nan
                        f1_score_metrics_eval = np.nan
                        break

                f1_score_metrics_eval = f1_score_metrics_eval / len(self.lbl_val.keys())

                if self.tensorboard_dir is not None:
                    with tf.summary.create_file_writer(
                            self.tensorboard_dir + '/validation').as_default():
                        tf.summary.scalar('squared_error', squared_error_metrics_eval, step=epoch - 1)
                        tf.summary.scalar('f1_score', f1_score_metrics_eval, step=epoch - 1)

                report_row['squared_error_metrics_eval'] = squared_error_metrics_eval
                report_row['f1_score_metrics_eval'] = f1_score_metrics_eval
                print('squared_error_metrics_eval :{}'.format(squared_error_metrics_eval))
                print('f1_score_metrics_eval :{}'.format(f1_score_metrics_eval))
                self.log_train.write_mylines(
                    'squared_error_metrics_eval :{}'.format(squared_error_metrics_eval) + '\n')
                self.log_train.write_mylines(
                    'f1_score_metrics_eval :{}'.format(f1_score_metrics_eval) + '\n')
                # endregion Eval

                # region Save model
                if self.best_loss < 0 or float(num_val_metrics[1]) < self.best_loss:
                    self.log_train.write_mylines(
                        "======================================================================\n")
                    self.log_train.write_mylines(
                        "Found better checkpoint! Saving to {}\n".format(self.best_loss_checkpoint_dir))
                    self.log_train.write_mylines(
                        "======================================================================\n")
                    print("===========================================================================")
                    print("Found loss better checkpoint! Saving to {}".format(
                        self.best_loss_checkpoint_dir))
                    print("===========================================================================")
                    for f in os.listdir(self.best_loss_checkpoint_dir):
                        os.remove(os.path.join(self.best_loss_checkpoint_dir, f))

                    self.model.save_weights(
                        os.path.join(self.best_loss_checkpoint_dir,
                                     self.model_name + "-epoch-{}.weights.h5".format(epoch)))
                    self.best_loss = float(num_val_metrics[1])
                    self.bk_metric["best_loss"] = self.best_loss
                    bk_metric_file = open('{}/{}_bk_metric.txt'.format(self.log_dir, self.model_name),
                                          'w')
                    json.dump(self.bk_metric, bk_metric_file)
                    bk_metric_file.close()
                    self.last_weights = self.model.get_weights()

                if not math.isnan(squared_error_metrics_eval) and \
                        (squared_error_metrics_eval < self.best_squared_error_metrics or
                         self.best_squared_error_metrics < 0):
                    self.log_train.write_mylines(
                        "======================================================================\n")
                    self.log_train.write_mylines(
                        "Found better checkpoint! Saving to {}\n".format(
                            self.best_squared_error_checkpoint_dir))
                    self.log_train.write_mylines(
                        "======================================================================\n")
                    print("===========================================================================")
                    print("Found best new metric checkpoint! Saving to {}".format(
                        self.best_squared_error_checkpoint_dir))
                    print("===========================================================================")
                    for f in os.listdir(self.best_squared_error_checkpoint_dir):
                        os.remove(os.path.join(self.best_squared_error_checkpoint_dir, f))

                    self.model.save_weights(
                        os.path.join(self.best_squared_error_checkpoint_dir,
                                     self.model_name + "-epoch-{}.weights.h5".format(epoch)))
                    self.best_squared_error_metrics = squared_error_metrics_eval
                    self.bk_metric["best_squared_error_metrics"] = self.best_squared_error_metrics
                    bk_metric_file = open('{}/{}_bk_metric.txt'.format(self.log_dir, self.model_name),
                                          'w')
                    json.dump(self.bk_metric, bk_metric_file)
                    bk_metric_file.close()
                    self.wait = 0
                    # Record the best weights if current results is better (less).
                    self.best_weights = self.model.get_weights()
                elif not math.isnan(
                        squared_error_metrics_eval) and self.best_squared_error_metrics >= 0:
                    self.wait += 1
                    if self.wait > self.patience:
                        self.epoch_early_stopping = epoch
                        self.model.stop_training = True
                        self.log_train.write_mylines(
                            "======================================================================\n")
                        self.log_train.write_mylines("Restoring model weights from best new metric\n")
                        self.log_train.write_mylines(
                            "======================================================================\n")
                        self.model.set_weights(self.best_weights)
                else:
                    self.wait += 1
                    if self.wait > self.patience:
                        self.epoch_early_stopping = epoch
                        self.model.stop_training = True
                        self.log_train.write_mylines(
                            "======================================================================\n")
                        self.log_train.write_mylines("Restoring model weights from loss better\n")
                        self.log_train.write_mylines(
                            "======================================================================\n")
                        self.model.set_weights(self.last_weights)

                if not math.isnan(f1_score_metrics_eval) and (
                        f1_score_metrics_eval > self.best_f1_score_metrics
                        or self.best_f1_score_metrics < 0):
                    self.log_train.write_mylines(
                        "======================================================================\n")
                    self.log_train.write_mylines(
                        "Found better checkpoint! Saving to {}\n".format(self.best_f1_checkpoint_dir))
                    self.log_train.write_mylines(
                        "======================================================================\n")
                    print("===========================================================================")
                    print("Found best new f1 score checkpoint! Saving to {}".format(
                        self.best_f1_checkpoint_dir))
                    print("===========================================================================")
                    for f in os.listdir(self.best_f1_checkpoint_dir):
                        os.remove(os.path.join(self.best_f1_checkpoint_dir, f))

                    self.model.save_weights(
                        os.path.join(self.best_f1_checkpoint_dir,
                                     self.model_name + "-epoch-{}.weights.h5".format(epoch)))
                    self.best_f1_score_metrics = f1_score_metrics_eval
                    self.bk_metric["best_f1_score_metrics"] = self.best_f1_score_metrics
                    bk_metric_file = open('{}/{}_bk_metric.txt'.format(self.log_dir, self.model_name),
                                          'w')
                    json.dump(self.bk_metric, bk_metric_file)
                    bk_metric_file.close()

                sys.stdout.flush()
                # endregion Save model

            with open(self.log_dir + '/{}_log.csv'.format(self.model_name), mode='a+') as report_file:
                report_writer = csv.DictWriter(report_file, fieldnames=self.fieldnames)
                report_writer.writerow(report_row)

        def on_train_end(self, logs=None):
            if self.epoch_early_stopping > 0:
                self.log_train.write_mylines(
                    "======================================================================\n")
                self.log_train.write_mylines(
                    "Early stopping! model weights from the end of the best squared_error_metrics_eval\n")
                self.log_train.write_mylines(
                    "======================================================================\n")
                print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

    class CustomRecall(keras.metrics.Recall):
        def __init__(self,
                     class_id=None,
                     name=None):
            super(CustomRecall, self).__init__(class_id=class_id, name=name)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            return super(CustomRecall, self).update_state(y_true, y_pred, sample_weight)

        def result(self):
            return super(CustomRecall, self).result()

        def reset_states(self):
            super(CustomRecall, self).reset_states()

        def get_config(self):
            return super(CustomRecall, self).get_config()

    class CustomPrecision(keras.metrics.Precision):
        def __init__(self,
                     class_id=None,
                     name=None):
            super(CustomPrecision, self).__init__(class_id=class_id, name=name)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            return super(CustomPrecision, self).update_state(y_true, y_pred, sample_weight)

        def result(self):
            return super(CustomPrecision, self).result()

        def reset_states(self):
            super(CustomPrecision, self).reset_states()

        def get_config(self):
            return super(CustomPrecision, self).get_config()

    class CustomMeanMetricWrapper(keras.metrics.Mean):

        def __init__(self, fn, name=None, dtype=None, **kwargs):
            super(CustomMeanMetricWrapper, self).__init__(name=name, dtype=dtype)
            self._fn = fn
            self._fn_kwargs = kwargs

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            y_true = math_ops.cast(y_true, self._dtype)
            y_pred = math_ops.cast(y_pred, self._dtype)
            [y_true, y_pred], sample_weight = \
                metrics_utils.ragged_assert_compatible_and_get_flat_values(
                    [y_true, y_pred], sample_weight)
            y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

            ag_fn = autograph.tf_convert(self._fn, ag_ctx.control_status_ctx())
            matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
            return super(CustomMeanMetricWrapper, self).update_state(matches,
                                                                     sample_weight=sample_weight)

    class CustomCategoricalAccuracy(CustomMeanMetricWrapper):
        def __init__(self, name='categorical_accuracy', dtype=None):
            super(CustomCategoricalAccuracy, self).__init__(
                keras.metrics.categorical_accuracy, name, dtype=dtype)

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

    with tf.device("/cpu:0"):
        train_filenames = _get_tfrecord_filenames(train_directory, True)
        train_dataset = tf.data.TFRecordDataset(train_filenames)

        train_dataset = train_dataset.map(partial(_preprocess_proto,
                                                  feature_len=feature_len,
                                                  label_len=num_block,
                                                  class_num=len(beat_class.keys())),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_dataset = train_dataset.shuffle(buffer_size=8192)

        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(batch_size * 5)

        val_filenames = _get_tfrecord_filenames(eval_directory, False)
        val_dataset = tf.data.TFRecordDataset(val_filenames)

        val_dataset = val_dataset.map(partial(_preprocess_proto,
                                              feature_len=feature_len,
                                              label_len=num_block,
                                              class_num=len(beat_class.keys())),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(batch_size * 5)

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
        num_loop = int(_qrs_model_path[m])
        num_filters = np.asarray([int(i) for i in _qrs_model_path[m + 1].split('.')], dtype=int)
        try:
            from_logits = bool(int(_qrs_model_path[m + 2]))
        except:
            from_logits = False

        # print("Info: {}_{}_{}_{}_{}_{}".format(feature_len,
        #                                         len(beat_class),
        #                                         from_logits,
        #                                         num_filters,
        #                                         num_loop,
        #                                         float(_qrs_model_path[-1])))
        train_model = getattr(beat_model, func)(feature_len,
                                                len(beat_class),
                                                from_logits,
                                                num_filters,
                                                num_loop,
                                                float(_qrs_model_path[-1]))
        train_model.summary()
    else:
        return None

    # keras.utils.plot_model(train_model,
    #                           to_file=log_dir + '/ModelGraph.png',
    #                           show_shapes=True,
    #                           show_dtype=True)

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss = keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    if from_logits:
        metrics = [
            ConfusionMatrix(classes=len(beat_class), name='confusion_matrix'),
            CustomRecall(name='recall'),
            CustomPrecision(name='precision'),
            CustomCategoricalAccuracy(name='accuracy')
        ]
        beat = [c for _, c in enumerate(beat_class.keys())]
        for i in range(len(beat_class)):
            metrics.append(CustomRecall(class_id=i, name='{}_Se'.format(beat[i])))
            metrics.append(CustomPrecision(class_id=i, name='{}_P'.format(beat[i])))
    else:
        metrics = [
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            ConfusionMatrix(classes=len(beat_class), name='confusion_matrix'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision')
        ]
        beat = [c for _, c in enumerate(beat_class.keys())]
        for i in range(len(beat_class)):
            metrics.append(keras.metrics.Recall(class_id=i, name='{}_Se'.format(beat[i])))
            metrics.append(keras.metrics.Precision(class_id=i, name='{}_P'.format(beat[i])))

    train_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if resume_from is not None:
        begin_at_epoch = int(resume_from.split("-")[-1])
        log_train.write_mylines("Restoring checkpoint from {}\n".format(resume_from))
        log_train.write_mylines("Beginning at epoch {}\n".format(begin_at_epoch + 1))
        print("Restoring checkpoint from {}".format(resume_from))
        print("Beginning at epoch {}".format(begin_at_epoch + 1))
        train_model.load_weights(tf.train.latest_checkpoint(last_checkpoint_dir)).expect_partial()
    else:
        begin_at_epoch = 0
        print("===============================================================================")
        print("WARNING: --resume_from checkpoint flag is not set. Training model from scratch.")
        print("===============================================================================")

    if bk_metric is None:
        best_loss = -1
        best_squared_error_metrics = -1
        best_f1_score_metrics = -1
        bk_metric = dict()
        bk_metric["best_loss"] = -1
        bk_metric["best_squared_error_metrics"] = -1
        bk_metric["best_f1_score_metrics"] = -1
        bk_metric["stop_train"] = False
        bk_metric_file = open('{}/{}_bk_metric.txt'.format(log_dir, model_name), 'w')
        json.dump(bk_metric, bk_metric_file)
        bk_metric_file.close()
    else:
        best_loss = bk_metric["best_loss"]
        best_squared_error_metrics = bk_metric["best_squared_error_metrics"]
        best_f1_score_metrics = bk_metric["best_f1_score_metrics"]

    tensorboard_dir = model_dir + '/logs'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
        os.makedirs(tensorboard_dir + '/train')
        os.makedirs(tensorboard_dir + '/validation')

    log_callback = CustomCallback(
        model_name=model_name,
        log_train=log_train,
        stopped_epoch=begin_at_epoch + epoch_num,
        last_checkpoint_dir=last_checkpoint_dir,
        best_loss_checkpoint_dir=best_loss_checkpoint_dir,
        best_squared_error_checkpoint_dir=best_squared_error_checkpoint_dir,
        best_f1_checkpoint_dir=best_f1_checkpoint_dir,
        bk_metric=bk_metric,
        lbl_train=beat_class,
        lbl_val=val_class,
        log_dir=log_dir,
        field_name=fieldnames,
        best_loss=best_loss,
        best_squared_error_metrics=best_squared_error_metrics,
        best_f1_score_metrics=best_f1_score_metrics,
        length_train=_calc_num_steps(datastore_dict['train']['total_sample'], batch_size),
        length_valid=_calc_num_steps(datastore_dict['eval']['total_sample'], batch_size),
        valid_freq=valid_freq,
        patience=patience,
        tensorboard_dir=tensorboard_dir)

    with tf.device('/gpu:{}'.format(use_gpu_index if use_gpu_index >= 0 else 0)):
        train_model.fit(x=train_dataset,
                        epochs=begin_at_epoch + epoch_num,
                        verbose=0,
                        callbacks=[log_callback], # tf.compat.v1.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)],
                        validation_data=val_dataset,
                        validation_freq=[valid_freq * (x + 1) for x in
                                         range((begin_at_epoch + epoch_num) // valid_freq)],
                        initial_epoch=begin_at_epoch)

    bk_metric["stop_train"] = True
    bk_metric_file = open('{}/{}_bk_metric.txt'.format(log_dir, model_name), 'w')
    json.dump(bk_metric, bk_metric_file)
    bk_metric_file.close()

    log_train.write_mylines('\nEnd : {}\n'.format(str(datetime.datetime.now())))

    return False
