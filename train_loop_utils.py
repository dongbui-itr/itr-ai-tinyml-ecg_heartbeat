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
import io
from functools import partial
from glob import glob
from random import shuffle
import numpy as np

# import model as beat_model
import model_2D as beat_model
from utils.logging import TextLogging
from all_config import CLASS_WEIGHTS, CLASS_WEIGHTS_RETRAIN
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
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

    tensorboard_log_dir = model_dir + '/tensorboard_log_dir'
    last_checkpoint_dir = model_dir + '/last'
    best_loss_checkpoint_dir = model_dir + '/best_loss'
    best_f1_checkpoint_dir = model_dir + '/best_avg'
    for i in [last_checkpoint_dir, tensorboard_log_dir,
              best_loss_checkpoint_dir, best_f1_checkpoint_dir]:
        if not os.path.exists(i):
            os.makedirs(i)

    best_f1_class_checkpoint_dir = dict()
    best_f1_class_value = dict()
    best_acc_class_checkpoint_dir = dict()
    best_acc_class_value = dict()
    for i in beat_class.keys():
        best_f1_class_checkpoint_dir[i] = f"{model_dir}/best_f1_{i}"
        best_f1_class_value[i] = 100
        best_acc_class_checkpoint_dir[i] = f"{model_dir}/best_accuracy_{i}"
        best_acc_class_value[i] = 0
        if not os.path.exists(f"{model_dir}/best_f1_{i}"):
            os.makedirs(f"{model_dir}/best_f1_{i}")
        if not os.path.exists(f"{model_dir}/best_accuracy_{i}"):
            os.makedirs(f"{model_dir}/best_accuracy_{i}")

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
        # return sample, label
        return tf.expand_dims(tf.expand_dims(sample, axis=0), axis=-1), tf.expand_dims(label, axis=0)

    class ConfusionMatrix(keras.callbacks.Callback):
        """
        Callback for CM, per-class F1 using a tf.data.Dataset validation set.

        Args:
            validation_dataset (tf.data.Dataset): The validation dataset.
                It should yield tuples of (features, labels).
                Labels MUST be integers (not one-hot encoded) for this callback.
            num_classes (int): The total number of classes. Must be provided.
            class_names (list): Optional list of class names for plotting/reporting.
            print_every (int): Print metrics every N epochs. Default is 1.
            plot_every (int): Plot CM every N epochs. Set to 0 to disable. Default is 1.
            log_dir (str): Optional TensorBoard log directory for CM plots.
            file_writer (tf.summary.SummaryWriter): Optional existing TB writer for CM.
            steps (int): Number of steps (batches) in the validation dataset.
                         If None, it will iterate until the dataset is exhausted,
                         which is recommended for tf.data.Dataset.
        """

        def __init__(self, validation_dataset, num_classes, class_names=None, best_f1_checkpoint_dir=None, best_loss_checkpoint_dir=None,
                     print_every=1, plot_every=1, log_dir=None, file_writer=None, model_name="",
                     steps=None):
            super().__init__()
            if not isinstance(validation_dataset, tf.data.Dataset):
                raise TypeError("`validation_dataset` must be a tf.data.Dataset.")
            if not isinstance(num_classes, int) or num_classes <= 0:
                raise ValueError("`num_classes` must be a positive integer.")

            self.val_dataset = validation_dataset
            self.num_classes = num_classes
            self.class_names_provided = class_names
            self.print_every = max(1, int(print_every))
            self.plot_every = max(0, int(plot_every))
            self.log_dir = log_dir
            self.file_writer_cm = file_writer
            self.steps = steps  # Number of batches to iterate over
            self.f1_macro = -1
            self.f1_avg = -1
            self.model_name = model_name

            self._setup_class_names()
            self.best_f1_checkpoint_dir = best_loss_checkpoint_dir
            self.best_f1_avg_checkpoint_dir = best_f1_checkpoint_dir
            self.best_f1_class_checkpoint_dir = best_f1_class_checkpoint_dir
            self.best_f1_class_value = best_f1_class_value
            self.best_acc_class_checkpoint_dir = best_acc_class_checkpoint_dir
            self.best_acc_class_value = best_acc_class_value

            if self.log_dir and self.plot_every > 0 and self.file_writer_cm is None:
                cm_log_path = os.path.join(self.log_dir, 'cm')
                print(f"Confusion matrix plots will be logged to TensorBoard: {cm_log_path}")
                # Ensure parent directory exists if log_dir is nested
                os.makedirs(os.path.dirname(cm_log_path), exist_ok=True)
                try:
                    self.file_writer_cm = tf.summary.create_file_writer(cm_log_path)
                    print("TensorBoard SummaryWriter created successfully.")
                except Exception as e:
                    print(f"Error creating SummaryWriter at {cm_log_path}: {e}")
                    self.file_writer_cm = None  # Disable logging if creation fails

        def _setup_class_names(self):
            """Sets up class names based on num_classes."""
            if self.class_names_provided is None:
                self.class_names = [str(i) for i in range(self.num_classes)]
            elif len(self.class_names_provided) != self.num_classes:
                print(f"\nWarning: Provided class_names length ({len(self.class_names_provided)}) "
                      f"doesn't match provided num_classes ({self.num_classes}). "
                      f"Using default names ['0', '1', ...].")
                self.class_names = [str(i) for i in range(self.num_classes)]
            else:
                self.class_names = self.class_names_provided

        def _plot_confusion_matrix(self, cm, epoch):
            """ Helper function to plot the confusion matrix."""
            figure = plt.figure(figsize=(max(6, self.num_classes * 0.8), max(6, self.num_classes * 0.8)))
            sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, square=True,
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
            return figure

        def _log_cm_to_tensorboard(self, figure, epoch):
            """ Writes the plot image to TensorBoard summary."""
            if not self.file_writer_cm:
                print("Skipping TensorBoard CM plot logging: file_writer not available.")
                plt.close(figure)  # Still close the figure
                return
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(figure)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            try:
                # Ensure writer is correctly initialized
                if self.file_writer_cm._is_init:
                    with self.file_writer_cm.as_default(step=epoch):
                        tf.summary.image("Confusion Matrix", image)
                    self.file_writer_cm.flush()
                    # print(f"CM plot logged for epoch {epoch + 1}") # Optional verbose log
                else:
                    print("Skipping TensorBoard CM plot logging: file_writer not initialized.")

            except Exception as e:
                print(f"\nError logging confusion matrix to TensorBoard: {e}")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            all_y_true = []
            all_y_pred = []
            print(f"\nEpoch {epoch + 1}: Calculating validation metrics...", end='')

            # Iterate over the validation dataset
            try:
                # Use take(steps) if steps is defined, otherwise iterate until exhausted
                dataset_to_iterate = self.val_dataset.take(self.steps) if self.steps else self.val_dataset
                for batch_num, (x_batch, y_batch) in enumerate(dataset_to_iterate):
                    # Predict
                    y_pred_raw_batch = self.model.predict_on_batch(x_batch)  # Use predict_on_batch for efficiency

                    # Process predictions
                    if y_pred_raw_batch.ndim == 1 or y_pred_raw_batch.shape[-1] == 1:
                        y_pred_batch = (y_pred_raw_batch > 0.5).astype(int).flatten()
                        if self.num_classes < 2: self.num_classes = 2  # Ensure num_classes is 2 for binary
                    elif y_pred_raw_batch.ndim > 1 and y_pred_raw_batch.shape[-1] > 1:
                        y_pred_batch = np.argmax(y_pred_raw_batch, axis=-1)
                    else:
                        print(
                            f"\nWarning: Unusual model output shape {y_pred_raw_batch.shape} in batch {batch_num}. Skipping batch.")
                        continue  # Skip this batch if output is weird

                    # Process true labels (assuming integer labels in the dataset)
                    # Convert eager tensor to numpy
                    # y_true_batch = y_batch.numpy().flatten().astype(int)
                    y_true_batch = np.argmax(y_batch, axis=-1).flatten().astype(int)

                    all_y_true.append(y_true_batch)
                    all_y_pred.append(y_pred_batch)

                # Concatenate results from all batches
                if not all_y_true:
                    print(" No validation batches processed. Skipping metrics calculation.")
                    logs['val_f1_macro'] = 0.0  # Log default for checkpointing
                    return

                y_true = np.concatenate(all_y_true)
                y_pred = np.concatenate(all_y_pred).flatten()
                print(" Done.")  # Finish the "Calculating..." message

            except Exception as e:
                print(f" Error during validation data iteration or prediction: {e}")
                logs['val_f1_macro'] = 0.0  # Log default for checkpointing
                return  # Exit if iteration fails

            # --- Calculate Metrics ---
            labels_range = list(range(self.num_classes))
            try:
                cm = confusion_matrix(y_true, y_pred, labels=labels_range)
                accuracy_score_class = precision_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
                f1_scores_per_class = f1_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
                f1_macro = f1_score(y_true, y_pred, labels=labels_range, average='macro', zero_division=0)
                f1_weighted = f1_score(y_true, y_pred, labels=labels_range, average='weighted', zero_division=0)
            except ValueError as e:
                print(f"\nError calculating metrics (label mismatch?): {e}")
                cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
                accuracy_score_class = np.zeros(self.num_classes)
                f1_scores_per_class = np.zeros(self.num_classes)
                f1_macro = -1.0
                f1_weighted = -1.0
            except Exception as e:
                print(f"\nUnexpected error calculating metrics: {e}")
                cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
                accuracy_score_class = np.zeros(self.num_classes)
                f1_scores_per_class = np.zeros(self.num_classes)
                f1_macro = -1.0
                f1_weighted = -1.0

            # --- Print Metrics ---
            if (epoch + 1) % self.print_every == 0:
                print(f"\n----- Epoch {epoch + 1} Validation Metrics -----")
                print("Confusion Matrix:")
                print(cm)


                log_file = open(self.log_dir + "/train_log.txt", "a+")
                log_file.writelines(f"\n----- Epoch {epoch + 1} Validation Metrics -----\n")
                log_file.writelines("Confusion Matrix:\n")
                log_file.writelines(f"{cm}")
                log_file.writelines("\nPer-Class F1 Scores:\n")

                if len(f1_scores_per_class) == len(self.class_names):
                    print("\nPer-Class F1 Scores:")
                    for i, score in enumerate(f1_scores_per_class):
                        print(f"  - Class '{self.class_names[i]}' ({i}): {score:.4f}")
                        if self.best_f1_class_value[self.class_names[i]] > score:
                            print(f"\nMacro Avg F1-Score of {self.class_names[i]}:    {score:.4f}")
                            self.best_f1_class_value[self.class_names[i]] = score
                            ckt_name = os.path.join(self.best_f1_class_checkpoint_dir[self.class_names[i]], self.model_name + "-epoch-{}.weights.h5".format(epoch))
                            for f in os.listdir(self.best_f1_class_checkpoint_dir[self.class_names[i]]):
                                os.remove(os.path.join(self.best_f1_class_checkpoint_dir[self.class_names[i]], f))
                            log_file.writelines(f"==================================================\n")
                            log_file.writelines(f"\nMacro Avg F1-Score of {self.class_names[i]}:    {score:.4f}")
                            log_file.writelines(f"==================================================\n")
                            self.model.save_weights(ckt_name)
                    print("\nPer-Class Precision Scores:")
                    for i, score in enumerate(accuracy_score_class):
                        print(f"  - Class '{self.class_names[i]}' ({i}): {score:.4f}")
                        if i > 0 and self.best_acc_class_value[self.class_names[i]] < score:
                            print(f"\nMacro Avg Precision-Score of {self.class_names[i]}:    {score:.4f}")
                            self.best_acc_class_value[self.class_names[i]] = score
                            ckt_name = os.path.join(self.best_acc_class_checkpoint_dir[self.class_names[i]], self.model_name + "-epoch-{}.weights.h5".format(epoch))
                            for f in os.listdir(self.best_acc_class_checkpoint_dir[self.class_names[i]]):
                                os.remove(os.path.join(self.best_acc_class_checkpoint_dir[self.class_names[i]], f))
                            log_file.writelines(f"==================================================\n")
                            log_file.writelines(f"\nMacro Avg Precision-Score of {self.class_names[i]}:    {score:.4f}")
                            log_file.writelines(f"==================================================\n")
                            self.model.save_weights(ckt_name)

                else:
                    print(f"  F1 Scores raw: {f1_scores_per_class}")

                print(f"\nMacro Avg F1-Score:    {f1_macro:.4f}")
                print(f"Weighted Avg F1-Score: {f1_weighted:.4f}")
                print("------------------------------------")
                if (self.f1_macro > f1_macro and f1_macro != -1) or self.f1_macro == -1:
                    self.f1_macro = f1_macro
                    ckt_name = os.path.join(self.best_f1_checkpoint_dir,self.model_name + "-epoch-{}.weights.h5".format(epoch))
                    for f in os.listdir(self.best_f1_checkpoint_dir):
                        os.remove(os.path.join(self.best_f1_checkpoint_dir, f))

                    log_file.writelines(f"==================================================\n")
                    log_file.writelines(f"\nMacro F1-Score:    {f1_macro:.4f}\n")
                    log_file.writelines(f"==================================================\n")
                    self.model.save_weights(ckt_name)

                if (self.f1_avg > f1_weighted and f1_weighted != -1) or self.f1_avg == -1:
                    self.f1_avg = f1_weighted
                    ckt_name = os.path.join(self.best_f1_avg_checkpoint_dir,self.model_name + "-epoch-{}.weights.h5".format(epoch))
                    for f in os.listdir(self.best_f1_avg_checkpoint_dir):
                        os.remove(os.path.join(self.best_f1_avg_checkpoint_dir, f))

                    log_file.writelines(f"==================================================\n")
                    log_file.writelines(f"\nMacro Avg F1-Score:    {f1_weighted:.4f}\n")
                    log_file.writelines(f"==================================================\n")
                    self.model.save_weights(ckt_name)


                if len(f1_scores_per_class) == len(self.class_names):
                    for i, score in enumerate(f1_scores_per_class):
                        log_file.writelines(f"  - Class '{self.class_names[i]}' ({i}): {score:.4f}\n")
                else:
                    log_file.writelines(f"  F1 Scores raw: {f1_scores_per_class}\n")

                log_file.writelines(f"\nMacro Avg F1-Score:    {f1_macro:.4f}\n")
                log_file.writelines(f"Weighted Avg F1-Score: {f1_weighted:.4f}\n")
                log_file.writelines("------------------------------------\n")
                log_file.close()


            # --- Plot Confusion Matrix ---
            if self.plot_every > 0 and (epoch + 1) % self.plot_every == 0:
                try:
                    figure = self._plot_confusion_matrix(cm, epoch)
                    if self.file_writer_cm:
                        self._log_cm_to_tensorboard(figure, epoch)
                    else:
                        plt.show()
                        plt.close(figure)
                except Exception as e:
                    print(f"\nError plotting/logging confusion matrix: {e}")

            # --- Log Metrics to Keras logs dictionary ---
            logs['val_f1_macro'] = f1_macro
            logs['val_f1_weighted'] = f1_weighted
            if len(f1_scores_per_class) == len(self.class_names):
                for i, score in enumerate(f1_scores_per_class):
                    log_key = f'val_f1_{self.class_names[i]}'
                    log_key = ''.join(c if c.isalnum() else '_' for c in log_key)
                    logs[log_key] = score
            else:
                for i, score in enumerate(f1_scores_per_class):
                    logs[f'val_f1_class_{i}'] = score

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                logs['val_tp'] = tp
                logs['val_fp'] = fp
                logs['val_fn'] = fn
                logs['val_tn'] = tn

    train_filenames = _get_tfrecord_filenames(train_directory, True)
    train_dataset = tf.data.TFRecordDataset(train_filenames)

    train_dataset = train_dataset.map(partial(_preprocess_proto,
                                              feature_len=feature_len,
                                              label_len=num_block,
                                              class_num=len(beat_class.keys())),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # train_dataset = train_dataset.shuffle(buffer_size=8192)

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    # train_dataset = train_dataset.prefetch(batch_size * 5)

    val_filenames = _get_tfrecord_filenames(eval_directory, False)
    val_dataset = tf.data.TFRecordDataset(val_filenames)

    val_dataset = val_dataset.map(partial(_preprocess_proto,
                                          feature_len=feature_len,
                                          label_len=num_block,
                                          class_num=len(beat_class.keys())),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    # val_dataset = val_dataset.prefetch(batch_size * 5)

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

        train_model = getattr(beat_model, func)(feature_len,
                                                len(beat_class),
                                                from_logits,
                                                num_filters,
                                                num_loop,
                                                float(_qrs_model_path[-1]))
        # train_model.summary()
    else:
        return None

    # keras.utils.plot_model(train_model,
    #                           to_file=log_dir + '/ModelGraph.png',
    #                           show_shapes=True,
    #                           show_dtype=True)

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    # loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    loss = keras.losses.BinaryCrossentropy(from_logits=from_logits)

    # --- Instantiate Callbacks ---

    # a) Our custom callback for CM and F1 scores
    # Pass integer labels y_val_int here
    cm_f1_callback = ConfusionMatrix(
        validation_dataset=val_dataset,
        num_classes=len(list(beat_class.keys())),
        class_names=list(beat_class.keys()),
        print_every=1,
        plot_every=0,  # Disable direct plotting if using TensorBoard heavily
        log_dir=tensorboard_log_dir,  # Specify log dir for CM plots in TensorBoard
        best_loss_checkpoint_dir=best_loss_checkpoint_dir,
        best_f1_checkpoint_dir=best_f1_checkpoint_dir,
        model_name=model_name
    )

    # b) ModelCheckpoint to save the best model based on val_f1_macro
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_name + "/best_model_f1_macro_tf.weights.h5",
        monitor='val_f1_macro',  # Monitor the macro F1 score calculated by our callback
        mode='max',  # We want to maximize F1 score
        save_best_only=True,  # Only save when the monitored quantity improves
        save_weights_only=True,  # Set to False to save the entire model (`.keras` format recommended)
        verbose=1  # Print messages when saving
    )

    # c) Standard TensorBoard callback for logging scalars (loss, acc, F1 scores)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,  # Optional: log histograms
        write_graph=True  # Optional: log model graph
    )

    train_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    with tf.device('/gpu:{}'.format(use_gpu_index if use_gpu_index >= 0 else 0)):
        print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
        train_model.fit(x=train_dataset,
                        epochs=epoch_num,
                        verbose=1,
                        steps_per_epoch = datastore_dict["train"]["total_sample"]//batch_size,
                        callbacks=[cm_f1_callback,             # Calculates metrics, logs them to `logs` dict
                                   model_checkpoint_callback,  # Reads 'val_f1_macro' from `logs` and saves model
                                   tensorboard_callback   ],  # tf.compat.v1.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)],
                        validation_data=val_dataset,
                        # class_weight=CLASS_WEIGHTS,
                        validation_steps=1,
                        )

    # bk_metric["stop_train"] = True
    # bk_metric_file = open('{}/{}_bk_metric.txt'.format(log_dir, model_name), 'w')
    # json.dump(bk_metric, bk_metric_file)
    # bk_metric_file.close()
    train_model.save_weights(os.path.join(last_checkpoint_dir, model_name + ".weights.h5"))
    log_train.write_mylines('\nEnd : {}\n'.format(str(datetime.datetime.now())))

    return False


def retrain_freeze_beat_classification(use_gpu_index,
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
                                       epoch_num,
                                       model_retrain_dir):
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
                if bk_metric["stop_train"] and not resume_from:
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
    # from tensorflow.python.autograph.core import ag_ctx
    # from tensorflow.python.autograph.impl import api as autograph
    # from tensorflow.python.keras.utils import losses_utils
    # from tensorflow.python.keras.utils import metrics_utils
    # from tensorflow.python.ops import math_ops
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

    # class CustomMeanMetricWrapper(keras.metrics.Mean):
    #
    #     def __init__(self, fn, name=None, dtype=None, **kwargs):
    #         super(CustomMeanMetricWrapper, self).__init__(name=name, dtype=dtype)
    #         self._fn = fn
    #         self._fn_kwargs = kwargs
    #
    #     def update_state(self, y_true, y_pred, sample_weight=None):
    #         y_pred = tf.nn.softmax(y_pred, axis=-1)
    #         y_true = math_ops.cast(y_true, self._dtype)
    #         y_pred = math_ops.cast(y_pred, self._dtype)
    #         [y_true, y_pred], sample_weight = \
    #             metrics_utils.ragged_assert_compatible_and_get_flat_values(
    #                 [y_true, y_pred], sample_weight)
    #         y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
    #
    #         ag_fn = autograph.tf_convert(self._fn, ag_ctx.control_status_ctx())
    #         matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
    #         return super(CustomMeanMetricWrapper, self).update_state(matches,
    #                                                                  sample_weight=sample_weight)

    # class CustomCategoricalAccuracy(CustomMeanMetricWrapper):
    #     def __init__(self, name='categorical_accuracy', dtype=None):
    #         super(CustomCategoricalAccuracy, self).__init__(
    #             keras.metrics.categorical_accuracy, name, dtype=dtype)

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

        train_model = getattr(beat_model, func)(feature_len,
                                                5, #len(beat_class),
                                                from_logits,
                                                num_filters,
                                                num_loop,
                                                float(_qrs_model_path[-1]))
        # train_model.summary()
    else:
        return None

    # keras.utils.plot_model(train_model,
    #                           to_file=log_dir + '/ModelGraph.png',
    #                           show_shapes=True,
    #                           show_dtype=True)

    if resume_from:
        # begin_at_epoch = int(resume_from.split("-")[-1])
        # log_train.write_mylines("Restoring checkpoint from {}\n".format(resume_from))
        # log_train.write_mylines("Beginning at epoch {}\n".format(begin_at_epoch + 1))
        # print("Restoring checkpoint from {}".format(resume_from))
        # print("Beginning at epoch {}".format(begin_at_epoch + 1))
        begin_at_epoch = 0
        checkpoint = glob(model_retrain_dir + '/*.h5')[0]
        train_model.load_weights(checkpoint)
        
        train_model = beat_model.freeze_model(train_model, num_class=len(beat_class.keys()))
        train_model.summary()

    else:
        begin_at_epoch = 0
        print("===============================================================================")
        print("WARNING: --resume_from checkpoint flag is not set. Training model from scratch.")
        print("===============================================================================")

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss = keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    if from_logits:
        metrics = [
            ConfusionMatrix(classes=len(beat_class), name='confusion_matrix'),
            CustomRecall(name='recall'),
            CustomPrecision(name='precision'),
            # CustomCategoricalAccuracy(name='accuracy')
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
        # for i_epoch_num in range(epoch_num):
        train_model.fit(x=train_dataset,
                        # epochs=begin_at_epoch + i_epoch_num + 1,
                        epochs=begin_at_epoch + epoch_num,
                        verbose=0,
                        # callbacks=[log_callback],  # tf.compat.v1.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)],
                        validation_data=val_dataset,
                        # validation_freq=[valid_freq * (x + 1) for x in
                        #                  range((begin_at_epoch + epoch_num) // valid_freq)],
                        class_weight=CLASS_WEIGHTS_RETRAIN,
                        initial_epoch=begin_at_epoch)
        # begin_at_epoch = begin_at_epoch + i_epoch_num
        #
        # if i_epoch_num % 4 == 0:
        #     from run_ec57_utils import run_ec57
        #     from all_config import DB_TESTING as test_ec57_dir
        #     from multiprocessing import Pool
        #     from os.path import basename, dirname
        #     from utils.ec57_test import ec57_eval, del_result, bxb_eval, del_result2, ec57_eval_event
        #     from run_ec57_multiprocess_utils import process_beat_classification
        #
        #     import time
        #
        #     ext_ai = 'tmpatr'
        #     for db in test_ec57_dir:
        #         path2db = PATH_DATA_EC57 + db[0]
        #         # file_names = glob(path2db + '/*.dat')
        #         file_names = glob(path2db + f'/{FILE_NAME}.dat')
        #         # Get rid of the extension
        #         file_names = [p[:-4] for p in file_names
        #                       if basename(p)[:-4] not in ['104', '102', '107', '217', 'bw', 'em', 'ma']
        #                       if '_200hz' not in basename(p)[:-4]]
        #
        #         file_names = sorted(file_names)
        #
        #         num_file_each_process = int(len(file_names) / num_of_process)
        #         while num_file_each_process == 0:
        #             num_of_process -= 1
        #             num_file_each_process = int(len(file_names) / num_of_process)
        #
        #         file_process_split = [file_names[x:x + num_file_each_process] for x in range(0, len(file_names),
        #                                                                                      num_file_each_process)]
        #         arg_list = list()
        #         for i, file_list in enumerate(file_process_split):
        #             arg = (i,
        #                    use_gpu_index,
        #                    file_list,
        #                    model_name,
        #                    '', #checkpoint_dir,
        #                    datastore_dict,
        #                    ext_ai,
        #                    True,
        #                    False,
        #                    0,
        #                    0, #overlap,
        #                    1024,
        #                    None, #dir_image,
        #                    True )
        #
        #             arg_list.append(arg)
        #
        #         num_of_process = 4
        #         with Pool(processes=num_of_process) as pool:
        #             # print same numbers in arbitrary order
        #             for log_lines in pool.starmap(process_beat_classification, arg_list):
        #                 continue

    bk_metric["stop_train"] = True
    bk_metric_file = open('{}/{}_bk_metric.txt'.format(log_dir, model_name), 'w')
    json.dump(bk_metric, bk_metric_file)
    bk_metric_file.close()

    log_train.write_mylines('\nEnd : {}\n'.format(str(datetime.datetime.now())))

    return False
