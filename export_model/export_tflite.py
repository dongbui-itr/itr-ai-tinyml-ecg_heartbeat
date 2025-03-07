from os.path import join, basename
from glob import glob
from tensorflow import lite
from functools import partial

import os.path
import tensorflow as tf
import datetime
import numpy as np
import model_2D as model_new



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

def _get_tfrecord_filenames(dir_path, is_training):
    if not os.path.exists(dir_path):
        raise FileNotFoundError("{}; No such file or directory.".format(dir_path))

    filenames = sorted(glob(os.path.join(dir_path, "*.tfrecord")))
    if not filenames:
        raise FileNotFoundError("No TFRecords found in {}".format(dir_path))

    return filenames


class Export_tflite():
    def __init__(self, model_name, model, eval_data, work_dir, batch_size=1, quantization=False):
        self.model_name = model_name
        self.model = model
        self.eval_data = eval_data
        self.work_dir = work_dir
        self.quantization = quantization
        self.batch_size = batch_size
        self.tfLite_name = '{}_bz{}_quanz{}.tflite'.format(self.model_name, self.batch_size, str(self.quantization))

    def convert_to_tf_lite(self, quantization=True, batch_size=1, c_export=False):
        # self.model.load_weights(tf.train.latest_checkpoint(join(self.work_dir, 'checkpoint'))).expect_partial()
        # model_convert = self.model
        # self.model = tf.keras.models.load_model(self.model)

        input_shape = self.model.inputs[0].shape.as_list()
        input_shape[0] = batch_size
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if not quantization:
            tf_lite_model = converter.convert()

        else:
            def representative_dataset_gen():
                train_data = self.eval_data
                for sample in train_data.as_numpy_iterator():
                    yield [sample[0]]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset_gen
            converter.experimental_new_converter = True
            # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32

            converter._experimental_calibrate_only = False
            tf_lite_model = converter.convert()

        with open(join(self.work_dir, self.tfLite_name), 'wb') as file:
            file.write(tf_lite_model)

        if c_export:
            from tensorflow.lite import python as py
            source_text, header_text = self.convert_bytes_to_c_source(tf_lite_model, self.model_name)
            with open(join(self.work_dir, '{}_bz{}_quanz{}.h'.format(self.model_name, self.batch_size, str(self.quantization))), 'w') as file:
                file.write(header_text)
            with open(join(self.work_dir, '{}_bz{}_quanz{}.cpp'.format(self.model_name, self.batch_size, str(self.quantization))), 'w') as file:
                file.write(source_text)

    def eval_tflite(self, eval_data):
        interpreter = tf.lite.Interpreter(model_path=join(self.work_dir, self.tfLite_name))
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_scale, input_zero_point = input_details["quantization"]
        len_input = input_details['shape'][2]
        labels = []
        predicts = np.asarray([])
        k = 0
        for data in eval_dataset.as_numpy_iterator():
            if k > 1000:
                break

            k += 1

            if input_details['dtype'] == np.int8:
                sample = data[0] / input_scale + input_zero_point
                sample = sample.astype(np.int8)
                label = data[0] / input_scale + input_zero_point
                label = label.astype(np.int8)
            else:
                sample = data[0]
                label = data[1]

            sample = np.expand_dims(sample, axis=0)
            labels.append(label)

            interpreter.resize_tensor_input(input_details['index'], sample.shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details['index'], sample)
            interpreter.invoke()
            output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            if len(predicts) == 0:
                predicts = output
            else:
                predicts = np.concatenate((predicts, output), axis=0)

        labels = np.squeeze(np.asarray(labels))
        predicts = np.squeeze(np.asarray(predicts))
        eval_mse = np.round(np.linalg.norm(predicts - labels, axis=-1) ** 2 / len_input, 4)
        histogram_eval = np.histogram(eval_mse, bins=50)

        from matplotlib import pyplot as plt
        indx = np.flatnonzero(histogram_eval[0] != 0)
        plt.bar((np.diff(histogram_eval[1]) + histogram_eval[1][:-1])[indx], histogram_eval[0][indx], color='r')
        plt.show()

    def convert_bytes_to_c_source(self, data,
                                  array_name,
                                  max_line_width=80,
                                  include_guard=None,
                                  include_path=None,
                                  use_tensorflow_license=False):
        """Returns strings representing a C constant array containing `data`.

        Args:
          data: Byte array that will be converted into a C constant.
          array_name: String to use as the variable name for the constant array.
          max_line_width: The longest line length, for formatting purposes.
          include_guard: Name to use for the include guard macro definition.
          include_path: Optional path to include in the source file.
          use_tensorflow_license: Whether to include the standard TensorFlow Apache2
            license in the generated files.

        Returns:
          Text that can be compiled as a C source file to link in the data as a
          literal array of values.
          Text that can be used as a C header file to reference the literal array.
        """

        starting_pad = "   "
        array_lines = []
        array_line = starting_pad
        for value in bytearray(data):
            if (len(array_line) + 4) > max_line_width:
                array_lines.append(array_line + "\n")
                array_line = starting_pad
            array_line += " 0x%02x," % (value,)
        if len(array_line) > len(starting_pad):
            array_lines.append(array_line + "\n")
        array_values = "".join(array_lines)

        if include_guard is None:
            include_guard = "TENSORFLOW_LITE_UTIL_" + array_name.upper() + "_DATA_H_"

        if include_path is not None:
            include_line = "#include \"{include_path}\"\n".format(
                include_path=include_path)
        else:
            include_line = ""

        if use_tensorflow_license:
            license_text = """
    /* Copyright {year} The TensorFlow Authors. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    ==============================================================================*/
    """.format(year=datetime.date.today().year)
        else:
            license_text = ""

        source_template = """{license_text}
    // This is a TensorFlow Lite model file that has been converted into a C data
    // array using the tensorflow.lite.util.convert_bytes_to_c_source() function.
    // This form is useful for compiling into a binary for devices that don't have a
    // file system.

    {include_line}
    // We need to keep the data array aligned on some architectures.
    #ifdef __has_attribute
    #define HAVE_ATTRIBUTE(x) __has_attribute(x)
    #else
    #define HAVE_ATTRIBUTE(x) 0
    #endif
    #if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
    #define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
    #else
    #define DATA_ALIGN_ATTRIBUTE
    #endif

    const unsigned char {array_name}[] DATA_ALIGN_ATTRIBUTE = {{
    {array_values}}};
    const int {array_name}_len = {array_length};
    """

        source_text = source_template.format(
            array_name=array_name,
            array_length=len(data),
            array_values=array_values,
            license_text=license_text,
            include_line=include_line)

        header_template = """
    {license_text}

    // This is a TensorFlow Lite model file that has been converted into a C data
    // array using the tensorflow.lite.util.convert_bytes_to_c_source() function.
    // This form is useful for compiling into a binary for devices that don't have a
    // file system.

    #ifndef {include_guard}
    #define {include_guard}

    extern const unsigned char {array_name}[];
    extern const int {array_name}_len;

    #endif  // {include_guard}
    """

        header_text = header_template.format(
            array_name=array_name,
            include_guard=include_guard,
            license_text=license_text)

        return source_text, header_text


if __name__ == '__main__':

    # eval_directory = '/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/abnormal_noise_5s_240906_1.0.1.1/eval/'
    # model_name = 'models_squeeze_unet_2d_2'
    # model = '/mnt/Dataset/ECG/PortalData_2/QRS_Classification_portal_data/abnormal_noise_5s_240906_1.0.1.1/240906_172511/models_squeeze_unet_2d_2/best_checkpoint/squeeze_unet_2d_2-002-0.05242.h5'
    # work_dir = os.path.dirname(model)

    eval_directory = '/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/241210/250_05_78_0_0_0_5_0_0.99_c1/eval'
    model_name = 'beat_concat_seq3_250Hz'
    ckt = '/mnt/MegaProject/Dong_data/QRS_Classification_portal_data/241205/250_05_78_0_0_0_5_0_0.99_c2/output/model/beat_concat_seq3_250Hz_2_8.16.8_0_0.5_2/best_squared_error_metric/beat_concat_seq3_250Hz_2_8.16.8_0_0.5-epoch-28.weights.h5'
    work_dir = os.path.dirname(ckt)

    feature_len = 1250
    class_num = 3
    from_logits = False
    num_filters = [8, 16, 8]
    num_loop = 2

    eval_filenames = _get_tfrecord_filenames(eval_directory, True)
    eval_dataset = tf.data.TFRecordDataset(eval_filenames)

    eval_dataset = eval_dataset.map(partial(_preprocess_proto,
                                            feature_len=feature_len,
                                            label_len=78,
                                            class_num=class_num),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

    beat_model = getattr(model_new, model_name)(feature_len,
                                          class_num,
                                          from_logits,
                                          num_filters,
                                          num_loop,
                                          0.5,
                                          False)
    check_point = glob(ckt)
    beat_model.load_weights(check_point)

    export_tflite = Export_tflite(model_name=model_name,
                                  model=beat_model,
                                  eval_data=eval_dataset,
                                  work_dir=work_dir,
                                  batch_size=1,
                                  quantization=False)

    export_tflite.convert_to_tf_lite(c_export=True)

    # export_tflite.eval_tflite(eval_data=eval_dataset)

