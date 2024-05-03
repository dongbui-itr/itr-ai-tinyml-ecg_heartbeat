import os
import subprocess


def generate_script(MODEL_TFLITE, MODEL_TFLITE_MICRO, REPLACE_TEXT, MODEL_NAME):
    """

    :param bashlink:
    :param dir:
    :param ext_out:
    :param ext_in:
    :return:
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(curr_dir + '/generate_microcontroller_model.sh' + ' ' +
                    MODEL_TFLITE + ' ' +  # $1
                    MODEL_TFLITE_MICRO + ' ' +  # $2
                    REPLACE_TEXT + ' ' +  # $3
                    MODEL_NAME,  # $4
                    shell=True)
