import os
import tensorflow as tf


class ExportModel(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)])
    def score(self, segment):  # 'segment' is input signature
        result = self.model(segment)
        return {"prediction": result}  # 'prediction' is output signature


def export_model(model, output_path, signatures='beats'):
    """

    Parameters
    ----------
    model: keras sequential or keras model
    output_path: path to output model

    Returns
    -------

    """
    print("Output Path:", output_path)
    os.makedirs(output_path, exist_ok=True)
    module = ExportModel(model)
    print("signatures:", module.score)
    tf.saved_model.save(module, output_path,
                        signatures={'beats': module.score})  # beats is signature and model name
    exit()
