import tensorflow as tf
import numpy as np


def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1, 256, 256, 1)
        yield [data.astype(np.float32)]


def tflite_converter(model_list):
    for i, model in enumerate(model_list):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # This enables quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # This sets the representative dataset for quantization
        converter.representative_dataset = representative_data_gen

        converter.target_spec.supported_types = [tf.int8]

        # These set the input and output tensors to uint8 (added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True

        tflite_model = converter.convert()
        tflite_model_name = f"model_{i}.tflite"

        with open(tflite_model_name, 'wb') as f:
            f.write(tflite_model)
