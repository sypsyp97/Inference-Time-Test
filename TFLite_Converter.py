import tensorflow as tf
import numpy as np


def representative_dataset_generator():
    for i in range(100):
        yield [np.array(np.random.uniform(0.0, 1.0, (1, 256, 256, 3)), dtype=np.float32)]


def tflite_converter(model, i):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset_generator

    converter.target_spec.supported_types = [tf.int8]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    tflite_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model, gpu_compatibility=True)
    tflite_model_name = f"model_{i}.tflite"

    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)

    return tflite_model_name
