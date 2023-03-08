import gc
import os
import pickle
import time
import psutil

import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter

from TFLite_Converter import tflite_converter
from Compile_Edge_TPU import compile_edgetpu
from raw_inference_time import test_inference_time
from reference import create_model, check_model


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


if __name__ == "__main__":
    gc.enable()
    inference_times = []
    tpu_inference_times = []
    gc_threshold = 24 * 1024 ** 3

    image_file = 'test.jpg'
    image = Image.open(image_file).convert('RGB')
    image = np.array(image)

    for i in range(500):
        try:
            model_array = np.random.randint(0, 2, (9, 18))
            model = create_model(model_array=model_array, num_classes=5, input_shape=(256, 256, 3))
            while check_model(model):
                model_array = np.random.randint(0, 2, (9, 18))
                model = create_model(model_array=model_array, num_classes=5, input_shape=(256, 256, 3))
            del model_array

            tflite_model_name = tflite_converter(model, i)
            edgetpu_model_name = compile_edgetpu(tflite_model_name)
            del tflite_model_name

            interpreter = make_interpreter(edgetpu_model_name)
            del edgetpu_model_name
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()[0]

            input_tensor = np.expand_dims(image, axis=0).astype(input_details['dtype'])
            interpreter.set_tensor(input_details['index'], input_tensor)

            del input_details
            del input_tensor

            start_time = time.monotonic()
            interpreter.invoke()
            tpu_inference_time = (time.monotonic() - start_time) * 1000
            tpu_inference_times.append(tpu_inference_time)
            del interpreter
            del start_time

            inference_time = test_inference_time(model)
            del model
            inference_times.append(inference_time)
            print(f"Model:{i}")
            print(inference_time, tpu_inference_time)

            del tpu_inference_time
            del inference_time

            if get_memory_usage() > gc_threshold:
                gc.collect()
                print("Memory cleared")

        except Exception as e:
            print(e)

        finally:
            print(inference_times)
            print(tpu_inference_times)
            with open('results/inference_times.pkl', 'wb') as f:
                pickle.dump(inference_times, f)
            with open('results/tpu_inference_times.pkl', 'wb') as f:
                pickle.dump(tpu_inference_times, f)
