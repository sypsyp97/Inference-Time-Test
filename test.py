from reference import create_model
from reference import check_model
from raw_inference_time import test_inference_time
from TFLite_Converter import tflite_converter
from Compile_Edge_TPU import compile_edgetpu
import csv

import time
import os
import numpy as np
from PIL import Image

from pycoral.utils.edgetpu import make_interpreter


if __name__ == "__main__":
    inference_times = []
    tpu_inference_times = []

    for i in range(1):
        try:
            model_array = np.random.randint(0, 2, (9, 18))
            model = create_model(model_array=model_array, num_classes=5, input_shape=(256, 256, 3))
            while check_model(model):
                model_array = np.random.randint(0, 2, (9, 18))
                model = create_model(model_array=model_array, num_classes=5, input_shape=(256, 256, 3))

            print(f"Generation:{i}")
            inference_time = test_inference_time(model)
            inference_times.append(inference_time)
            tflite_model_name = tflite_converter(model, i)
            edgetpu_model_name = compile_edgetpu(tflite_model_name)

            interpreter = make_interpreter(edgetpu_model_name)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]

            image_file = 'test.jpg'
            image = Image.open(image_file).convert('RGB')
            image = np.array(image.resize((input_details['shape'][1], input_details['shape'][2])))

            input_tensor = np.expand_dims(image, axis=0).astype(input_details['dtype'])
            interpreter.set_tensor(input_details['index'], input_tensor)

            start_time = time.monotonic()
            interpreter.invoke()
            tpu_inference_time = (time.monotonic() - start_time) * 1000
            tpu_inference_times.append(tpu_inference_time)
        except Exception as e:
            print(e)







