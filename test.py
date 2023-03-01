from reference import create_first_population
from raw_inference_time import test_inference_time
from TFLite_Converter import tflite_converter
from Compile_Edge_TPU import compile_edgetpu
import argparse
import time
import os
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

if __name__ == "__main__":
    _, model_list = create_first_population(population=100, num_classes=5)
    inference_times = test_inference_time(model_list)
    tflite_converter(model_list)
    compile_edgetpu()

    tpu_files = []
    for file in os.listdir('.'):
        if file.endswith('.tflite') and 'edgetpu' in file:
            tpu_files.append(file)

    tpu_inference_time = []

    for i in range(len(tpu_files)):
        model_file = f"model_{i}_edgetpu.tflite"
        interpreter = make_interpreter(model_file)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        # Load the input image
        image_file = 'test.jpg'
        image = Image.open(image_file).convert('RGB')
        image = np.array(image.resize((input_details['shape'][1], input_details['shape'][2])))

        # Set the input tensor
        input_tensor = np.expand_dims(image, axis=0).astype(input_details['dtype'])
        interpreter.set_tensor(input_details['index'], input_tensor)

        # Run inference and measure the time taken
        start_time = time.monotonic()
        interpreter.invoke()
        inference_time = time.monotonic() - start_time
        tpu_inference_time.append(inference_time * 1000)

        print('Inference time: {:.2f} ms'.format(inference_time * 1000))



