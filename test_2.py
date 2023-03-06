from reference import create_model
from reference import check_model
from raw_inference_time import test_inference_time
from TFLite_Converter import tflite_converter
from Compile_Edge_TPU import compile_edgetpu

import time
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from multiprocessing import Pool, cpu_count


def create_model_and_test_inference_time(i):
    try:
        model_array = np.random.randint(0, 2, (9, 18))
        model = create_model(model_array=model_array, num_classes=5, input_shape=(256, 256, 3))
        while check_model(model):
            model_array = np.random.randint(0, 2, (9, 18))
            model = create_model(model_array=model_array, num_classes=5, input_shape=(256, 256, 3))

        tflite_model_name = tflite_converter(model, i)
        edgetpu_model_name = compile_edgetpu(tflite_model_name)

        interpreter = make_interpreter(edgetpu_model_name)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]

        image_file = 'test.jpg'
        image = Image.open(image_file).convert('RGB')
        image = np.array(image.resize((input_details['shape'][1], input_details['shape'][2])))

        input_tensor = np.expand_dims(image, axis=0).astype(input_details['dtype'])
        interpreter.set_tensor(input_details['index'], input_tensor)

        start_time = time.monotonic()
        interpreter.invoke()
        tpu_inference_time = (time.monotonic() - start_time) * 1000
        inference_time = test_inference_time(model)
        print(f"Model:{i}")
        print(inference_time, tpu_inference_time)

        return inference_time, tpu_inference_time
    except Exception as e:
        print(e)


if __name__ == "__main__":
    num_models = 500
    inference_times = []
    tpu_inference_times = []

    with Pool(processes=min(cpu_count() - 1, 10)) as p:
        results = p.map(create_model_and_test_inference_time, range(num_models))

    for r in results:
        if r is not None:
            inference_times.append(r[0])
            tpu_inference_times.append(r[1])

    print("Inference times:", inference_times)
    print("TPU inference times:", tpu_inference_times)
