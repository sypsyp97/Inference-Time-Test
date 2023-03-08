from reference import create_model
from reference import check_model
from raw_inference_time import test_inference_time
from TFLite_Converter import tflite_converter
from Compile_Edge_TPU import compile_edgetpu
import pickle
import matplotlib.pyplot as plt

import time
import numpy as np
from PIL import Image

from pycoral.utils.edgetpu import make_interpreter
from multiprocessing import Pool, cpu_count


def inference(model_index):
    try:
        model_array = np.random.randint(0, 2, (9, 18))
        model = create_model(model_array=model_array, num_classes=5, input_shape=(256, 256, 3))
        while check_model(model):
            model_array = np.random.randint(0, 2, (9, 18))
            model = create_model(model_array=model_array, num_classes=5, input_shape=(256, 256, 3))

        tflite_model_name = tflite_converter(model, model_index)
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

        print(f"Model:{model_index}")
        print(inference_time, tpu_inference_time)

        return inference_time, tpu_inference_time
    except Exception as e:
        print(e)
        return None
    finally:
        del model
        del interpreter
        del input_tensor


if __name__ == "__main__":
    pool = Pool(processes=8)

    num_models = 24
    results = pool.map(inference, range(num_models))
    inference_times, tpu_inference_times = zip(*results)
    inference_times = [x for x in inference_times if x is not None]
    tpu_inference_times = [x for x in tpu_inference_times if x is not None]

    print(inference_times)
    print(tpu_inference_times)

    with open('results/inference_times.pkl', 'wb') as f:
        pickle.dump(inference_times, f)

    with open('results/tpu_inference_times.pkl', 'wb') as f:
        pickle.dump(tpu_inference_times, f)

    with open('results/inference_times.pkl', 'rb') as f:
        inference_times = pickle.load(f)

    with open('results/tpu_inference_times.pkl', 'rb') as f:
        tpu_inference_times = pickle.load(f)

    plt.plot(inference_times, tpu_inference_times)

    # Adding title and labels to the plot
    plt.title('Inference Times vs TPU Inference Times')
    plt.xlabel('Inference Time (seconds)')
    plt.ylabel('TPU Inference Time (seconds)')

    # Displaying the plot
    plt.show()
