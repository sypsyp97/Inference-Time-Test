import timeit
import numpy as np


def test_inference_time(model):

    input_data = np.random.randn(1, 256, 256, 3).astype(np.float32)
    start_time = timeit.default_timer()
    _ = model.predict(input_data)
    end_time = timeit.default_timer()
    inference_time = (end_time - start_time) * 1000  # in milliseconds

    return inference_time
