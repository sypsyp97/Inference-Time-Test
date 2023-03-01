import timeit
import numpy as np


def test_inference_time(model_list):
    input_data = np.random.randn(1, 256, 256, 3).astype(np.float32)

    # Create a list to store the inference times for each model
    inference_times = []

    # Loop over the models and measure the inference time for each one
    for model in model_list:
        start_time = timeit.default_timer()
        _ = model.predict(input_data)
        end_time = timeit.default_timer()
        inference_time = (end_time - start_time) * 1000  # in milliseconds
        inference_times.append(inference_time)

    # Save the inference times to a CSV file
    with open('inference_times.csv', 'w') as f:
        for i, model in enumerate(model_list):
            model_name = f"model_{i}"
            f.write('%s,%f\n' % (model_name, inference_times[i]))
    return inference_times
