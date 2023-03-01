from reference import create_first_population
from raw_inference_time import test_inference_time
from TFLite_Converter import tflite_converter
from Compile_Edge_TPU import compile_edgetpu

if __name__ == "__main__":
    _, model_list = create_first_population(population=100, num_classes=5)
    inference_times = test_inference_time(model_list)
    tflite_converter(model_list)
    compile_edgetpu()

