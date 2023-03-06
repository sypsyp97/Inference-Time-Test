import os


def compile_edgetpu(tflite_model_name):
    if not os.path.exists(tflite_model_name):
        print(f"{tflite_model_name} not found")
        return

    if not tflite_model_name.endswith('.tflite'):
        print(f"{tflite_model_name} is not a TFLite model file")
        return

    edgetpu_model_name = tflite_model_name.replace('.tflite', '_edgetpu.tflite')
    os.system('edgetpu_compiler {}'.format(tflite_model_name))

    return edgetpu_model_name

