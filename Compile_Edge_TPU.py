import os


def compile_edgetpu(tflite_model_name):
    if not os.path.exists(tflite_model_name):
        print(f"{tflite_model_name} not found")
        return

    output_folder = "models"
    edgetpu_model_name = tflite_model_name.replace('.tflite', '_edgetpu.tflite')

    os.system('edgetpu_compiler -sa {} -o {}'.format(tflite_model_name, output_folder))

    # os.system('edgetpu_compiler -sa {}'.format(tflite_model_name))

    return edgetpu_model_name
