import os
import fnmatch


def compile_edgetpu():
    tflite_files = []
    for file in os.listdir('.'):
        if fnmatch.fnmatch(file, '*.tflite'):
            tflite_files.append(file)

    for model_file in tflite_files:
        os.system('edgetpu_compiler {}'.format(model_file))
