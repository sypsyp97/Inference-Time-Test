import matplotlib.pyplot as plt
import pickle


with open('inference_times.pkl', 'rb') as f:
    inference_times = pickle.load(f)

with open('tpu_inference_times.pkl', 'rb') as f:
    tpu_inference_times = pickle.load(f)

with plt.style.context('seaborn-whitegrid'):
    fig, ax = plt.subplots()
    ax.scatter(inference_times, tpu_inference_times, alpha=0.5)
    ax.set_title('Inference Times vs TPU Inference Times')
    ax.set_xlabel('Inference Time (seconds)')
    ax.set_ylabel('TPU Inference Time (seconds)')

    # Displaying the plot
    plt.show()