import os, torch, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # kill compiler warnings
from keras import backend

def main():
    if len(backend.tensorflow_backend._get_available_gpus()) > 0:
        print("\nNumber of Keras-enabled GPUs: ", len(backend.tensorflow_backend._get_available_gpus()))
    else: print("No Keras-enabled GPUs detected")
    if torch.cuda.device_count() > 0:
        print("Number of Torch-enabled GPUs: ", torch.cuda.device_count())
        print("\nGPU Name: ", torch.cuda.get_device_name(0))
    else: print("No Torch-enabled GPUs detected")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = round(end_time - start_time, 3)
    print("\nRuntime for program: " + str(duration) + " secs")