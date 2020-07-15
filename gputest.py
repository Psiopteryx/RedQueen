import os, torch, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # kill compiler warnings
import tensorflow as tf


def main():
    print("\nNumber of Tensorflow GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
 #   else: print("No Keras-enabled GPUs detected")
    if torch.cuda.device_count() > 0:
        print("\nNumber of Torch-enabled GPUs: ", torch.cuda.device_count())
        print("\nGPU Name: ", torch.cuda.get_device_name(0))
    else: print("No Torch-enabled GPUs detected")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = round(end_time - start_time, 3)
    print("\nRuntime for program: " + str(duration) + " secs")
