import time, os, sys, humanfriendly, warnings, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from playsound import playsound
from PIL import Image, ImageFilter

source_directory = '/sources/source/'
results_directory = '/results/'

image_length = 128
image_width = 128
batch_size = 128

discriminator_learn_rate = 0.0001
generator_learn_rate = 0.0002

epochs = 25
leaky_ReLU = 0.2
label_noise = 0.2  # True/false noise for discriminator

epoch_image_batch_count = 2    # Cycle through entire batch of images x times + fixed_noise batch
final_image_batch_count = 5
zoom = 3  # Zoom factor, and edge enhance final output

random_noise_size = 200 # seed for generator

matplotlib.rcParams['animation.embed_limit'] = 2**128
Image.MAX_IMAGE_PIXELS = None

def load_dataset(source_directory, batch_size, image_shape):
    dataset = ImageDataGenerator()
    dataset = dataset.flow_from_directory(source_directory, target_size=(image_shape[0], image_shape[1]),
                                          batch_size=batch_size, class_mode=None)
    return dataset, dataset.samples

def build_discriminator(image_shape):
    discriminator = Sequential()
    discriminator.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding='same',
                data_format='channels_last',  kernel_initializer='glorot_uniform', input_shape=(image_shape)))
    discriminator.add(LeakyReLU(leaky_ReLU))

    discriminator.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                data_format='channels_last', kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(leaky_ReLU))

    discriminator.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                data_format='channels_last', kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(leaky_ReLU))

    discriminator.add(Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                data_format='channels_last', kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(leaky_ReLU))

    #new
    discriminator.add(Conv2D(filters=1024, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                data_format='channels_last', kernel_initializer='glorot_uniform'))
    discriminator.add(BatchNormalization(momentum=0.5))
    discriminator.add(LeakyReLU(leaky_ReLU))
    #end new, this was to go from 64 to 128

    discriminator.add(Flatten())
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))
    optimizer = Adam(discriminator_learn_rate, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=None)

    return discriminator

def build_generator():
    generator = Sequential()
    # changed to 1024 from 512, should increase input random shape as well
    generator.add(Dense(units=4 * 4 * 1024, kernel_initializer='glorot_uniform', input_shape=(1, 1, random_noise_size)))
    generator.add(Reshape(target_shape=(4, 4, 1024)))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    #new
    generator.add(Conv2DTranspose(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                  data_format='channels_last', kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=256, kernel_size=(5,5), strides=(2,2), padding='same',
                                  data_format='channels_last', kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(2,2), padding='same',
                                  data_format='channels_last', kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=64, kernel_size=(5,5), strides=(2,2), padding='same',
                                  data_format='channels_last', kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=3, kernel_size=(5,5), strides=(2,2), padding='same',
                                  data_format='channels_last', kernel_initializer='glorot_uniform'))
    generator.add(Activation('tanh'))

    optimizer = Adam(lr=generator_learn_rate, beta_1=0.5)
    generator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=None)

    return generator

def save_sample_images(generated_images, epoch):
    # save a summary gallery
    plt.figure(figsize=(8,8), num=2)
    gs1 = gridspec.GridSpec(4,4)
    gs1.update(wspace=0, hspace=0)

    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        #ax1.set_aspect('equal')
        image = generated_images[i, :, :, :]
        image +=1
        image *= 127.5
        fig = plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    if (epoch < epochs - 1):
        gallery_save_name = results_directory + 'Epoch ' + str(epoch + 1) + ' Gallery.png'
    else: gallery_save_name = results_directory + 'Final ' + ' Gallery.png'
    try: os.remove(gallery_save_name)
    except: pass
    plt.savefig(gallery_save_name, bbox_inches='tight', pad_inches=0)

def train_dcgan(batch_size, epochs, image_shape, source_directory):
    generator = build_generator()
    discriminator = build_discriminator(image_shape)
    gan = Sequential()
    discriminator.trainable = False  # Only for adversarial model
    gan.add(generator)
    gan.add(discriminator)
    optimizer = Adam(lr=generator_learn_rate, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=None)

    dataset, num_images = load_dataset(source_directory, batch_size, image_shape)
    num_batches = int(num_images / batch_size)
    start_time = time.time()
    total_images = np.empty((0, batch_size, image_shape[0], image_shape[1], 3))
    for batch_number in range(num_batches):
        real_images = dataset[batch_number]
        real_images /= 127.5  # normalise across 255 colours per RGB
        real_images -= 1
        try:
            total_images_temp = np.append(total_images, [real_images], axis=0)
            total_images = total_images_temp
            batch_elapsed_time = time.time() - start_time
            batch_load_time_ETA = (batch_elapsed_time * num_batches / batch_number) - batch_elapsed_time
            batch_load_time_ETA = humanfriendly.format_timespan(round(batch_load_time_ETA, 0))
            print("\rProcessing batch: " + str(batch_number) + " of " + str(num_batches) + " ETA: " + str(
                batch_load_time_ETA), end="")
        except:
            print("Error appending. ")

    adversarial_loss = np.empty(shape=1)
    discriminator_loss = np.empty(shape=1)
    batches = np.empty(shape=1)
    try: os.makedirs(results_directory)
    except: pass
    fixed_noise = np.random.normal(0, 1, size=(batch_size,) + (1, 1, random_noise_size))  # for a fixed batch over time
    batch_count, batch_number = 0, 0
    for epoch in range(epochs):
        for batch_number in range(num_batches):
            batch_start_time = time.time()
            real_images = total_images[batch_number]
            current_batch_size = real_images.shape[0]  # for final batch (smaller)

            noise = np.random.normal(0, 1, size=(current_batch_size,) + (1, 1, random_noise_size))
            generated_images = generator.predict(noise)

            # introduce noise to discriminator True/False label
            real_y = (np.ones(current_batch_size) - np.random.random_sample(current_batch_size) * label_noise)
            fake_y = np.random.random_sample(current_batch_size) * label_noise

            # train discriminator
            discriminator.trainable = True
            #print("Training Discriminator on Real")
            d_loss = discriminator.train_on_batch(real_images, real_y)
            #print("Training Discriminator on Fake")
            d_loss += discriminator.train_on_batch(generated_images, fake_y)
            discriminator_loss = np.append(discriminator_loss, d_loss)

            # train generator
            discriminator.trainable = False
            noise = np.random.normal(0, 1, size=(current_batch_size * 2,) + (1, 1, random_noise_size))

            # noise to discriminator True/False label again
            fake_y = (np.ones(current_batch_size * 2) - np.random.random_sample(current_batch_size *2) * label_noise)

            #print("Training Generator")
            g_loss = gan.train_on_batch(noise, fake_y)
            adversarial_loss = np.append(adversarial_loss, g_loss)
            batches = np.append(batches, batch_count)

            batch_time_elapsed = time.time() - batch_start_time
            batch_total_run_est_time = (batch_time_elapsed * num_batches * (epochs - epoch))\
                                       - (batch_time_elapsed * (batch_number / num_batches))
            batch_total_ETA = humanfriendly.format_timespan(round(batch_total_run_est_time, 0))

            print("\rEpoch: " + str(epoch + 1) + "/" + str(epochs) + "\tBatch: " + str(batch_number + 1)
                  + "/" + str(num_batches) + "\tGenerator loss: " + str(round(g_loss, 2)) + "\tDscriminator loss: "
                  + str(round(d_loss, 2)) + '\tETA: ' + batch_total_ETA, end="")

            batch_count += 1

        generated_images = generator.predict(fixed_noise)
        # Create a thumbnail gallery
        save_sample_images(generated_images, epoch)
        # save full galleries
        if(epoch < epochs - 1):
            image_count = 1
            for epoch_image_batch in range(epoch_image_batch_count):
                noise = np.random.normal(0, 1, size=(batch_size,) + (1, 1, random_noise_size))
                generated_novel_images = generator.predict(noise)
                save_epoch_directory = results_directory + 'Epoch ' + str(epoch + 1) + '/'
                try: os.makedirs(save_epoch_directory)
                except: pass
                for i in range(batch_size):
                    image = generated_novel_images[i, :, :, :]
                    image += 1
                    image *= 127.5
                    save_image = Image.fromarray(image.astype(np.uint8))
                    image_save_name = save_epoch_directory + ' Image ' + str(image_count) + '.png'
                    try: os.remove(image_save_name)
                    except: pass
                    save_image.save(image_save_name)
                    save_image.close()
                    image_count += 1
        else:
            image_count = 1
            save_final_directory = results_directory + 'Final/'
            save_final_enhanced_directory = results_directory + 'Final Enhanced/'
            try: os.makedirs(save_final_directory)
            except: pass
            try: os.makedirs(save_final_enhanced_directory)
            except: pass
            for final_image_batch in range(final_image_batch_count):
                noise = np.random.normal(0, 1, size=(batch_size,) + (1, 1, random_noise_size))
                generated_novel_images = generator.predict(noise)
                for i in range(batch_size):
                    image = generated_novel_images[i, :, :, :]
                    image += 1
                    image *= 127.5
                    save_image = Image.fromarray(image.astype(np.uint8))
                    image_save_name = save_final_directory + 'Image ' + str(image_count) + '.png'
                    try: os.remove(image_save_name)
                    except: pass
                    save_image.save(image_save_name)
                    image_save_name = save_final_enhanced_directory + 'Image ' + str(image_count) + '.png'
                    try: os.remove(image_save_name)
                    except: pass
                    width, height = save_image.size
                    save_image = save_image.resize((zoom * height, zoom * width), resample=Image.LANCZOS)
                    save_image = save_image.filter(ImageFilter.EDGE_ENHANCE)
                    save_image.save(image_save_name)
                    save_image.close()
                    image_count += 1
                # Creates an html index file for all images, Windows (sometimes)
                os.chdir(save_final_enhanced_directory)
                try: os.remove('index.html')
                except: pass
                if os.name == 'nt':
                    subprocess.call('for %i in (*.png) do echo ^<img src="%i" /^> >> index.html',
                                    shell=True, stdout=open(os.devnull, 'wb'))

        plt.figure(1)
        plt.plot(batches, adversarial_loss, color='green', label='Generator Loss')
        plt.plot(batches, discriminator_loss, color='blue', label='Discriminator Loss')
        plt.title("RedQueen Results")
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        if epoch == 0:
            plt.legend()
        results_save_name = results_directory + 'Statistics.png'
        try: os.remove(results_save_name)
        except: pass
        plt.savefig(results_save_name)

def main():
    image_shape = (image_length, image_width, 3)
    train_dcgan(batch_size, epochs, image_shape, source_directory)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = humanfriendly.format_timespan(round(end_time - start_time, 1))
    print("\nRuntime for GAN: " + str(duration))
    try:
        os.chdir(sys.path[0])
        playsound('.\cloak.mp3')
    except: pass

