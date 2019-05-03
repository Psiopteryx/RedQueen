import time, os, sys, humanfriendly, warnings, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFilter

source_directory = '/birds/'
results_directory = '/results/'

image_size = 64
batch_size = 64

discriminator_learn_rate = 0.0002
generator_learn_rate = 0.00015

epochs = 3
leaky_ReLU = 0.2
label_noise = 0.2  # True/false noise for discriminator

epoch_image_batch_count = 2    # Cycle through entire batch of images x times + fixed_noise batch
final_image_batch_count = 5
zoom = 3  # Zoom factor, and edge enhance final output

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

    discriminator.add(Flatten())
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))
    optimizer = Adam(discriminator_learn_rate, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=None)

    return discriminator

def build_generator():
    generator = Sequential()
    generator.add(Dense(units=4 * 4 * 512, kernel_initializer='glorot_uniform', input_shape=(1, 1, 100)))
    generator.add(Reshape(target_shape=(4, 4, 512)))
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
    gs1 = gridspec.GridSpec(8,8)
    gs1.update(wspace=0, hspace=0)

    for i in range(batch_size):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = generated_images[i, :, :, :]
        image +=1
        image *= 127.5
        fig = plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    if (epoch < epochs - 1):
        gallery_save_name = results_directory + 'Epoch ' + str(epoch) + ' Gallery.png'
    else: gallery_save_name = results_directory + 'Final ' + ' Gallery.png'
    try: os.remove(gallery_save_name)
    except: pass
    plt.savefig(gallery_save_name, bbox_inches='tight', pad_inches=0)
    plt.pause(0.000000000001)
    plt.show()

def train_dcgan(batch_size, epochs, image_shape, source_directory):
    generator = build_generator()
    discriminator = build_discriminator(image_shape)

    gan = Sequential()
    discriminator.trainable = False  # Only for adversarial model
    gan.add(generator)
    gan.add(discriminator)
    optimizer = Adam(lr=generator_learn_rate, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=None)

    image_shape = (image_size, image_size, 3)
    dataset, num_images = load_dataset(source_directory, batch_size, image_shape)
    num_batches = int(num_images / batch_size)

    adversarial_loss = np.empty(shape=1)
    discriminator_loss = np.empty(shape=1)
    batches = np.empty(shape=1)

    try: os.makedirs(results_directory)
    except: pass
    fixed_noise = np.random.normal(0, 1, size=(batch_size,) + (1, 1, 100))  # for a fixed batch over time
    batch_count = 0
    plt.ion()
    for epoch in range(epochs):
        for batch_number in range(num_batches):
            batch_start_time = time.time()
            real_images = dataset.next()
            real_images /= 127.5  # normalise
            real_images -= 1
            current_batch_size = real_images.shape[0]  # for final batch (smaller)

            noise = np.random.normal(0, 1, size=(current_batch_size,) + (1,1,100))
            generated_images = generator.predict(noise)

            # introduce noise to discriminator True/False label
            real_y = (np.ones(current_batch_size) - np.random.random_sample(current_batch_size) * label_noise)
            fake_y = np.random.random_sample(current_batch_size) * label_noise

            # train discriminator
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(real_images, real_y)
            d_loss += discriminator.train_on_batch(generated_images, fake_y)
            discriminator_loss = np.append(discriminator_loss, d_loss)

            # train generator
            discriminator.trainable = False
            noise = np.random.normal(0, 1, size=(current_batch_size * 2,) + (1, 1, 100))

            # noise to discriminator True/False label again
            fake_y = (np.ones(current_batch_size * 2) - np.random.random_sample(current_batch_size *2) * label_noise)

            g_loss = gan.train_on_batch(noise, fake_y)
            adversarial_loss = np.append(adversarial_loss, g_loss)
            batches = np.append(batches, batch_count)

            time_elapsed = time.time() - start_time

            print("\rEpoch: " + str(epoch) + "/" + str(epochs) + "\tBatch: " + str(batch_number + 1)
                  + "/" + str(num_batches) + "\tGenerator loss: " + str(round(g_loss, 2)) + "\tDscriminator loss: "
                  + str(round(d_loss, 2)) + '\tBatch time: ' + str(round(time_elapsed, 0)) + ' sec', end="")

            batch_count += 1

        generated_images = generator.predict(fixed_noise)
        # Create a thumbnail gallery
        save_sample_images(generated_images, epoch)
        # save full galleries
        if(epoch < epochs - 1):
            for epoch_image_batch in range(epoch_image_batch_count):
                noise = np.random.normal(0, 1, size=(batch_size,) + (1, 1, 100))
                generated_novel_images = generator.predict(noise)
                save_epoch_directory = results_directory + 'Epoch ' + str(epoch) + '/'
                try: os.makedirs(save_epoch_directory)
                except: pass
                for i in range(batch_size):
                    image = generated_novel_images[i, :, :, :]
                    image += 1
                    image *= 127.5
                    save_image = Image.fromarray(image.astype(np.uint8))
                    image_save_name = save_epoch_directory +' Image ' + str(i) + '.png'
                    try: os.remove(image_save_name)
                    except: pass
                    save_image.save(image_save_name)
                    save_image.close()
        else:
            save_final_directory = results_directory + 'Final ' + '/'
            try: os.makedirs(save_final_directory)
            except: pass
            for final_image_batch in range(final_image_batch_count):
                noise = np.random.normal(0, 1, size=(batch_size,) + (1, 1, 100))
                generated_novel_images = generator.predict(noise)
                for i in range(batch_size):
                    image = generated_novel_images[i, :, :, :]
                    image += 1
                    image *= 127.5
                    save_image = Image.fromarray(image.astype(np.uint8))
                    image_save_name = save_final_directory + 'Image ' + str(i) + '.png'
                    try: os.remove(image_save_name)
                    except: pass
                    width, height = save_image.size
                    save_image = save_image.resize((zoom * height, zoom * width), resample=Image.LANCZOS)
                    save_image = save_image.filter(ImageFilter.EDGE_ENHANCE)
                    save_image.save(image_save_name)
                    save_image.close()
                # Creates an html index file for all images, Windows
                os.chdir(save_final_directory)
                try: os.remove('index.html')
                except: pass
                if os.name == 'nt':
                    subprocess.call('for %i in (*.jpg) do echo ^<img src="%i" /^> >> index.html',
                                    shell=True, stdout=open(os.devnull, 'wb'))

        plt.figure(1)
        plt.plot(batches, adversarial_loss, color='green', label='Generator Loss')
        plt.plot(batches, discriminator_loss, color='blue', label='Discriminator Loss')
        plt.title("RedQueen Results")
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        if epoch == 0:
            plt.legend()
        plt.pause(0.0000000001)
        plt.show()
        results_save_name = results_directory + 'Statistics.png'
        try: os.remove(results_save_name)
        except: pass
        plt.savefig(results_save_name)

def main():
    image_shape = (64, 64, 3)
    train_dcgan(batch_size, epochs, image_shape, source_directory)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = humanfriendly.format_timespan(round(end_time - start_time, 1))
    print("\nRuntime for GAN: " + str(duration))
