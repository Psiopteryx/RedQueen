from __future__ import print_function
import torch, torchvision, time, humanfriendly, os, sys, matplotlib, warnings, subprocess
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageFilter
from playsound import playsound

source_directory = '/source/'
save_directory = '/results/'

num_epochs = 25
epoch_image_batch_count = 2    # Program will cycle through entire batch of images x times
final_image_batch_count = 20
zoom = 3  # Zoom factor, and edge enhance final output

workers = 3  # threads for dataloader. 0 avoids multithread problems in Windows
batch_size = 128
image_size = 128
nc = 3  # colour channels RGB
nz = 200  # size of generator input noise
ngf = 128  # feature maps in generator
ndf = 64  # feature maps in discriminator. Dumb it down.

lrD = 0.0001  # learning rate for Discriminator optimiser. 0002 in literature
lrG = 0.0002  # learning rate for Generator optimizer. .0002 in literature. Higher than Discriminator?

beta1 = 0.5
ngpu = 1
device = torch.device("cuda:0")

dataset = dset.ImageFolder(root = source_directory,
                           transform = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=workers)

matplotlib.rcParams['animation.embed_limit'] = 2**128
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", "Possibly corrupt EXIF data", UserWarning)

# Initialises all Conv weights with mean = 0 and stdev = 0.2. Not sure why code is 0.02?! Experiment
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(image_size, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrG
                        , betas=(beta1, 0.999))

def generate_images(epoch):
    if (epoch < (num_epochs - 1)):
        save_epoch_directory = save_directory + "Epoch " + str(epoch) + '/'
        selected_image_batch_count = epoch_image_batch_count
    else:
        save_epoch_directory = save_directory + 'Final' + '/'
        selected_image_batch_count = final_image_batch_count
    for batch_count in range(selected_image_batch_count):
        with torch.no_grad():
            if batch_count == 0:
                painting_batch = netG(fixed_noise)  # Allows for first batch to show image evolution over time
            else:
                new_noise = torch.randn(image_size, nz, 1, 1, device=device)
                painting_batch = netG(new_noise)
            painting_batch = painting_batch.detach().cpu()
        for batch_image_count in range(painting_batch.shape[0]):
            painting = painting_batch[batch_image_count]
            #if batch_count == 0:
            #    print("Generating Batch: " + str(batch_count + 1) + " Image: " + str(batch_image_count + 1), end="")
            #else:
            #    print("\rGenerating Batch: " + str(batch_count + 1) + " Image: " + str(batch_image_count + 1), end="")
            painting_title = "Batch " + str(batch_count + 1) + " Image " + str(batch_image_count) + str(".jpg")
            painting_save_path = save_epoch_directory + painting_title
            try: os.makedirs(save_epoch_directory)
            except: pass
            try: os.remove(painting_save_path)
            except: pass
            torchvision.utils.save_image(painting, painting_save_path, normalize=True)

def final_enhance():
    print("Enhancing Final Images... ", end="")
    open_path = save_directory + 'Final/'
    enhanced_save_directory = save_directory + 'Final Enhanced/'
    try: os.makedirs(enhanced_save_directory)
    except: pass
    for filename in os.listdir(open_path):
        if filename.endswith(".jpg"):
            painting = Image.open(open_path + filename)
            width, height = painting.size
            painting = painting.resize((zoom * height, zoom * width), resample=Image.LANCZOS)
            painting = painting.filter(ImageFilter.EDGE_ENHANCE)
            try: os.remove(enhanced_save_directory + filename)
            except: pass
            painting.save(enhanced_save_directory + filename)
            painting.close()
    print("done.")
    # Creates an html index file for all images, Windows
    os.chdir(enhanced_save_directory)
    try: os.remove('index.html')
    except: pass
    if os.name == 'nt':
        subprocess.call('for %i in (*.jpg) do echo ^<img src="%i" /^> >> index.html',
                        shell=True, stdout=open(os.devnull, 'wb'))

img_list, G_losses, D_losses, D_x_record, D_G_z1_record, D_G_z2_record = [], [], [], [], [], []

def main():
    print("Starting " + str(num_epochs) + " Epoch Training Loop")
    for epoch in range(num_epochs):
        G_losses_pass, D_losses_pass, D_x_record_pass, D_G_z1_record_pass, D_G_z2_record_pass = [], [], [], [], []
        epoch_start_time = time.time()
        batch_start_time = time.time()
        for batch_number, data in enumerate(dataloader, 0):
            # Train discriminator with all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            #print("Label size: " + str(label.size()))
            # Forward pass the real batch through Discriminator
            output = netD(real_cpu).view(-1)
            #print("Output size: " + str(output.size()))
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            # Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train generator
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Forward another pass of all-fake through Discriminator as it was just updated
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            batch_end_time = time.time()
            batch_total_time = batch_end_time - batch_start_time
            batch_total_run_est_time = (batch_total_time * len(dataloader) * (num_epochs - 1 - epoch)) \
                                       - (batch_total_time*(batch_number / len(dataloader)))
            batch_total_ETA = humanfriendly.format_timespan(round(batch_total_run_est_time, 0))

            if batch_number != 0:
                print("\rEpoch: " + str(epoch) + "/" + str(num_epochs -1) + "\tBatch: " + str(batch_number) + "/"
                  + str(len(dataloader) - 1) + "\tETA Based on Batch Speed: " + batch_total_ETA, end="")
                G_losses_pass.append(errG.item())
                D_losses_pass.append(errD.item())
                D_x_record_pass.append(D_x)
                D_G_z1_record_pass.append(D_G_z1)
                D_G_z2_record_pass.append(D_G_z2)

            batch_start_time = time.time()

        fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        G_losses.append(np.nanmean(G_losses_pass))
        D_losses.append(np.nanmean(D_losses_pass))
        discriminator_accuracy = np.nanmean(D_x_record_pass)
        D_x_record.append(np.nanmean(discriminator_accuracy))
        generator_accuracy = np.nanmean(D_G_z2_record_pass)
        D_G_z1_record.append(np.nanmean(D_G_z1_record_pass))
        D_G_z2_record.append(generator_accuracy)

        epoch_end_time = time.time()
        epoch_time_elapsed = epoch_end_time - epoch_start_time
        total_est_time = epoch_time_elapsed * (num_epochs - 1 - epoch)
        ETA = humanfriendly.format_timespan(round(total_est_time, 0))
        if epoch != num_epochs:
            print("\nETA Based on Epoch Speed: " + ETA + "\tDiscriminator Accuracy: " + str(discriminator_accuracy)
                  + "\tGenerator Accuracy: " + str(generator_accuracy))
        generate_images(epoch)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = humanfriendly.format_timespan(round(end_time - start_time, 1))
    print("\nRuntime for GAN: " + str(duration))
    final_enhance()

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Accuracy")
    # plt.plot(G_losses, label = "Generator")
    # plt.plot(D_losses, label = "Discriminator")
    plt.plot(D_x_record, label="Disc Approach to 0.5")
    plt.plot(D_G_z1_record, label="Gen1 Approach to 0.5")
    plt.plot(D_G_z2_record, label="Gen2 Approach to 0.5")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    results_save_path = save_directory + 'Results.png'
    try: os.remove(results_save_path)
    except: pass
    plt.savefig(results_save_path)

    generated_images_save_path = save_directory + 'Sample Images.png'
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    try: os.remove(generated_images_save_path)
    except: pass
    plt.savefig(generated_images_save_path)

    # Cute animation
    animation_save_path = save_directory + 'Evolution.mp4'
    fig = plt.figure(figsize=(16, 8))
    plt.axis("off")
    plt.title("Generator Evolution")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=500, blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='Pandora AI'), bitrate=1800)
    try: os.remove(animation_save_path)
    except: pass
    ani.save(animation_save_path, writer=writer)

    # Opens files for display
    os.chdir(save_directory)
    os.startfile('Evolution.mp4')
    os.startfile('Results.png')
    os.startfile('Sample Images.png')

    try:
        os.chdir(sys.path[0])
        playsound('.\cloak_romulan.mp3')
    except: pass

