import argparse
import os
import numpy as np
import math
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import helper
import matplotlib.pyplot as plt

cuda_tensor = torch.cuda.FloatTensor


# Outpuut figure function for generate desire output
def output_fig(images_array, file_name="./results"):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()


os.makedirs("gpimages", exist_ok=True)  # observation for saves
os.makedirs("gpimgsep", exist_ok=True)  # observation for training(in epoch)

# Parser for printing important parameters
parser = argparse.ArgumentParser()
parser.add_argument("--max_epochs", type=int, default=10000,
                    help="big enough epoch count to maker it never ends")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="size of the latent z vector")
parser.add_argument("--img_size", type=int, default=64, help="image size")
opt = parser.parse_args()
print(opt)


# Generator function
# conv 5 times to make image
# have same size as training image.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 100 for latent
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 3 for channels
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        img = self.model(noise)
        return img


# Change Discriminator to
# convolution layers, and do instancenorm
# before put in layers.
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.InstanceNorm2d(3),
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(-1)
        return out


# The main structure for this model is
# "WGanGP", which the main diiferent of
# "WGan" and "WGanGP" is that "WGanGP"
# contains addition gp(gradient penalty)
# loss.
def compute_gradient_penalty(D, real_samples, gen_samples):
    # Setup random paramter
    alpha = cuda_tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Random interpolate between real and generate image
    interpolates = (alpha * real_samples + (
        (1 - alpha) * gen_samples)).requires_grad_(True)
    interpolates_D = D(interpolates)
    gen = Variable(cuda_tensor(
        real_samples.shape[0]).fill_(1.0), requires_grad=False)
    # Get gradient interpoloate
    gradients = autograd.grad(
        outputs=interpolates_D,
        inputs=interpolates,
        grad_outputs=gen,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Put both generator and discriminator oin gpu for faster computation
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# Use adam as optimizer: froim gam hack
optimizer_g = torch.optim.Adam(
    generator.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(
    discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

# Data loader for loading data with
# data augmentation.
dataset = datasets.ImageFolder(root="data",
                               transform=transforms.Compose([
                                    # Get this information from
                                    # helper.py's formula:
                                    # main focus of picture(face)
                                    # should be 108*108 size from cemter
                                    #
                                    # After crop,
                                    # resize to size between 112 and 28
                                    transforms.CenterCrop(108),
                                    transforms.Resize(opt.img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
# Load preset dataset to dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=0)

# Main Training part
for epoch in range(opt.max_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = Variable(imgs.type(cuda_tensor))
        optimizer_d.zero_grad()
        noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1).cuda()
        # Genertate random image from random noise
        #
        # Calculate loss to evaluate discriminator
        gen_imgs = generator(noise)
        real_validity = discriminator(real_imgs)
        gen_validity = discriminator(gen_imgs)
        if(gen_imgs.shape == real_imgs.shape):
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs.data, gen_imgs.data)  # compute gp
            # mean of valididity _+ gp loss weight * gp
            loss_d = -torch.mean(
                real_validity) + torch.mean(
                    gen_validity) + 7.5 * gradient_penalty
            loss_d.backward()
            optimizer_d.step()  # update discriminator
            optimizer_g.zero_grad()
            if i % 5 == 0:  # update per 5 batches
                gen_imgs = generator(noise)
                gen_validity = discriminator(gen_imgs)
                loss_g = -torch.mean(gen_validity)
                loss_g.backward()
                optimizer_g.step()  # update generator
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.max_epochs, i, len(
                        dataloader), loss_d.item(), loss_g.item())
                )

        # For last batch of input of each epoch:
        # we generate rhe result output to a new folder
        # to evaluate the result during training.
        if len(dataloader) - i == 1:
            # create new folder for this epoch
            os.makedirs("gpimgsep/ep%d" % (epoch+1), exist_ok=True)
            num = 0
            for num in range(500):
                np.random.seed(num)  # change random seed base on num
                noise = torch.randn(
                    opt.batch_size, opt.latent_dim, 1, 1).cuda()
                gen_imgs = generator(noise)  # generate image from noise
                # Tensor move from cuda to cpu,
                # then trasform from tensor to numpy array
                gen_imgs = gen_imgs.data[:9].cpu().numpy()
                # switch dimension to requirement
                #
                # use output_fig to generate result
                gen_imgs = np.rollaxis(gen_imgs, 1, 4)
                output_fig(gen_imgs, file_name=(
                    "./gpimgsep/ep%d/{}_image" % (epoch+1)).format(
                        str.zfill(str(num+1), 3)))
