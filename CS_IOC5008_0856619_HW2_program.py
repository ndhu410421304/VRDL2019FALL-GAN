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

Tensor = torch.cuda.FloatTensor

# outpuut figure function for generate desire output
def output_fig(images_array, file_name="./results"):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()

os.makedirs("gpimages", exist_ok=True) # observation for saves
os.makedirs("gpimgsep", exist_ok=True) # observation for training(in epoch) 

# parser for printing important parameters 
parser = argparse.ArgumentParser()
parser.add_argument("--max_epochs", type=int, default=10000, help="big enough epoch count to maker it never ends")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--latent_dim", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--img_size", type=int, default=64, help="image size")
opt = parser.parse_args()
print(opt)

# generator function
# conv 5 times to make image
# have same size as training image.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False), # 100 for latent
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
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), # 3 for channels
            nn.Tanh()
        )

    def forward(self, noise):
        img = self.model(noise)
        return img

# Using only full connected layer
# for faster output and not-too-strong
# discriminator.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod((3, opt.img_size, opt.img_size))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

# The main structure for this model is
# "WGanGP", which the main diiferent of
# "WGan" and "WGanGP" is that "WGanGP"
# contains addition gp(gradient penalty)
# loss.
def compute_gradient_penalty(D, real_samples, gen_samples):
    # setup random paramter
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # use random parameter to random interpolate between real and generate image
    interpolates = (alpha * real_samples + ((1 - alpha) * gen_samples)).requires_grad_(True)
    interpolates_D = D(interpolates)
    gen = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # get gradient interpoloategithub
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

# put both generator and discriminator oin gpu for faster computation
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# use adam as optimizer: froim gam hack
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

# Data loader for loading data with
# data augmentation.
dataset = datasets.ImageFolder(root="data",
                               transform=transforms.Compose([
                               #get this information friom helper.py's formula
                               transforms.CenterCrop(108),
                               transforms.Resize(opt.img_size), # resize to size between 112 and 28
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# load preset dataset to dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=0)

# Main Training part
for epoch in range(opt.max_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = Variable(imgs.type(Tensor))
        optimizer_D.zero_grad()
        noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1).cuda()
        gen_imgs = generator(noise) # genertate random image from random noise
        real_validity = discriminator(real_imgs) # loss of discriminator evaluate real image
        gen_validity = discriminator(gen_imgs) # loss of discriminator evaluate generated image
        if(gen_imgs.shape == real_imgs.shape):
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data) # compute gp
            loss_D = -torch.mean(real_validity) + torch.mean(gen_validity) + 10 * gradient_penalty # mean of valididity _+ gp loss weight * gp
            loss_D.backward()
            optimizer_D.step() # update discriminator 
            optimizer_G.zero_grad()
            if i % 5 == 0: # update per 5 batches
                gen_imgs = generator(noise)
                gen_validity = discriminator(gen_imgs)
                loss_G = -torch.mean(gen_validity)
                loss_G.backward()
                optimizer_G.step() # update generator
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.max_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
                )

        # For last batch of input of each epoch:
        # we generate rhe result output to a new folder
        # to evaluate the result during training.
        if len(dataloader) - i == 1:
            os.makedirs("gpimgsep/ep%d"%(epoch+1), exist_ok=True) # create new folder for ttthis epoch
            num = 0
            for num in range(500):
                np.random.seed(num) # change random seed base on num
                noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1).cuda()
                gen_imgs = generator(noise) # generate image from noise
                gen_imgs = gen_imgs.data[:9].cpu().numpy() # tensor move from cuda to cpu, then trasform from tensor to numpy array
                gen_imgs = np.rollaxis(gen_imgs, 1, 4) #switch dimension to requirement
                output_fig(gen_imgs, file_name=("./gpimgsep/ep%d/{}_image"%(epoch+1)).format(str.zfill(str(num+1), 3))) # use output_fig to generate result
