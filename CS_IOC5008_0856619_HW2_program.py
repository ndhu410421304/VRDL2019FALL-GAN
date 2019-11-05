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

def output_fig(images_array, file_name="./results"):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()

os.makedirs("gpimages", exist_ok=True) # observation for saves
os.makedirs("gpimgsep", exist_ok=True) # observation for training(in epoch) 

parser = argparse.ArgumentParser()
parser.add_argument("--max_epochs", type=int, default=10000, help="big enough epoch count to maker it never ends")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--latent_dim", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--img_size", type=int, default=64, help="image size")
opt = parser.parse_args()
print(opt)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
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
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        img = self.model(noise)
        return img


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


def compute_gradient_penalty(D, real_samples, gen_samples):
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * gen_samples)).requires_grad_(True)
    interpolates_D = D(interpolates)
    gen = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
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


generator = Generator().cuda()
discriminator = Discriminator().cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))


dataset = datasets.ImageFolder(root="data",
                               transform=transforms.Compose([
                               transforms.CenterCrop(108),
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=0)


for epoch in range(opt.max_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = Variable(imgs.type(Tensor))
        optimizer_D.zero_grad()
        noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1).cuda()
        gen_imgs = generator(noise)
        real_validity = discriminator(real_imgs)
        gen_validity = discriminator(gen_imgs)
        if(gen_imgs.shape == real_imgs.shape):
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
            loss_D = -torch.mean(real_validity) + torch.mean(gen_validity) + 10 * gradient_penalty # mean of valididity _+ lambda factor * gradient penalty
            loss_D.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            if i % 5 == 0: # update per 5 batches
                gen_imgs = generator(noise)
                gen_validity = discriminator(gen_imgs)
                loss_G = -torch.mean(gen_validity)
                loss_G.backward()
                optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.max_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
                )

        if len(dataloader) - i == 1:
            os.makedirs("gpimgsep/ep%d"%(epoch+1), exist_ok=True) # create new folder for ttthis epoch
            num = 0
            for num in range(500):
                np.random.seed(num)
                noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1).cuda()
                gen_imgs = generator(noise)
                gen_imgs = gen_imgs.data[:9].cpu().numpy()
                gen_imgs = np.rollaxis(gen_imgs, 1, 4) #switch dimension to requirement
                output_fig(gen_imgs, file_name=("./gpimgsep/ep%d/{}_image"%(epoch+1)).format(str.zfill(str(num+1), 3)))
