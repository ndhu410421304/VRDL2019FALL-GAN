#
# This is a code for gatheriung required training data
#

import os
import helper
import matplotlib.pyplot as plt
from glob import glob
import torch
use_gpu = True if torch.cuda.is_available() else False

data_dir = './data'
helper.download_extract(data_dir)

show_n_images = 9
image_size = 64
plt.figure(figsize=(10, 10))
images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], image_size, image_size, 'RGB')
plt.imshow(helper.images_square_grid(images))
plt.show()


width = height = 56
generated_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:9], 112, 112, 'RGB')

# Your generated_images.shape should be [9, width, height, 3]
print(generated_images.shape)

def output_fig(images_array, file_name="./results"):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)

output_fig(generated_images)

print(generated_images.dtype)
