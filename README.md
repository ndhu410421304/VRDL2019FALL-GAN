# VRDL2019FALL-GAN
This is a implementation of WGANGP model testing on celebA .
Before use this model, you may neddd to first run the helper.py to help collecting required data.

The model had already been fine-tuned, butit may be possible to improve more. I had read the instruction from Pytorch-GAN repository, and try to start by modifying itsd code. So it may be a good start before start to modify the model. What could be modify for this model may be:

## 1. Generator and discriminator
Remember to adjust input of the model based the model stucture. different structure may require different input, for example, the input noise to generator. Also mention that the strength of discriminator and generator should be compitable, which could help model to not easily been statisfy to its genertated result in early stage when generator is not well trained enough, which means, if discriminator is train faster, it could easily recognize fakes, then generator would not now how to improve itself; on the otherhand, if generator can easily fouled the discriminator, it would not try to improve itself furthermore.

## 2. Gradient Penalty Loss
Graident penalty loss provide additoonal information to improve discriminator loss for a more smoothing training process. You can change the structure of how to compute gradient penalty if you want, or you can adjust the weight of gradient penalty first to see the performance change / loss change progress.

## 3. Data augmentation
This may not be a nust-do, but by doing so, you could enlarge your training dataset by only a few lines of could. For example, you can randomly flip the image to generatere flipped images, which could easily enlarge your dataset by 2x bigger of ral images. Other implemenmtio of data augmerntation may be welcome, but try not to generate too 'fake' images, for example, rotate an image in too big degree; which we could not expect the result of output to become better or not. Other thing ios that randomcrop is not going to work for this case, since for this celebA testing dataset, the human face were mostly located at center with size of 108*108. Since we only want face information for training, we would like to let each face were in same position. There is also more prescise data about where the faces were located in image, you can find it in dataset's webpage.

Something you could observe for model evaluation during training process:

## 1. Observe the output result of each epoch
This model well let generator to generate 500 * 9 result images for each epochs. Don't give up too soon, this model needs serveral epcohs to be able to learn a general face feature! You can also generate a observing image for each batch to more precisely observe traiing process.

## 2. The loss matters
For general: no matter how the model structure you constuct, rembemer the model may be coverge ones any loss is not changing much after serveral batches. For my case, I would observe the generator loss to check if it overfitted: if generator loss near -1 for long, it means that generator can easily foul the discrimiator by fake image, so it may not try to further improve itself; if loss near 0 for long, it means the discriminator can easily recognize the fake image, which would let generator don't know how to improve. By checking Loss could quicky recognize model's current status.

Finally before starting, yopu nay like to check if you have an GPU! If not, try to replace .cuda parts by .cpu to train the model on cpu, but the better solution would be finiding an accessabler online GPU like colab to train your model. Goodluck! 



