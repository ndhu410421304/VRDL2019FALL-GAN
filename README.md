# VRDL2019FALL-GAN
This is a implementation of WGANGP model.
Before use this model, you may neddd to first run the helper.py to help collecting required data.

The model had already been fine-tuned, butit may be possible to improve more. I had read the instruction from Pytorch-GAN repository, so it may be a good start before start to modify the model. What could be modify for this model may be:

## 1. Generator and discriminator
Remember to adjust input of the model based the model stucture. different structure may require different input, for example, the input noise to generator. Also mention that the strength of discriminator and generator should be compitable, which could help model to not easily been statisfy to its genertated result in early stage when generator is not well trained enough, which means, if discriminator is train faster, it could easily recognize fakes, then generator would not now how to improve itself; on the otherhand, if generator can easily fouled the discriminator, it would not try to improve itself furthermore.

## 2. Gradient Penalty Loss
Graident penalty loss provide additoonal information to improve discriminator loss for a more smoothing training process.



