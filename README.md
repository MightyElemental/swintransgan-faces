# Generating Images Using Generative Deep Learning
## How batch and window sizes affect image quality in Swin Transformer GANs

This is my university undergraduate dissertation on the use of SwinTransformer GANs and how batch size and window size affects generated image quality. It uses PyTorch to define and train the models. Details can be found in the included PDF file.

The project was undertaken on the Viking Cluster, which is a high performance compute facility provided by the University of York. I am grateful for the computational support from the University of York High Performance Computing service, Viking and the Research Computing team.

## Abstract
With the recent interest in transformers and image generation, the need to test novel methods arises. This paper investigates how window size and batch size affects SwinTranformer GAN image quality. All models had the same number of parameters - 10M for the generator, and 1.7M for the discrimintor. The size difference between the generator and discriminator is because the generator uses transformers whereas the discriminator uses a simpler DCGAN. After testing nine different models, the best was found to have a window size of 8 and a batch size of 100 (FID score of 88.4). The worst was found to have a window size of 8 and batch size of 200 (FID score of 180.6). Future investigation could be undertaken to study the effects of learning rate, attention head count, and transformer layer count. Switching to use a Wasserstein discriminator could also beneficial as it could suppress the mode collapse issues encounted during training.

## Sample Results

### Hand-picked generated images from a set of 5000 images
Each image in this set was visually inspected by me to compile a set of what I believe to be the best examples.

![Hand-picked generated images from a set of 5000 images](imgs/BestManualSelected.png)

### Best set (window size 8, batch size 100)
Unlike the hand-picked set, this set used random initial vectors for each image.

![8-100](imgs/8-100.jpg)

### Worst set (window size 8, batch size 200)
This set also used random initial vectors for each image. However, the quality on this set is the worst out of any experiment I recorded in the project. Note the repeated faces and general lack of feature definition.

![8-200](imgs/8-200.jpg)