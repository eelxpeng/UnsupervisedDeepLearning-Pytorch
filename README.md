# Unsupervised Deep Learning with Pytorch

This repository tries to provide unsupervised deep learning models with Pytorch for convenient use.

## Denoising Autoencoder
1-layer autoencoder. Corrupt the input (masking), then reconstruct the original input.

## Stacked Denoising Autoencoder
Layerwise pretraining by denoising autoenocder. Then stack all layers and finetune further.

## VAE
The famous Variational Autoencoder from the paper

`Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." ICLR (2014).`

## Convolutional VAE
VAE using convolutional and deconvolutional networks is demonstrated with SVHN dataset. Expample code is `test_convvae.py`. The reconstruction and samples generated is show as follows:

![SVHN Reconstruction](/figure/reconstruction_99.png)
![SVHN Sample](/figure/sample_99.png)

## Variational Deep Embedding
Implementation of Variational Deep Embedding from the IJCAI2017 paper:

`Jiang, Zhuxi, et al. "Variational deep embedding: An unsupervised and generative approach to clustering." International Joint Conference on Artificial Intelligence. 2017.`

The original code is written in [Keras](https://github.com/slim1017/VaDE). However, the original code is incorrect when computing the loss function. And I have corrected the loss function part with my code. The example usage can be found in `test/test_vade-3layer.py`, and it uses the pretrained weights from autoencoder in `test/model/pretrained_vade-3layer.pt`. 

Note: 

* The pretrained weights is important to initialize the weights of VaDE.
* Unlike the original code using combined training and test data for training and evaluation, I split the training and test data, and only use training data for training and test data for evaluation. I think it is a more appropriate way to evaluate the method for generalization.
* As found, with above evaluation scheme and training for 3000 epochs, the clustering accuracy achieved is 94\%.
