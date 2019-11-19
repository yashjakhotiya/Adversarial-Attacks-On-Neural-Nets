# Adversarial-Attacks-On-Neural-Nets

Contains code to attack neural networks adversarially.

## FGSM on MobileNetV2

Weights corresponding to MobileNetV2 trained on Imagenet are downloaded (and stored in `~/.keras/` directory).

An image from the `sample_images/` directory is given to the model, after preprocessing, to classify.

Next, a perturbation matrix is generated using the fast gradient sign method as described in [this paper](https://arxiv.org/abs/1412.6572)
and added to the original image in varying magnitudes. The adversarial image, thus created, is given to the model which then misclassifies
it, thereby completing the adversarial attack.



