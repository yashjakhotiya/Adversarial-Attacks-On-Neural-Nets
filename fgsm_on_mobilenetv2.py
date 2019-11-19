import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

image_height = 224
image_width = 224

image_path = os.path.join('sample_images', 'car_on_road.jpg')
loss = tf.keras.losses.CategoricalCrossentropy()
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (image_height, image_width))
    image = image[None, ...]
    return image

def create_adversarial_pattern(input_image, input_label):
    prediction = pretrained_model(input_image)
    image_loss = loss(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tf.gradients(image_loss, input_image)

    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False

def display_image(image, description):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        image_to_plot = sess.run(image)
    plt.figure()
    plt.imshow(image_to_plot.reshape(image_height, image_width, 3))
    image_probs = pretrained_model.predict(image, steps=1)
    image_class, image_description, class_confidence = decode_predictions(image_probs, top=1)[0][0]
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description, image_description, class_confidence*100))
    plt.show()
    return image_probs

image_raw = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image_raw)

image = preprocess(image)
description = "Original Image"
image_probs = display_image(image, description)

index = np.max(image_probs)
label = tf.one_hot(index, image_probs.shape[-1])

perturbation = create_adversarial_pattern(image, label)[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    image_to_plot = sess.run(perturbation)
plt.figure()
plt.imshow(image_to_plot[0])
plt.title("FGSM Generated Perturbation")
plt.show()

epsilons = [0.05, 0.085]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbation
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    _ = display_image(adv_x, descriptions[i])