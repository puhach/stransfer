import tensorflow as tf
import tensorflow_hub as hub
# TODO: later use VGG19 and, probably, Resnet, Inception etc
from tensorflow.keras.applications import VGG16
import imageio

print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)


content_img = imageio.imread('data/content/chicago.jpg')

model = VGG16(include_top=True, weights='imagenet')
print(model.summary())
content_prep = tf.keras.applications.vgg16.preprocess_input(content_img)
content_prep = tf.image.resize(content_prep, size=(224,224))
content_prep = content_prep[tf.newaxis, :] # expand dims
prob = model(content_prep)
pred = tf.keras.applications.vgg16.decode_predictions(prob.numpy(), top=3)
print(pred)

for layer in model.layers:
  print(layer.name, ":", type(layer))
