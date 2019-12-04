import tensorflow as tf
import tensorflow_hub as hub
# TODO: later use VGG19 and, probably, Resnet, Inception etc
from tensorflow.keras.applications import VGG16
import imageio

print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)


content_img = imageio.imread('data/content/chicago.jpg')

""" model = VGG16(include_top=True, weights='imagenet')
print(model.summary())
content_prep = tf.keras.applications.vgg16.preprocess_input(content_img)
content_prep = tf.image.resize(content_prep, size=(224,224))
content_prep = content_prep[tf.newaxis, :] # expand dims
prob = model(content_prep)
pred = tf.keras.applications.vgg16.decode_predictions(prob.numpy(), top=3)
print(pred)

for layer in model.layers:
  print(layer.name, ":", type(layer))
 """

 # Content layers to get content feature maps
content_layers = ['block4_conv2'] 

# Style layers of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']


def create_feature_extractor(layer_names):
  
  vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
  vgg.trainable = False

  print(vgg.summary())

  outputs = [vgg.get_layer(layer_name).output for layer_name in layer_names]

  model = tf.keras.Model(vgg.inputs, outputs)

  return model


layers_of_interest = content_layers + style_layers
feature_extractor = create_feature_extractor(layers_of_interest)
content_prep = tf.keras.applications.vgg16.preprocess_input(content_img)
content_prep = tf.image.resize(content_prep, size=(224, 224))
content_prep = content_prep[tf.newaxis, :]
outputs = feature_extractor(content_prep)

for name, output in zip(layers_of_interest, outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()