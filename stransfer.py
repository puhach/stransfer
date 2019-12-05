import tensorflow as tf
import tensorflow_hub as hub
# TODO: later use VGG19 and, probably, Resnet, Inception etc
from tensorflow.keras.applications import VGG16
import imageio

print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)


content_img = imageio.imread('data/content/chicago.jpg')
style_img = imageio.imread('data/style/wave.jpg')

 # Content layers to get content feature maps
content_layers = ['block4_conv2'] 

# Style layers of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']


def preprocess_image(image):
  image_prep = tf.keras.applications.vgg16.preprocess_input(image)
  image_prep = tf.image.resize(image_prep, size=(224, 224))
  image_prep = image_prep[tf.newaxis, :]
  return image_prep


def create_feature_extractor(layer_names):
  
  vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
  vgg.trainable = False

  print(vgg.summary())

  outputs = [vgg.get_layer(layer_name).output for layer_name in layer_names]

  model = tf.keras.Model(vgg.inputs, outputs)

  return model


layers_of_interest = content_layers + style_layers
feature_extractor = create_feature_extractor(layers_of_interest)
""" content_prep = preprocess_image(content_img)
content_prep = tf.constant(content_prep)
outputs = feature_extractor(content_prep)

for name, output in zip(layers_of_interest, outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()
 """


# preprocess images
content_prep = preprocess_image(content_img)
style_prep = preprocess_image(style_img)

# get content and style features only once before training
content_features = feature_extractor(content_prep)
content_features = content_features[:len(content_layers)]

style_features = feature_extractor(style_prep)
style_features = style_features[len(content_layers):]
#content_features = get_features(content_prep, feature_extractor)
#style_features = get_features(style_prep, feature_extractor)

# calculate the gram matrices for each layer of our style representation
#style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start off with the target as a copy of our *content* image
# then iteratively change its style
#target = tf.Variable(content_prep)
#target = content.clone().requires_grad_(True).to(device)