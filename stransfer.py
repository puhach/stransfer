import tensorflow as tf
import tensorflow_hub as hub
# TODO: later use VGG19 and, probably, Resnet, Inception etc
from tensorflow.keras.applications import VGG16
import imageio

print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)




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

def compute_gram_matrix(layer_features):

  # get the batch_size, depth, height, and width of the Tensor
  b, h, w, d = layer_features.shape

  assert b==1, "The function expects features extracted from a single image."

  # reshape so we're multiplying the features for each channel
  #tensor = tensor.view(d, h * w)
  tensor = tf.reshape(layer_features, [h*w, d])
  
  # calculate the gram matrix
  gram = tf.matmul(tf.transpose(tensor), tensor)
  #gram = torch.mm(tensor, tensor.t())

  return gram




content_img = imageio.imread('data/content/chicago.jpg')
style_img = imageio.imread('data/style/wave.jpg')

# preprocess images
content_prep = preprocess_image(content_img)
style_prep = preprocess_image(style_img)


 # Content layers to get content feature maps
content_layers = ['block4_conv2'] 

# Style layers of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']


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



# get content and style features only once before training
input_content_features = feature_extractor(content_prep)
input_content_features = input_content_features[:len(content_layers)]

input_style_features = feature_extractor(style_prep)
input_style_features = input_style_features[len(content_layers):]

# map content layers to the features extracted from these layers
content_targets = { layer_name : content_layer_feats for 
                    layer_name, content_layer_feats in zip(content_layers, input_content_features) }

for content_layer_name, content_layer_features in content_targets.items():
  print(content_layer_name)
  print(content_layer_features.shape)

# calculate the gram matrices for each layer of our style representation
#style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
style_targets = { layer_name: compute_gram_matrix(style_layer_feats) for 
                  layer_name, style_layer_feats in zip(style_layers, input_style_features)} 

for style_target_name, style_target_gram in style_targets.items():
  print(style_target_name)
  print(style_target_gram.shape)


# Create a third output image and prepare it for change.
# To make this quick, start off with a copy of our content image, then iteratively change its style.
output_image = tf.Variable(content_prep)
#target = content.clone().requires_grad_(True).to(device)


optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

