import tensorflow as tf
import tensorflow_hub as hub
# TODO: later use VGG19 and, probably, Resnet, Inception etc
from tensorflow.keras.applications import VGG16


def load_model(name, weights='imagenet'):

  if name.lower() == "vgg16":
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
  elif name.lower() == "vgg19":
    model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  elif name.lower() == "inception_v3":
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
  else:
    raise Exception(f'Model "{name}" is not supported.')

  model.trainable = False

  return model

def extract_conv_layers(model):

  print(model.summary())

  # Extract convolutional layers from the model
  conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
  #print(conv_layers)
  return conv_layers
  



def create_feature_extractor(vgg, layers_of_interest):
  
  #vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
  #vgg.trainable = False


  outputs = {layer_name : vgg.get_layer(layer_name).output for layer_name in layers_of_interest}
  #outputs = [vgg.get_layer(layer_name).output for layer_name in layer_names]
  #outputs = [vgg.get_layer(layer_name).output for layer_name, layer_weight in layers.items() if layer_weight > 0]

  model = tf.keras.Model(vgg.inputs, outputs)

  return model

def compute_gram_matrix(layer_features):

  # Get the batch_size, depth, height, and width of the Tensor
  b, h, w, d = layer_features.shape

  assert b==1, "The function expects features extracted from a single image."

  # Reshape so we're multiplying the features for each channel
  tensor = tf.reshape(layer_features, [h*w, d])
  
  # Calculate the gram matrix
  gram = tf.matmul(tensor, tensor, transpose_a=True)

  return gram

def build_content_layer_map(features, content_layers):
    
    # TODO: describe expected features shape: (1, H, W, C) ?

    #content_map = { layer_name : layer_feats for 
    #                layer_name, layer_feats in 
    #                zip(content_layers, features[:len(content_layers)]) }

    content_map = { layer_name : features[layer_name] for layer_name in content_layers }
    #content_map = dict(filter(lambda f: f[0] in content_layers, features))

    return content_map


def build_style_layer_map(features, style_layers):
    
    # TODO: describe expected features shape: (1, H, W, C) ?

    # Each layer's Gram matrix is divided by height*width of the feature map. It makes easier to calculate 
    # the style loss afterwards:
    # https://github.com/udacity/deep-learning-v2-pytorch/issues/174
    
    gram_norm = lambda f: compute_gram_matrix(f) / (f.shape[1]*f.shape[2])
    style_map = { layer_name : gram_norm(features[layer_name])
                  for layer_name in style_layers
                }
                
    #style_map = { layer_name: compute_gram_matrix(layer_feats)/(layer_feats.shape[1]*layer_feats.shape[2]) for 
    #              layer_name, layer_feats in zip(style_layers, features[len(features)-len(style_layers):])} 

    return style_map
