import tensorflow as tf
import tensorflow_hub as hub
# TODO: later use VGG19 and, probably, Resnet, Inception etc
from tensorflow.keras.applications import VGG16
import imageio
from featureextractor import load_model, create_feature_extractor, build_content_layer_map, build_style_layer_map
#from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
import random
import streamlit as st
#import altair as alt
import pandas as pd
import glob
import os
#import uuid

print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)




#def preprocess_image(image):
def adjust_shape(image, size):  
  # TODO: Allow a user to tweak this size
  image_prep = tf.image.resize(image, size=size, method='lanczos5')  # resize appropriately 
  #image_prep = tf.image.resize(image, size=(224, 224), method='lanczos5')  # resize appropriately 
  image_prep = image_prep[tf.newaxis, ..., :3]  # add the batch dimension and discard the alpha channel
  return image_prep


def preprocess_image(image):
  # TODO: add model name parameter to choose which model we aim at
  return tf.keras.applications.vgg16.preprocess_input(image)


def get_layer_weights(model):
  #if st.sidebar.button('Rerun'):
  #  raise st.ScriptRunner.RerunException(st.ScriptRequestQueue.RerunData(_get_widget_states()))
  
  # Extract convolutional layers from the model
  conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
  #print(conv_layers)

  assert len(conv_layers) > 0, "The model has no convolutional layers."

  max_layer_weight = 10.0

  # Initialize chosen layers with random weights and give the rest the weight of zero.
  layer_weight_func = lambda layer_name, chosen_layers, suffix : st.sidebar.slider(
      label=layer_name, min_value=0.0, max_value=max_layer_weight,
      value=random.uniform(a=0.0, b=max_layer_weight) if layer_name in chosen_layers else 0.0,
      step=0.01, key='slider_'+layer_name+'_'+suffix)

  st.sidebar.subheader("Content weights")

  chosen_content_layers = random.sample(conv_layers, k=1)
  
  content_weights = {
    name : layer_weight_func(name, chosen_content_layers, 'content')
    for name in conv_layers 
  }

  st.sidebar.subheader("Style weights")

  chosen_style_layers = random.sample(conv_layers, k=3)

  style_weights = {
    #name : st.sidebar.slider(label=name, min_value=0.0, max_value=1.0, value=0.5)
    name : layer_weight_func(name, chosen_style_layers, 'style')
    for name in conv_layers
  }

  return content_weights, style_weights


# Initialize GUI

st.title("Neural Style Transfer")

st.sidebar.header("Settings")

progress_text = st.empty()
progress_text.text("Preparing...")


# Select the content image

content_path = st.sidebar.selectbox('Content image', options=glob.glob('data/content/*.*'),
                                    format_func=lambda glob_path: os.path.basename(glob_path))

#content_img = imageio.imread('data/content/irynka.png')
content_img = imageio.imread(content_path)

st.sidebar.image(image=np.asarray(content_img), use_column_width=True, 
  caption=None, clamp=True, channels='RGB')


# Select the style image

style_path = st.sidebar.selectbox('Style image', options=glob.glob('data/style/*.*'),
                                  format_func=lambda glob_path: os.path.basename(glob_path))

#style_img = imageio.imread('data/style/wave.jpg')
style_img = imageio.imread(style_path)

st.sidebar.image(image=np.asarray(style_img), use_column_width=True, 
  caption=None, clamp=True, channels='RGB')


# Specify the resolution the input images should be resized to before they are passed to VGG network

size = st.sidebar.slider( label='Intermediate image size', min_value=100, max_value=1000, value=500, 
                          step=1, format='%d')

# Resize the images and add the batch dimension
content_resized = adjust_shape(content_img, (size, size))
style_resized = adjust_shape(style_img, (size, size))

#st.sidebar.image(image=[np.asarray(content_img), np.asarray(style_img)], use_column_width=True, 
#  caption=['Content image', 'Style image'], clamp=True, channels='RGB')

# Preprocess the images
content_prep = preprocess_image(content_resized)
style_prep = preprocess_image(style_resized)


# Instantiate VGG network
vgg = load_model('VGG16')

get_layer_weights(vgg)

# Weights for each content layer
content_layer_weights = { 'block1_conv2' : 1.0,
                          'block5_conv1' : 0.5, 
                          'block5_conv3' : 0.2
                        }


# Weights for each style layer. Weighting earlier layers more will result in larger style artifacts.
# TODO: later try to combine it with style_layers
style_layer_weights = { 'block1_conv1': 1.,
                        'block2_conv1': 0.75,
                        'block3_conv1': 0.2,
                        'block4_conv1': 0.2,
                        'block5_conv1': 0.2
                      }

#assert len(style_layer_weights) == len(style_layers), "Style layer weights mismatch the style layer names"


# Content layers to get content feature maps
#content_layers = ['block1_conv2', 
#                  'block5_conv1',
#                  'block5_conv3'] 

# Style layers of interest
#style_layers = ['block1_conv1',
#                'block2_conv1',
#                'block3_conv1', 
#                'block4_conv1', 
#                'block5_conv1']


#layers_of_interest = content_layers + style_layers
layers_of_interest = list(set(content_layer_weights).union(style_layer_weights))
feature_extractor = create_feature_extractor(vgg, layers_of_interest)


# Get content and style features only once before training.
# TODO: Perhaps, try tf.constant() here
input_content_features = feature_extractor(content_prep)

input_style_features = feature_extractor(style_prep)

# map content layers to the features extracted from these layers
content_targets = build_content_layer_map(input_content_features, content_layer_weights.keys())
#import timeit
#t = timeit.timeit(lambda : build_content_layer_map(input_content_features, content_layer_weights.keys()) , number=100)
#print("time:", t)

#for content_layer_name, content_layer_features in content_targets.items():
#  print(content_layer_name)
#  print(content_layer_features.shape)

# calculate the gram matrices for each layer of our style representation
style_targets = build_style_layer_map(input_style_features, style_layer_weights.keys())
#import timeit
#t = timeit.timeit(lambda : build_style_layer_map(input_style_features, style_layer_weights.keys()), number=100)
#print("time:", t)

#for style_target_name, style_target_gram in style_targets.items():
#  print(style_target_name)
#  print(style_target_gram.shape)


# Just like in the paper, we define an alpha (content_weight) and a beta (style_weight). This ratio will affect 
# how stylized the final image is.
# TODO: perhaps, we could get by style layer weights and, similarly, content layer weights
content_weight = 1  # alpha
style_weight = 1e1  # beta



# Create a third output image and prepare it for change.
# To make this quick, start off with a copy of our content image, then iteratively change its style.
# For TF docs: GradientTape records operations if they are executed within its context manager and at least one
# of their inputs is being "watched". Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, 
# where trainable=True is default in both cases) are automatically watched. Tensors can be manually watched 
# by invoking the watch method on the GradientTape context manager.
#output_image = tf.Variable(content_resized / 255.0)
output_image = tf.Variable(content_resized)
print(output_image.numpy().min(), output_image.numpy().max())

#optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
optimizer = tf.optimizers.Adam(learning_rate=0.8)


# TODO: it might be interesting to create a video of images obtained after each epoch to see 
# how style transfer is progressing.

progress_bar = st.progress(0)
output_image_placeholder = st.empty()
#output_image_placeholder.text("Preparing...")

epochs = 20

for epoch in range(1, epochs+1):

  # Report progress
  progress_text.text(f"Step {epoch}/{epochs}")
  progress_bar.progress(epoch/epochs)
  print(f"Epoch {epoch}")

  with tf.GradientTape() as tape: # Record operations for automatic differentiation

    # Preprocess the output image before we pass it to VGG
    output_prep = preprocess_image(output_image)
    #output_prep = preprocess_image(output_image*255)

    # Extract content and style features from the output image
    output_features = feature_extractor(output_prep)
    #output_features = feature_extractor(output_image)
    output_content_map = build_content_layer_map(output_features, content_layer_weights.keys())
    output_style_map = build_style_layer_map(output_features, style_layer_weights.keys())


    # Calculate the content loss
    content_loss = tf.add_n([content_layer_weight * tf.reduce_mean(
                            (output_content_map[content_layer_name] - content_targets[content_layer_name])**2) 
                            for content_layer_name, content_layer_weight in content_layer_weights.items()
                            if content_layer_weight > 0 ]) 
    #content_loss = tf.add_n( [tf.reduce_mean((output_content_map[content_layer] - content_targets[content_layer])**2) 
    #                          for content_layer in content_layers ]) 

    # Calculate the style loss
    style_loss = tf.add_n([style_layer_weight * tf.reduce_mean(
                          (output_style_map[style_layer_name] - style_targets[style_layer_name])**2 ) 
                          for style_layer_name, style_layer_weight in style_layer_weights.items()
                          if style_layer_weight > 0]) 
    #style_loss = tf.add_n([style_layer_weights[style_layer] * tf.reduce_mean(
    #                      (output_style_map[style_layer] - style_targets[style_layer])**2 ) 
    #                      for style_layer in style_layers])  

    # Add up the content and style losses
    # TODO: Later we can try to use layer weights for both content and style maps instead of these factors
    total_loss = content_weight*content_loss + style_weight * style_loss

  # Calculate loss gradients
  grads = tape.gradient(total_loss, output_image)  

  print("before gradients:")
  print(output_image.numpy().min(), output_image.numpy().max())

  
  # Apply the gradients to alter the output image 
  optimizer.apply_gradients([(grads, output_image)])

  print("grads:")
  print(grads)

  print("after gradients:")
  print(output_image.numpy().min(), output_image.numpy().max())

  # Keep the pixel values between 0 and 255
  #output_image.assign(tf.clip_by_value(output_image, clip_value_min=0.0, clip_value_max=1.0))
  output_image.assign(tf.clip_by_value(output_image, clip_value_min=0.0, clip_value_max=255.0))

  #print("new")
  #print(output_image)

  # Show currently obtained image
  print(output_image.numpy().max())
  
  #output_image_resized = tf.image.resize(output_image[0], size=content_img.shape[:2], method='gaussian')
  output_image_resized = tf.image.resize(output_image[0], size=content_img.shape[:2], method='bilinear')
  #output_img_array = np.array(output_image*255, np.uint8)
  #output_img_array = np.array(output_image.value(), np.uint8)
  output_img_array = np.array(output_image_resized, np.uint8)
  #output_img_array = output_img_array.squeeze()  
  #imageio.imwrite(f"z:/test/{epoch}.jpg", output_img_array)
  output_image_placeholder.image(output_img_array, caption='Output image', use_column_width=True, clamp=True, channels='RGB')
  
progress_text.text("Done!")
progress_bar.empty()

