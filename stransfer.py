import tensorflow as tf
import tensorflow_hub as hub
# TODO: later use VGG19 and, probably, Resnet, Inception etc
from tensorflow.keras.applications import VGG16, VGG19
import imageio
from featureextractor import *
#from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
import random
import streamlit as st
import altair as alt
import pandas as pd
import glob
import os
import traceback 

print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)




def adjust_shape(image, size):  
  image_prep = tf.image.resize(image, size=size, method='lanczos5')  # resize appropriately 
  image_prep = image_prep[tf.newaxis, ..., :3]  # add the batch dimension and discard the alpha channel
  return image_prep


def preprocess_image(image, model_name):
  
  if model_name.lower() == "vgg16":
    return tf.keras.applications.vgg16.preprocess_input(image)
  elif model_name.lower() == "vgg19":
    return tf.keras.applications.vgg19.preprocess_input(image)
  else:
    raise Exception(f'Model "{model_name}" is not supported')



def get_layer_weights(conv_layers, chosen_layers, layer_type):

  chart_placeholder = st.sidebar.empty()

  max_layer_weight = 1.0

  # Initialize chosen layers with weights=1.0 and give the rest the weight of zero.
  # Random weights cause unexpected slider movements.
  layer_weight_func = lambda layer_name, chosen_layers : st.sidebar.slider(
      label=layer_name, min_value=0.0, max_value=max_layer_weight,
      value=1.0 if layer_name in chosen_layers else 0.0,
      step=0.01, key='slider_'+layer_name+'_'+layer_type)

  # Display layer weights in the chart.  
  # Not sure what is better: displaying all layers in the chart is more logical,
  # but then they are very tight to each other.

  data = pd.DataFrame.from_records(columns=['layer', 'weight'],
    data=[(name, layer_weight_func(name, chosen_layers)) for name in conv_layers])

  # Also works:

  #layer_weights = {
  #  name : layer_weight_func(name, chosen_layers)
  #  for name in conv_layers 
  #}

  #layer_weights = {
  #  name: value for (name, value) in 
  #  [(name, layer_weight_func(name, chosen_layers)) for name in conv_layers]
  #  if value > 0
  #}

  #data = pd.DataFrame.from_dict({ 'layer' : list(layer_weights.keys()),
  #                                'weight' : list(layer_weights.values())})

  
  chart = alt.Chart(data).mark_bar().encode(
    y='layer',
    x='weight'
  )

  chart_placeholder.altair_chart(altair_chart=chart)

  # Grouped Bar Chart
  #data = pd.DataFrame.from_dict({
  #                    'layer': list(content_weights.keys()) + list(style_weights.keys()), 
  #                    'weight': list(content_weights.values()) + list(style_weights.values()),
  #                    'type': ['C']*len(content_weights) + ['S']*len(style_weights)})

  #chart = alt.Chart(data).mark_bar().encode(
  #  row='layer',
  #  y='type',
  #  x='weight',
  #  color='type'
  #)


  #st.sidebar.altair_chart(altair_chart=chart, width=200)
  

  # Create layer weight map containing only layers with positive weight.

  #layer_weights = { k : v for (k,v) in layer_weights.items() if v > 0 }
  layer_weights = { row['layer'] : row['weight'] for index, row in data.iterrows() 
                    if row['weight'] > 0 }
  
  return layer_weights


try:

  # Initialize GUI

  st.title("Neural Style Transfer")

  st.sidebar.header("Settings")

  progress_text = st.empty()
  progress_text.text("Preparing...")

  
  # Select the content image

  content_path = st.sidebar.selectbox('Content image', options=glob.glob('data/content/*.*'),
                                      format_func=lambda glob_path: os.path.basename(glob_path))

  content_img = imageio.imread(content_path)

  st.sidebar.image(image=np.asarray(content_img), use_column_width=True, 
    caption=None, clamp=True, channels='RGB')


  # Select the style image

  style_path = st.sidebar.selectbox('Style image', options=glob.glob('data/style/*.*'),
                                    format_func=lambda glob_path: os.path.basename(glob_path))

  style_img = imageio.imread(style_path)

  st.sidebar.image(image=np.asarray(style_img), use_column_width=True, 
    caption=None, clamp=True, channels='RGB')

  #st.sidebar.image(image=[np.asarray(content_img), np.asarray(style_img)], use_column_width=True, 
  #  caption=['Content image', 'Style image'], clamp=True, channels='RGB')


  # Specify the number of steps.
  steps = st.sidebar.number_input(label='Steps', min_value=1, max_value=10000, value=20, step=1)

  # Choose the model.
  model_name = st.sidebar.selectbox(label='Model', options=['VGG16', 'VGG19'], index=0)

  # Specify the resolution the input images should be resized to before they are passed to VGG network.
  size = st.sidebar.slider( label='Intermediate image size', min_value=100, max_value=1000, value=500, 
                            step=1, format='%d')


  # Resize the images and add the batch dimension.
  content_resized = adjust_shape(content_img, (size, size))
  style_resized = adjust_shape(style_img, (size, size))

  # Preprocess the images.
  content_prep = preprocess_image(content_resized, model_name)
  style_prep = preprocess_image(style_resized, model_name) 

  # TODO: not sure if they make any tangible difference. Probably, they can be removed.
  # Overall content weight (alpha) and style weight (beta).
  alpha = st.sidebar.slider(label='Content reconstruction weight (alpha)', min_value=1, max_value=10000, value=1)
  beta = st.sidebar.slider(label='Style reconstruction weight (beta)', min_value=1, max_value=10000, value=1000)

  # Instantiate the VGG network.
  vgg = load_model(model_name)


  # Extract convolutional layers from the model

  conv_layers = extract_conv_layers(vgg)

  assert len(conv_layers) > 0, "The model has no convolutional layers."


  # Set weights for each content layer
  
  st.sidebar.subheader("Content layer weights")
  #content_layer_weights = get_layer_weights(conv_layers, conv_layers[:1], 'content')
  content_layer_weights = get_layer_weights(conv_layers, conv_layers[-1:], 'content')

  # Set weights for each style layer. Weighting earlier layers more will result in larger style artifacts.

  st.sidebar.subheader("Style layer weights")
  style_layer_weights = get_layer_weights(conv_layers, conv_layers[:1], 'style')
  
  assert len(content_layer_weights) > 0 and len(style_layer_weights) > 0, \
    f"At least one content layer and one style layer must have a positive weight."


  # Content and style layers with non-zero weight comprise the layers of interest,
  # which we use to get the intermediate model outputs from.

  layers_of_interest = list(set(content_layer_weights).union(style_layer_weights))

  feature_extractor = create_feature_extractor(vgg, layers_of_interest)


  # Get content and style features only once before training.
  # TODO: Perhaps, try tf.constant() here
  input_content_features = feature_extractor(content_prep)
  input_style_features = feature_extractor(style_prep)


  # Map content layers to the features extracted from these layers.
  content_targets = build_content_layer_map(input_content_features, content_layer_weights.keys())

  # Map style layers to the gram matrices calculated for each layer of our style representation.
  style_targets = build_style_layer_map(input_style_features, style_layer_weights.keys())
  


  # TODO: Consider wrapping TensorFlow stuff into a function to release memory:
  # https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution

  # Create a third output image and prepare it for change.
  # To make this quick, start off with a copy of our content image, then iteratively change its style.
  # For TF docs: GradientTape records operations if they are executed within its context manager and at least one
  # of their inputs is being "watched". Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, 
  # where trainable=True is default in both cases) are automatically watched. Tensors can be manually watched 
  # by invoking the watch method on the GradientTape context manager.
  #output_image = tf.Variable(content_resized / 255.0)
  output_image = tf.Variable(content_resized)
  #print(output_image.numpy().min(), output_image.numpy().max())

  # TODO: add a slider to tweak the optimizer
  #optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
  optimizer = tf.optimizers.Adam(learning_rate=0.8)


  # TODO: it might be interesting to create a video of images obtained after each epoch to see 
  # how style transfer is progressing.

  progress_bar = st.progress(0)
  output_image_placeholder = st.empty()

  #epochs = 20

  #for epoch in range(1, epochs+1):
  for step in range(1, steps+1):

    # Report progress
    progress_text.text(f"Step {step}/{steps}")
    progress_bar.progress(step/steps)
    print(f"Step {step}")

    with tf.GradientTape() as tape: # Record operations for automatic differentiation

      # Preprocess the output image before we pass it to VGG
      output_prep = preprocess_image(output_image, model_name)
      #output_prep = preprocess_image(output_image*255)

      # Extract content and style features from the output image.
      output_features = feature_extractor(output_prep)
      #output_features = feature_extractor(output_image)
      output_content_map = build_content_layer_map(output_features, content_layer_weights.keys())
      output_style_map = build_style_layer_map(output_features, style_layer_weights.keys())


      # Calculate the content loss
      content_loss = tf.add_n([content_layer_weight * tf.reduce_mean(
                              (output_content_map[content_layer_name] - content_targets[content_layer_name])**2) 
                              for content_layer_name, content_layer_weight in content_layer_weights.items()
                              if content_layer_weight > 0 ]) 

      # Calculate the style loss
      style_loss = tf.add_n([style_layer_weight * tf.reduce_mean(
                            (output_style_map[style_layer_name] - style_targets[style_layer_name])**2 ) 
                            for style_layer_name, style_layer_weight in style_layer_weights.items()
                            if style_layer_weight > 0]) 

      # TODO: try to use the total variation loss to reduce high frequency artifacts

      # Add up the content and style losses
      total_loss = alpha*content_loss + beta * style_loss

    # Calculate loss gradients
    grads = tape.gradient(total_loss, output_image)  

    #print("before gradients:")
    #print(output_image.numpy().min(), output_image.numpy().max())

    
    # Apply the gradients to alter the output image 
    optimizer.apply_gradients([(grads, output_image)])

    #print("grads:")
    #print(grads)

    #print("after gradients:")
    #print(output_image.numpy().min(), output_image.numpy().max())

    # Keep the pixel values between 0 and 255
    #output_image.assign(tf.clip_by_value(output_image, clip_value_min=0.0, clip_value_max=1.0))
    output_image.assign(tf.clip_by_value(output_image, clip_value_min=0.0, clip_value_max=255.0))

    #print("new")
    #print(output_image)

    ## Show currently obtained image
    #print(output_image.numpy().max())
    
    #output_image_resized = tf.image.resize(output_image[0], size=content_img.shape[:2], method='gaussian')
    output_image_resized = tf.image.resize(output_image[0], size=content_img.shape[:2], method='bilinear')
    #output_img_array = np.array(output_image*255, np.uint8)
    #output_img_array = np.array(output_image.value(), np.uint8)
    output_img_array = np.array(output_image_resized, np.uint8)
    output_image_placeholder.image(output_img_array, caption='Output image', use_column_width=True, clamp=True, channels='RGB')
    
  progress_text.text("Done!")
  progress_bar.empty()
  
except Exception as e:
  progress_text.text(e)
  traceback.print_exc()