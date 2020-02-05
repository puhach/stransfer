import tensorflow as tf
import imageio
import numpy as np
import random
import streamlit as st
import altair as alt
import pandas as pd
import glob
import os
import traceback 

print("TensorFlow version:", tf.__version__)
#print("TensorFlow Hub version:", hub.__version__)




class StyleTransfer:
  """ 
  Implements neural style transfer. It takes two images and by means of a pretrained model extracts their 
  features to create a new image combining the content and the style of the input images.
  """

  def __init__(self, model_name):

    # This line destroys the current TensorFlow graph preventing new layer names being generated for 
    # Inception_V3 model layers.
    tf.keras.backend.clear_session()

    # Instantiate the neural network.
    if model_name is not None:
      self.load_model(model_name)

  
  def load_model(self, name, weights='imagenet'):
    """
    Loads a pre-trained model to be used for feature extraction.
    :param name: Name of the model (VGG16, VGG19, Inception_v3, DenseNet, ResNet, ResNet_v2).
    :param weights: Model weights. Must be 'imagenet' (pre-training on ImageNet) or None (random initialization).
    """
    # load a pretrained model and set up the image preprocessing function
    if name.lower() == "vgg16":
      self.model = tf.keras.applications.VGG16(include_top=False, weights=weights)
      self.preprocess_image = tf.keras.applications.vgg16.preprocess_input
    elif name.lower() == "vgg19":
      self.model = tf.keras.applications.VGG19(include_top=False, weights=weights)
      self.preprocess_image = tf.keras.applications.vgg19.preprocess_input
    elif name.lower() == "inception_v3":
      self.model = tf.keras.applications.InceptionV3(include_top=False, weights=weights)
      self.preprocess_image = tf.keras.applications.inception_v3.preprocess_input    
    elif name.lower() == "xception":
      self.model = tf.keras.applications.Xception(include_top=False, weights=weights)
      self.preprocess_image = tf.keras.applications.xception.preprocess_input
    elif name.lower() == "densenet":
      self.model = tf.keras.applications.DenseNet121(include_top=False, weights=weights)
      self.preprocess_image = tf.keras.applications.densenet.preprocess_input
    elif name.lower() == "resnet":
      self.model = tf.keras.applications.ResNet50(include_top=False, weights=weights)
      self.preprocess_image = tf.keras.applications.resnet.preprocess_input
    elif name.lower() == "resnet_v2":
      self.model = tf.keras.applications.ResNet50V2(include_top=False, weights=weights)
      self.preprocess_image = tf.keras.applications.resnet_v2.preprocess_input
    else:
      raise Exception(f'Model "{name}" is not supported.')

    self.model.trainable = False
    self.model_name = name

    # Extract convolutional layers from the model
    self.conv_layers = [layer.name for layer in self.model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

    return self.model, self.conv_layers


  def get_conv_layers(self):
    """ 
    Returns convolutional layers of the model loaded by the load_model() method.
    """
    return self.conv_layers


  def __call__(self, content_img, style_img, 
              steps, size, content_layer_weights, style_layer_weights, 
              content_reconstruction_weight, style_reconstruction_weight, total_variation_weight, optimizer):
    """
    Takes the content and style images to perform style transfer using the specified number of steps, 
    content and style weights, image size, and the optimizer.
    :param content_img: The input image to capture content representation from.
    :param style_img: The iput image to capture style from.
    :param steps: The number of steps to perform style transfer for.
    :param size: The square size to which input images must be resized.
    :param content_layer_weights: The dictionary with layer names and corresponding weights defining 
              the importance of a particular layer for content representation of the output image.
    :param style_layer_weights: The dictionary with layer names and corresponding weights defining
              how much the resulting image style depends on the output of a particular layer.
    :param content_reconstruction_weight: The weight factor defining how much content representation 
              is important as compared to the style.
    :param style_reconstruction_weight: The weight factor defining how much style is important as
              compared to the content representation.
    :param total_variation_weight: The weight factor determining how much to decrease high frequency 
              artifacts.
    :param optimizer: The optimizer to perform style transfer optimization.
    """

    self.content_reconstruction_weight = content_reconstruction_weight
    self.style_reconstruction_weight = style_reconstruction_weight
    self.total_variation_weight = total_variation_weight
    self.content_layer_weights = content_layer_weights
    self.style_layer_weights = style_layer_weights
    self.optimizer = optimizer

    # Resize the images and add the batch dimension.
    content_resized = StyleTransfer.adjust_shape(content_img, (size, size))
    style_resized = StyleTransfer.adjust_shape(style_img, (size, size))

    # Preprocess the images.    
    content_prep = self.preprocess_image(content_resized)
    style_prep = self.preprocess_image(style_resized)
    #content_prep = StyleTransfer.preprocess_image(content_resized, self.model_name)
    #style_prep = StyleTransfer.preprocess_image(style_resized, self.model_name) 


    # Content and style layers with non-zero weight comprise the layers of interest,
    # which we use to get the intermediate model outputs from.

    layers_of_interest = list(set(content_layer_weights).union(style_layer_weights))

    # Create feature extractor.
    #feature_extractor = create_feature_extractor(vgg, layers_of_interest)
    outputs = {layer_name : self.model.get_layer(layer_name).output for layer_name in layers_of_interest}
    self.feature_extractor = tf.keras.Model(self.model.inputs, outputs)


    # Get content and style features only once before training.
    # Not sure whether tf.constant() is important here.
    input_content_features = self.feature_extractor(tf.constant(content_prep))
    input_style_features = self.feature_extractor(tf.constant(style_prep))


    # Map content layers to the features extracted from these layers.
    self.content_targets = StyleTransfer.build_content_layer_map(input_content_features, content_layer_weights.keys())

    # Map style layers to the gram matrices calculated for each layer of our style representation.
    self.style_targets = StyleTransfer.build_style_layer_map(input_style_features, style_layer_weights.keys())
    

    # Create a third output image and prepare it for change.
    # To make this quick, start off with a copy of our content image, then iteratively change its style.
    # For TF docs: GradientTape records operations if they are executed within its context manager and at least one
    # of their inputs is being "watched". Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, 
    # where trainable=True is default in both cases) are automatically watched. Tensors can be manually watched 
    # by invoking the watch method on the GradientTape context manager.
    #output_image = tf.Variable(content_resized / 255.0)
    output_image = tf.Variable(content_resized)

    for step in range(1, steps+1):
      
      self.__step(output_image)

      #output_image_resized = tf.image.resize(output_image[0], size=content_img.shape[:2], method='gaussian')
      output_image_resized = tf.image.resize(output_image[0], size=content_img.shape[:2], method='bilinear')
      #output_img_array = np.array(output_image*255, np.uint8)
      #output_img_array = np.array(output_image.value(), np.uint8)
      output_img_array = np.array(output_image_resized, np.uint8)
      
      yield step, output_img_array


  # According to TF docs, when a function is decorated with tf.function, it can be called like any other function, 
  # but it will be compiled into a graph, which means we get the benefits of faster execution, running on GPU or TPU, 
  # or exporting to SavedModel.
  # https://www.tensorflow.org/guide/function
  # Later it can be a part of StyleTransfer class.
  @tf.function 
  def __step(self, output_image):
    """
    Performs one step of style transfer.
    :param output_image: The image updated during style transfer optimization.
    """

    with tf.GradientTape() as tape: # Record operations for automatic differentiation

      # Preprocess the output image before we pass it to the model.
      # Inception models seem to fail to preprocess images as passed in as variables,
      # therefore value() method is used.
      #output_prep = preprocess_image(output_image, model_name)
      #output_prep = StyleTransfer.preprocess_image(output_image.value(), model_name)
      output_prep = self.preprocess_image(output_image.value())
      #output_prep = preprocess_image(output_image*255)

      # Extract content and style features from the output image.
      output_features = self.feature_extractor(output_prep)
      #output_features = feature_extractor(output_image)
      output_content_map = StyleTransfer.build_content_layer_map(output_features, content_layer_weights.keys())
      output_style_map = StyleTransfer.build_style_layer_map(output_features, style_layer_weights.keys())


      # Calculate the content loss
      content_loss = tf.add_n([content_layer_weight * tf.reduce_mean(
                              (output_content_map[content_layer_name] - self.content_targets[content_layer_name])**2) 
                              for content_layer_name, content_layer_weight in self.content_layer_weights.items()
                              if content_layer_weight > 0 ]) 

      # Calculate the style loss
      style_loss = tf.add_n([style_layer_weight * tf.reduce_mean(
                            (output_style_map[style_layer_name] - self.style_targets[style_layer_name])**2 ) 
                            for style_layer_name, style_layer_weight in self.style_layer_weights.items()
                            if style_layer_weight > 0]) 

      # Add up the content and style losses
      total_loss = self.content_reconstruction_weight*content_loss + self.style_reconstruction_weight*style_loss 

      # Use the total variation loss to reduce high frequency artifacts
      if self.total_variation_weight > 0:

        # Find horizontal and vertical variations and try to minimize them
        # (output_image has a format of [B, H, W, C])
        x_var = output_image[:, :, 1:, :] - output_image[:, :, 0:-1, :]
        y_var = output_image[:, 1:, :, :] - output_image[:, 0:-1, :, :]

        variation_loss = tf.reduce_sum(tf.abs(x_var)) + tf.reduce_sum(tf.abs(y_var))

        total_loss += self.total_variation_weight * variation_loss
        

    # Calculate loss gradients
    grads = tape.gradient(total_loss, output_image)  

    # Apply the gradients to alter the output image 
    optimizer.apply_gradients([(grads, output_image)])

    # Keep the pixel values between 0 and 255
    #output_image.assign(tf.clip_by_value(output_image, clip_value_min=0.0, clip_value_max=1.0))
    output_image.assign(tf.clip_by_value(output_image, clip_value_min=0.0, clip_value_max=255.0))



  @staticmethod
  def adjust_shape(image, size):  
    """

    :param image: 
    :param size: 

    """
    image_prep = tf.image.resize(image, size=size, method='lanczos5')  # resize appropriately 
    image_prep = image_prep[tf.newaxis, ..., :3]  # add the batch dimension and discard the alpha channel
    return image_prep

  
  # @staticmethod decorator doesn't seem to make a big difference, unless we want to call this method on an instance.
  # Without a decorator it also produces a pylint warning.
  @staticmethod
  def compute_gram_matrix(layer_features):
    """

    :param layer_features: 

    """

    # Get the batch_size, depth, height, and width of the Tensor
    b, h, w, d = layer_features.shape

    assert b==1, "The function expects features extracted from a single image."

    # Reshape so we're multiplying the features for each channel
    tensor = tf.reshape(layer_features, [h*w, d])
    
    # Calculate the gram matrix
    gram = tf.matmul(tensor, tensor, transpose_a=True)

    return gram

  @staticmethod
  def build_content_layer_map(features, content_layers):
    """

    :param features: 
    :param content_layers: 

    """
    
    # TODO: describe expected features shape: (1, H, W, C) ?

    #content_map = { layer_name : layer_feats for 
    #                layer_name, layer_feats in 
    #                zip(content_layers, features[:len(content_layers)]) }

    content_map = { layer_name : features[layer_name] for layer_name in content_layers }
    #content_map = dict(filter(lambda f: f[0] in content_layers, features))

    return content_map

  @staticmethod
  def build_style_layer_map(features, style_layers):
    """

    :param features: 
    :param style_layers: 

    """
    
    # TODO: describe expected features shape: (1, H, W, C) ?

    # Each layer's Gram matrix is divided by height*width of the feature map. It makes easier to calculate 
    # the style loss afterwards:
    # https://github.com/udacity/deep-learning-v2-pytorch/issues/174
    
    gram_norm = lambda f: StyleTransfer.compute_gram_matrix(f) / (f.shape[1]*f.shape[2])
    style_map = { layer_name : gram_norm(features[layer_name])
                  for layer_name in style_layers
                }
                
    #style_map = { layer_name: compute_gram_matrix(layer_feats)/(layer_feats.shape[1]*layer_feats.shape[2]) for 
    #              layer_name, layer_feats in zip(style_layers, features[len(features)-len(style_layers):])} 

    return style_map





def get_layer_weights(conv_layers, chosen_layers, layer_type):
  """

  :param conv_layers: 
  :param chosen_layers: 
  :param layer_type: 

  """

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
  model_name = st.sidebar.selectbox(label='Model', 
    options=['VGG16', 'VGG19', 'Inception_V3', 'Xception', 'DenseNet', 'ResNet', 'ResNet_V2'], 
    index=0)


  # Specify the resolution the input images should be resized to before they are passed to VGG network.
  size = st.sidebar.slider( label='Intermediate image size', min_value=100, max_value=1000, value=500, 
                            step=1, format='%d')
  


  # Overall content weight (alpha) and style weight (beta).
  content_reconstruction_weight = st.sidebar.slider(label='Content reconstruction weight (alpha)', min_value=1, max_value=10000, value=1)
  style_reconstruction_weight = st.sidebar.slider(label='Style reconstruction weight (beta)', min_value=1, max_value=10000, value=1000)

  # A regularization term on the high frequency components of the image
  total_variation_weight = st.sidebar.slider(label='Total variation weight', min_value=0, max_value=100, value=12)


  style_transfer = StyleTransfer(model_name)

  # Extract convolutional layers from the model
  conv_layers = style_transfer.get_conv_layers()
  assert len(conv_layers) > 0, "The model has no convolutional layers."


  # Set weights for each content layer.
  
  st.sidebar.subheader("Content layer weights")
  content_layer_weights = get_layer_weights(conv_layers, conv_layers[-1:], 'content')

  # Set weights for each style layer. Weighting earlier layers more will result in larger style artifacts.

  st.sidebar.subheader("Style layer weights")
  style_layer_weights = get_layer_weights(conv_layers, conv_layers[:1], 'style')
  
  assert len(content_layer_weights) > 0 and len(style_layer_weights) > 0, \
    f"At least one content layer and one style layer must have a positive weight."


  
  # TODO: add a slider to tweak the optimizer
  #optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
  optimizer = tf.optimizers.Adam(learning_rate=0.8)


  # TODO: it might be interesting to create a video of images obtained after each epoch to see 
  # how style transfer is progressing.

  progress_bar = st.progress(0)
  output_image_placeholder = st.empty()

  for step, output_image in style_transfer(content_img, style_img, steps, size, 
                                          content_layer_weights, style_layer_weights, 
                                          content_reconstruction_weight, style_reconstruction_weight, total_variation_weight, optimizer):
    # Report progress
    progress_text.text(f"Step {step}/{steps}")
    progress_bar.progress(step/steps)
    print(f"Step {step}")

    output_image_placeholder.image(output_image, caption='Output image', use_column_width=True, clamp=True, channels='RGB')
    
  progress_text.text("Done!")
  progress_bar.empty()
  
except Exception as e:
  progress_text.text(e)
  traceback.print_exc()