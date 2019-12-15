import tensorflow as tf
import tensorflow_hub as hub
# TODO: later use VGG19 and, probably, Resnet, Inception etc
from tensorflow.keras.applications import VGG16
import imageio
from featureextractor import create_feature_extractor, build_content_layer_map, build_style_layer_map
#from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
import streamlit as st


print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)




#def preprocess_image(image):
def adjust_shape(image):
  #image_prep = tf.keras.applications.vgg16.preprocess_input(image)
  image_prep = tf.image.resize(image, size=(224, 224))
  image_prep = image_prep[tf.newaxis, :]
  return image_prep


def preprocess_image(image):
  return tf.keras.applications.vgg16.preprocess_input(image)



st.title("Neural Style Transfer")

st.sidebar.header("Settings")


content_img = imageio.imread('data/content/chicago.jpg')
style_img = imageio.imread('data/style/wave.jpg')

# Resize the images and add the batch dimension
content_resized = adjust_shape(content_img)
style_resized = adjust_shape(style_img)

#content_img_st = content_img[:]
#content_img_st = np.transpose(np.asarray(content_img), (1, 0, 2))
st.image(image=[np.asarray(content_img), np.asarray(style_img)], use_column_width=True, 
  caption=['Content image', 'Style image'], clamp=True, channels='RGB')

# Preprocess the images
content_prep = preprocess_image(content_resized)
style_prep = preprocess_image(style_resized)


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


# Get content and style features only once before training.
# TODO: Perhaps, try tf.constant() here
input_content_features = feature_extractor(content_prep)

input_style_features = feature_extractor(style_prep)

# map content layers to the features extracted from these layers
content_targets = build_content_layer_map(input_content_features, content_layers)

#for content_layer_name, content_layer_features in content_targets.items():
#  print(content_layer_name)
#  print(content_layer_features.shape)

# calculate the gram matrices for each layer of our style representation
style_targets = build_style_layer_map(input_style_features, style_layers)

#for style_target_name, style_target_gram in style_targets.items():
#  print(style_target_name)
#  print(style_target_gram.shape)



# Weights for each style layer. Weighting earlier layers more will result in larger style artifacts.
# TODO: later try to combine it with style_layers
style_layer_weights = { 'block1_conv1': 1.,
                        'block2_conv1': 0.75,
                        'block3_conv1': 0.2,
                        'block4_conv1': 0.2,
                        'block5_conv1': 0.2
                      }

assert len(style_layer_weights) == len(style_layers), "Style layer weights mismatch the style layer names"

# Just like in the paper, we define an alpha (content_weight) and a beta (style_weight). This ratio will affect 
# how stylized the final image is.
# TODO: perhaps, we could get by style layer weights and, similarly, content layer weights
content_weight = 1  # alpha
style_weight = 1e3  # beta



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

epochs = 2

for epoch in range(1, epochs+1):

  print(f"Epoch {epoch}")

  with tf.GradientTape() as tape: # Record operations for automatic differentiation

    # Preprocess the output image before we pass it to VGG
    output_prep = preprocess_image(output_image)
    #output_prep = preprocess_image(output_image*255)

    # Extract content and style features from the output image
    output_features = feature_extractor(output_prep)
    #output_features = feature_extractor(output_image)
    output_content_map = build_content_layer_map(output_features, content_layers)
    output_style_map = build_style_layer_map(output_features, style_layers)


    # Calculate the content loss
    content_loss = tf.add_n( [tf.reduce_mean((output_content_map[content_layer] - content_targets[content_layer])**2) 
                              for content_layer in content_layers ]) 

    # Calculate the style loss
    style_loss = tf.add_n([style_layer_weights[style_layer] * tf.reduce_mean(
                          (output_style_map[style_layer] - style_targets[style_layer])**2 ) 
                          for style_layer in style_layers]) 

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
  #output_img_array = np.array(output_image*255, np.uint8)
  output_img_array = np.array(output_image.value(), np.uint8)
  output_img_array = output_img_array.squeeze()
  imageio.imwrite(f"z:/test/{epoch}.jpg", output_img_array)  