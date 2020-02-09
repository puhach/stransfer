# Stransfer

The project leverages the power of streamlit to present the concept of neural style transfer. The artistic style of one image is combined with the content captured from another image by using a convolutional neural network. 

<div align="center">
<img src="./assets/styled3.jpg">
</div>



### Algorithm outline


*1. Load content and style images*

The content image defines objects and shapes whereas the style image suggests colors and textures for the new image. It is common to see a famous artwork used as a style image.


*2. Load a pre-trained neural network*

VGG was used in the original paper, but here I experimented with other models, such as Inception and ResNet. 


*3. Freeze the layers of interest*

We need to choose which layers of the model we would like to extract the style and the content from. Then we freeze the weights in the selected layers. Thus, the model can be used as a fixed feature extractor. 


*4. Extract image features from different layers the model*

The original content and style images are passed through the selected layers of the model and transformed into feature maps that bear essential information about the images. These feature maps can be thought of as their content representation. 


*5. Retrieve the style representation*

At this step we are finding correlations between feature maps extracted from the style layers we selected. Mathematically it is done by computing the Gram matrices of each style layer.


*6. Create the output image*

To make style transfer quicker the output image is usually initialized with the original content image. It will later be passed through the selected layers of the pre-trained model to find its content and style representation in the same manner we did for the input images.  


*7. Calculate the loss*

We need to define the metrics of how close the content and the style of the output image are to the original content and style images. In this implementation I use squared difference as a measure of proximity. Each layer has a corresponding weight which determines how much it contributes into a style or content loss.

In addition to that a total variation loss is used to decrease high frequency artifacts produced by the original algorithm.

Lastly, the total loss is a weighted sum of the content, style, and variation losses.


*8. Update the output image*

Updating the output image involves finding the gradients of the loss with respect to the output image. These gradients show how the pixel values of the output image should be changed so as to minimize the loss. 

When output image is updated, we repeat the process passing it through the chosen layers of the model, retrieving its content and style representations, calculating the loss, finding the loss gradients, and altering the image again according to the gradients. We stop after a desired number of steps is taken.


# Prerequisites

- Python 3.7
- TensorFlow 2.0
- Streamlit 0.51
- Imageio 2.6

For convenience there is *environment.yml* file which you can create a conda environment from:
```
conda env create -f environment.yml
```
This will install TensorFlow optimized using Intel Math Kernel Library for Deep Neural Networks (Intel MKL-DNN).


# Usage

The application is fairly simple to use. To start with, choose the **Content image** and the **Style image** from the lists. They show files from the 'data/content' and 'data/style' directories (just in case you want to try your own image).

<div align="center">
<img src="./assets/octopus_1.jpg">
</div>

Notice the style transfer is already running. On the right a new image is being created and you can see how it is gradually changing to become closer to the style image. 

There are many options you may tweak:

**Steps** 

The number of times the image should be updated. The larger the value, the stronger the effect of style transfer.

Must be an integer in range from 1 to 10000. Default is 20.


**Model**

The pre-trained model used for feature extraction. Supported models are: 
- VGG16
- VGG19
- Inception V3
- Xception 
- DenseNet
- ResNet
- ResNet V2

Default is VGG16.

**Intermediate image size**

The resolution the input images should be resized to before they are passed to the pre-trained model of choice.

Must be in range from 100 to 1000. Default is 500.




# Credits

Original paper by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

[Neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer) from TensorFlow tutorials.

Udacity's [Neural Style Transfer Lesson](https://classroom.udacity.com/courses/ud188/lessons/c1541fd7-e6ec-4177-a5b1-c06f1ce09dd8/concepts/af086838-4309-4ec1-8fb9-446f148ad815).

The picture of the Stata Center used as a content image sample was taken by [Juan Paulo](https://juanpaulo.me/).