# Advanced techniques with TensorFlow - course notes

This repo contains my notes and coding projects from the advanced techniques with TensorFlow course.

The repo is split by course week.

## Initial setup
### Install the required NVIDIA software on Ubuntu
If using tensorflow for the first time then set up using the instructions here `https://www.tensorflow.org/install/gpu`

## Sections
1. The tensorflow functional API - fuctional-api
a. functional-api.py - example of the functional api versus the standard tensorflow api
b. branching.py - example of a deep neural network DNN that branches producing outputs in one branch and carrying out additional processing in another branch
c. siamese.py - example of a siamese model that takes two inputs, processes them using the same DNN and then compares the resulting vectors
2. Custom loss functions - custom-loss-functions
a. huber.py - an example of implementation of a custom loss function of the Huber loss
3. Custom layers - custom-layers
a. lambda.py - example implementations of a lambda layer, using python lambda syntax and a predefined formula
b. custom-layers.py - example implementation of a fully custom layer
4. Custom models - custom-models
a. wideanddeep.py - example implementations of a wide and deep model using both the functional api and a subclassed approach
b. resnet.py - example implementation of a resnet using subclassed approach that defined a custom block and reuses it
