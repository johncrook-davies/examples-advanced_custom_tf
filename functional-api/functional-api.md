# Custom Models, Layers, and Loss Functions
## Functional API
Choosing this API over the more succinct sequential API gives greater flexibility. Layers are defined by functions instead of a list.
1. Define an input layer
For the examply of an MNIST or fasion MNIST:
```py
  from tensorflow.keras.layers import Input
  input = Input(shape=(28,28))
```
2. Define the layers connecting each layer using 'python functional syntax'
There is no list of layers like the typical sequential declaration. Each function is applied to the previous variable. Variables can be reused or newly defined.
For the examply of an MNIST or fasion MNIST:
```py
  from tensorflow.keras.layers import Dense, Flatten
  x = Flatten()(input)
  x= Dense(128, activation='relu')(x)
  predictions = Dense(10, activation='softmax')(x)
```
3. Define the model object and put it all in there
For the example of an MNIST or fashion MNIST
```py
  from tensorflow.keras.layers import Model
  func_model = Model(inputs=input, outputs=predictions)
```
## Branching models
The functional API allows us to define models with multiple branches - so not just one sequence. The 'Concatenate' function can then be used to recombine layers or alternatively the 'Model' function can be passed a list of inputs and can produce a list of outputs.
One example of a branching model is the 'Inception' model.

## Siamese networks
This architecture takes two different inputs and uses the same architecture to process the different inputs. The output from each side of the network is a vector representing the image. The Euclidean distance is then measured between the outputs and is an indicator of similarity between the images. The model is trained using pairs of inputs with a label specifying whether the inputs are similar or not. This architecture is used in a number of different areas of research.

Euclidean distance is the distance between the two vectors, in 3D space this is simply the calculated distance between two points in space. Generally it is the square root of the sum of squares.
