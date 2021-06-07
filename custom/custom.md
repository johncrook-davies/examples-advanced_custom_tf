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
For the examply of an MNIST or fasion MNIST
```py
  from tensorflow.keras.layers import Model
  func_model = Model(inputs=input, outputs=predictions)
```
