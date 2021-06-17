# Custom layers

## What is a layer?
A layer is a class that collects parameters and encapsulates state and computation to achieve it's purpose in a network. State here is the variables - variables can be trainable or non-trainable - when the variables are trainable, tf can tweak the values to achieve the best fit, the variables don't have to be trainable however. Computation is the means of transforming a batch of inputs into a batch of outputs, or the 'forward pass'.

## Simple custom layers
Simplest way to create a custom layer is to create a lambda layer. These layers are cable of executing arbitrary code.

Within the lambda layer you can call a python lambda function or a predefined custom function.

## Advanced, trainable, layers
For more advanced layer that are 'trainable', full custom layers must be created. To create a custom layer, it must:
* be a class
* inherit from `Layer`
* include the methods
  - `__init__` - initialising the class, sets the parameters and initialised the class variables. It must have a parameter called units, which specifies the number of units in the layer.
  - `build` - runs when instance is created, this specifies local input states plus anything else needed for their creation. tf supports a number of methods of initialising variable states eg. Randomly sampled from normal distribution. Variables are stored on self as a tensor using
  ```py
    tf.Variable(name='foo', initial_value=self.foo(shape=(x, self.units), trainable=True)
  ```
  - `call` - performs computation and is called during training to get the output
