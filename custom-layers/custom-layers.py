import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class SimpleDense(Layer):

    def __init__(self, units=32, activation=None): # When None, a linear activation function is used
    #Use this version if no activation funtion is needed
    #def __init__(self, units=32):
        '''Initializes the instance attributes'''
        super(SimpleDense, self).__init__()
        self.units = units
        # Comment following line out if no activation function is needed
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            name="kernel",
            initial_value=w_init(
                shape=(input_shape[-1],
                self.units),
                dtype='float32'
            ),
            trainable=True
        )

        # initialize the biases
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            name="bias",
            initial_value=b_init(
                shape=(self.units,),
                dtype='float32'
            ),
            trainable=True
        )

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        # Use this version if no activation function is needed
        #return tf.matmul(inputs, self.w) + self.b
        return self.activation(tf.matmul(inputs,self.w) + self.b)

if __name__ == '__main__':
    # Dummy absolute basic inputs and labels with a y = 2x + 1 relationship
    xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # use the Sequential API to build a model with our custom layer
    my_layer = SimpleDense(units=1, activation='relu')
    model = tf.keras.Sequential([my_layer])

    # configure and train the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500,verbose=0)

    # perform inference
    print(model.predict([10.0]))

    # see the updated state of the variables
    print(my_layer.variables)
