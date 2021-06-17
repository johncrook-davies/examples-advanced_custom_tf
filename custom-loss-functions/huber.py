import tensorflow as tf
import numpy as np
from tensorflow import keras

# As a function
## With no hyperparameters
def huberloss_func(y_true,y_pred):
    delta = 1
    a = y_true - y_pred
    is_small_error = tf.abs(a) <= delta
    small_error_loss = tf.square(a) / 2
    big_error_loss = delta * (tf.abs(a) - .5 * delta)
    return tf.where(is_small_error, small_error_loss, big_error_loss)
## With hyperparameters
def huberloss_wrap(delta):
    def _huber_loss(y_true,y_pred):
      a = y_true - y_pred
      is_small_error = tf.abs(a) <= delta
      small_error_loss = tf.square(a) / 2
      big_error_loss = delta * (tf.abs(a) - .5 * delta)
      return tf.where(is_small_error, small_error_loss, big_error_loss)
    return _huber_loss
# As a class
from tensorflow.keras.losses import Loss
class HuberLossClass(Loss):
    delta = 1
    def __init__(self, delta):
      super().__init__()
      self.delta = delta
    def call(self,y_true,y_pred):
      a = y_true - y_pred
      is_small_error = tf.abs(a) <= self.delta
      small_error_loss = tf.square(a) / 2
      big_error_loss = self.delta * (tf.abs(a) - .5 * self.delta)
      return tf.where(is_small_error, small_error_loss, big_error_loss)

if __name__ == '__main__':
    # Dummy absolute basic inputs and labels with a y = 2x + 1 relationship
    xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # First train model on stock mean square error to test
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500,verbose=0)
    print(model.predict([10.0]))

    # Next train model on custom Huber loss for comparison using the function api
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss=huberloss_func)
    model.fit(xs, ys, epochs=500,verbose=0)
    print(model.predict([10.0]))

    # ... And the wrapper function api
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss=huberloss_wrap(delta=1))
    model.fit(xs, ys, epochs=500,verbose=0)
    print(model.predict([10.0]))

    # ... And the class api
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss=HuberLossClass(delta=1))
    model.fit(xs, ys, epochs=500,verbose=0)
    print(model.predict([10.0]))
