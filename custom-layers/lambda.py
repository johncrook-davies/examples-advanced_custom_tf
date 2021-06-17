import tensorflow as tf
from tensorflow.keras import backend as K

def my_relu(x):
    return K.maximum(-0.1, x)

if __name__ == '__main__':
    # Download and prepare the fashion mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define fit and evaluate networks with lambda layers
    ## ... Using the python lambda syntax
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128),
      tf.keras.layers.Lambda(lambda x: tf.abs(x)),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
    ## ... Using a custom defined 'relu' function
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(my_relu),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
