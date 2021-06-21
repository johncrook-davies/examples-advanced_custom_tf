# Custom models
An example of where custom models might be useful is a 'deep and wide' network. A deep and wide model is where one one input layer is fed through a DNN and the other input is fed into the model further down the network.

The functional api can handle this architecture, it can also be ecapsulted in a custom model class. Using the class api has advantages:
* it keeps logic a bit cleaner
* it allows the definition of loops and multiple layers
* it allows if/then statements

Subclassing the tf model class allows:
* use of `.fit()`, `.evaluate()` and `.predict()`
* also allows override and customisation of these functions
* use of `.save()` to save the model and `.save_weights()` to save the model weights
* use of visual summaries such as `.summary()` and `tf.keras.util.plot_model()`

Using the sequanetial/functional APIs has disadvantages when looking at complex or exotic model architectures.
* only suited to models that don't loop back during training or inference
* not suited for dynamic networks

Model subclassing has the benefits of:
* being able to continue to use functional or sequential code within the class
* allows more modular architectures
* allows quick experimentation
* allows control of the flow of data  
