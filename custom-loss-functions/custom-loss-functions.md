# Custom loss functions
Using the loss object rather than the loss function string is useful because it enables us to pass parameters into the loss function.

Custom functions must:
* Accept two parameters,
  - y_true, the true values, and
  - y_pred, the predicted values

As an example, a custom loss function can be written for the Huber loss:
* for absolute difference (`a`) less than or equal to the threshold `delta`, the loss is `a**2/2`
* otherwise the loss is equal to the absolute value of `a` minus `delta/2`

```py
  def huber_loss(y_true,y_pred):
    delta = 1
    a = y_true - y_pred
    is_small_error = tf.abs(a) <= delta
    small_error_loss = tf.square(a) / 2
    big_error_loss = delta * (tf.abs(a) - .5 * delta)
    return tf.where(is_small_error, small_error_loss, big_error_loss)
```

This function can then be used in tf by passing the name as a string or reference into `model.compile`.
