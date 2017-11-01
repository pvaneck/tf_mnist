# TensorFlow MNIST

Show saving a simple model and using the saved model for inference.

## Train model

To train a basic Softmax regression model, do the following:

```
python ./train_basic_model.py
```

To run a one-convolutional layer model, do the following:

```
python ./train_cnn_model.py
```

When the script completes, the exported model we should use will
be saved, by default, as `./saved-model/optimized_mnist_model.pb`. This is
a [Protocol Buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) file
which represents a serialized version of our model with all the learned weights
and biases. This specific one is optimized for inference-only usage.

## Use model

Use the following script to classify an image of a digit. The image should be
a 28x28 image with the background black and foreground white.

```
python ./classify_mnist.py <img path>
```

Optional flags for this are:

* `--graph-file` for specifying your saved model file.
  Default is _./saved-model/optimized_mnist_model.pb_.

Several sample images to use for inference are in the _sample-images_ directory
of the project.
