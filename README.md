# TensorFlow MNIST

Show saving a simple model and using the saved model for inference.

## Train model

```
python ./train_model.py
```

When the script completes, the exported model we should use will
be saved, by default, as `./saved-model/optimized_mnist_model.pb`. This is
a [Protocol Buffer](https://en.wikipedia.org/wiki/Protocol_Buffers) file
which represents a serialized version of our model with all the learned weights
and biases. This specific one is optimized for inference-only usage.

## Use model

```
python ./classify_mnist.py <img path>
```

Optional flags for this are:

* `--graph-file` for specifying your saved model file.
  Default is _./saved-model/optimized_mnist_model.pb_.

Several sample images to use for inference are in the _sample-images_ directory
of the project.
