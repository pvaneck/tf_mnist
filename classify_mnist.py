#!/usr/bin/env python

import argparse
import os

import tensorflow as tf


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_GRAPH_FILE = os.path.join(
    SCRIPT_PATH, './saved-model/optimized_mnist_model.pb'
)


def read_image(file):
    """Read an image file and convert it into a 1-D floating point array."""
    file_content = tf.read_file(file)
    image = tf.image.decode_jpeg(file_content, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [1, 28*28])
    return image


def classify(args):
    """Classify a digit.

    This method will import the saved model from the given graph file, and will
    pass in the given image pixels as input for the classification. The top
    five predictions will be printed.
    """

    if not os.path.isfile(args.image):
        print('Error: Image %s not found.' % args.image)
        sys.exit(1)

    # Load graph and parse file.
    with tf.gfile.GFile(args.graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='mnist_basic',
            op_dict=None,
            producer_op_list=None
        )

    # Get relevant nodes.
    x = graph.get_tensor_by_name('mnist_basic/input:0')
    y = graph.get_tensor_by_name('mnist_basic/output:0')

    image = read_image(args.image)
    sess = tf.InteractiveSession()
    image_array = sess.run(image)

    with tf.Session(graph=graph) as graph_sess:
        predictions = graph_sess.run(y, feed_dict={x: image_array})
        prediction = predictions[0]

    # Get the indices that would sort the array, then only get the indices that
    # correspond to the top 5 predictions.

    sorted_indices = prediction.argsort()[::-1][:5]
    for index in sorted_indices:
        label = str(index)
        confidence = prediction[index]
        print('%s (confidence = %.5f)' % (label, confidence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str,
                        help='Image to pass to model for classification.')
    parser.add_argument('--graph-file', type=str, dest='graph_file',
                        default=DEFAULT_GRAPH_FILE,
                        help='The saved model graph file to use for '
                             'classification.')
    classify(parser.parse_args())
