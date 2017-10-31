import os

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import tensorflow as tf


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'mnist_model'
MODEL_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'saved-model')


def export_model(model_output_dir, input_node_names, output_node_name):
    """Export the model so we can use it later.

    This will create two Protocol Buffer files in the model output directory.
    These files represent a serialized version of our model with all the
    learned weights and biases. One of the ProtoBuf files is a version
    optimized for inference-only usage.
    """

    name_base = os.path.join(model_output_dir, MODEL_NAME)
    frozen_graph_file = os.path.join(model_output_dir,
                                     'frozen_' + MODEL_NAME + '.pb')
    freeze_graph.freeze_graph(
        name_base + '.pbtxt', None, False, name_base + '.chkp',
        output_node_name, "save/restore_all", "save/Const:0",
        frozen_graph_file, True, ""
    )

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    optimized_graph_file = os.path.join(model_output_dir,
                                        'optimized_' + MODEL_NAME + '.pb')
    with tf.gfile.FastGFile(optimized_graph_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Inference optimized graph saved at: " + optimized_graph_file)



def weight_variable(shape):
  """Generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name='weight')


def bias_variable(shape):
  """Generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name='bias')


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    input_node_name = 'input'
    output_node_name = 'output'

    # Placeholder that will be fed image data.
    x = tf.placeholder(tf.float32, [None, 784], name='input')
    # Placeholder that will be fed the correct labels.
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Convolutional layer - maps one grayscale image to 32 features.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = tf.nn.relu(x_conv1 + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    # After downsampling, our 28x28 image is now 14x14
    # with 32 feature maps.
    with tf.name_scope('flatten'):
        h_pool_flat = tf.reshape(h_pool1, [-1, 14*14*32])

    # Map the features to 10 classes, one for each digit
    with tf.name_scope('fc-classify'):
        W_fc2 = weight_variable([14*14*32, 10])
        b_fc2 = bias_variable([10])
        y = tf.matmul(h_pool_flat, W_fc2) + b_fc2

    tf.nn.softmax(y, name=output_node_name)
    ###########################

    # Define our loss.
    with tf.name_scope('loss'):
        # Use more numerically stable cross entropy.
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y),
            name='cross_entropy'
        )

    # Define our optimizer.
    with tf.name_scope('optimizer'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name='train_step')

    # Define accuracy.
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        correct_prediction = tf.cast(correct_prediction, tf.float32, name='correct_prediction')
        accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

    saver = tf.train.Saver()

    # Launch session.
    sess = tf.InteractiveSession()

    # Initialize variables.
    tf.global_variables_initializer().run()

    checkpoint_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME + '.chkp')
    # Save the graph definition to a file.
    tf.train.write_graph(sess.graph_def, MODEL_OUTPUT_DIR,
                         MODEL_NAME + '.pbtxt', True)

    # Do the training.
    for i in range(1100):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1]})
            print("Step %d, Training Accuracy %g" % (i, float(train_accuracy)))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    # Save a checkpoint after training has completed.
    saver.save(sess, checkpoint_file)

    # See how model did.
    print("Test Accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                             y_: mnist.test.labels}))

    # Export the model so we can use it later for inference.
    export_model(MODEL_OUTPUT_DIR, [input_node_name], output_node_name)
    sess.close()


if __name__ == '__main__':
    main()
