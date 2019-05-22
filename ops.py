import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

def make_parallel(fn, num_gpus, **kwargs):
    print("Making make_parallel for %d gpu(s)" % num_gpus)
    in_splits = {}
    for k, v in kwargs.items():
        if type(v) == list:
            in_splits[k] = zip(*[iter(v)]*( int( len(v) / num_gpus ) ))
        else:
            in_splits[k] = tf.split(v, num_gpus)

    out_split = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse= i!=0):# tf.AUTO_REUSE):
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    output_num = len(out_split[0])
    output = []
    for i in range(output_num):
        output.append([])
        for j in range(num_gpus):
            output[i].append(out_split[j][i])

    return output

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
          self.epsilon  = epsilon
          self.momentum = momentum
          self.name = name

    def __call__(self, x, train=True, reuse=False, trainable=True ):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      fused=True,
                      scale=True,
                      trainable=trainable,
                      reuse = reuse,
                      is_training=train,
                      scope=self.name)

def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(axis=3, values=[x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
           k_h=3, k_w=3, d_h=2, d_w=2, use_bias=True, stddev=0.02,
           name="conv2d", reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        if use_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), get_shape(conv))

        return conv

def deconv2d(input_, output_shape,
             kernel_size=(3,3), strides=(2,2), stddev=0.02, use_bias = True,
             name="deconv2d", with_w=False, reuse = False):
    '''
    with tf.variable_scope(name, reuse = reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
    '''
    if type(output_shape) == list():
        output_shape = output_shape[-1]
    return tf.layers.conv2d_transpose(input_, output_shape, kernel_size, strides, padding='SAME', data_format='channels_last', activation=None, use_bias=use_bias, 
                                        kernel_initializer=tf.random_normal_initializer(stddev=stddev), bias_initializer=tf.zeros_initializer(),
                                        trainable=True, name=name, reuse=reuse)

def maxpool2d(x, k=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

       
def prelu(x, name, reuse = False):
    shape = x.get_shape().as_list()[-1:]

    with tf.variable_scope(name, reuse = reuse):
        alphas = tf.get_variable('alpha', shape, tf.float32,
                            initializer=tf.constant_initializer(value=0.2))

        return tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5

def relu(x, name='relu'):
    return tf.nn.relu(x, name)  

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def elu(x, name='elu'):
  return tf.nn.elu(x, name)

def linear(input_, output_size, scope="Linear", reuse = False, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    print(shape)

    with tf.variable_scope(scope or "Linear", reuse = reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def triplet_loss(anchor_output, positive_output, negative_output, margin = 0.2 ):
    d_pos = tf.reduce_mean(tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_mean(tf.square(anchor_output - negative_output), 1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    
    return loss

def cosine_loss(anchor_output, positive_output):
    anchor_output_norm = tf.nn.l2_normalize(anchor_output, 1)
    positive_output_norm = tf.nn.l2_normalize(positive_output, 1)
    loss = 1 - tf.reduce_sum(tf.multiply(anchor_output_norm, positive_output_norm), 1)

    return loss

def cosine_triplet_loss(anchor_output, positive_output, negative_output, margin = 0.2 ):
    anchor_output_norm = tf.nn.l2_normalize(anchor_output, 1)
    positive_output_norm = tf.nn.l2_normalize(positive_output, 1)
    negative_output_norm = tf.nn.l2_normalize(negative_output, 1)

    sim_pos = tf.reduce_sum(tf.multiply(anchor_output_norm, positive_output_norm), 1)
    sim_neg = tf.reduce_sum(tf.multiply(anchor_output_norm, negative_output_norm), 1)

    loss = tf.maximum(0., margin - sim_pos + sim_neg)

    return loss

def norm_loss(predictions, labels, mask = None, loss_type = 'l1', reduce_mean = True, p=1):
    from tensorflow.python.ops import array_ops

    assert (loss_type in ['l1', 'l2', 'l2,1']), "Suporting loss type is ['l1', 'l2', 'l2,1']"

    diff = predictions - labels
    if mask != None:
        diff = tf.multiply(diff, mask)

    inputs_shape = array_ops.shape(diff)

    if loss_type == 'l1':
        loss = tf.reduce_sum( tf.abs(diff) )

    elif loss_type == 'l2':
        loss = tf.nn.l2_loss(diff)

    elif loss_type == 'l2,1':
        #inputs_rank = tf.cast(labels.get_shape().ndims, tf.int32)
        #spatial_dims = array_ops.slice(inputs_shape, [1], [2])
        #batch_dim = array_ops.slice(inputs_shape, [0], [1])

        loss = tf.sqrt( tf.reduce_sum ( tf.square (diff) + 1e-16, axis = [-1] ) )
        if p!= 1:
            loss = tf.pow(loss, p)
        loss = tf.reduce_sum(loss)

    if reduce_mean:
        numel = tf.cast(tf.reduce_prod(inputs_shape), diff.dtype)
        loss = tf.div(loss, numel)

    return loss
        
def total_variation(images, mask, name=None):

    ndims = images.get_shape().ndims
    if ndims == 3:
        # The input is a single image with shape [height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
        pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]

        # Sum for all axis. (None is an alias for all axis.)
        sum_axis = None
    elif ndims == 4:
        # The input is a batch of images with shape:
        # [batch, height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

        # Only sum for the last 3 axis.
        # This results in a 1-D tensor with the total variation for each image.
        sum_axis = [1, 2, 3]
    else:
        raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    pixel_dif1 = tf.multiply(pixel_dif1, mask[:, 1:,  :, :])
    pixel_dif2 = tf.multiply(pixel_dif2, mask[:,  :, 1:, :])

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) +
               tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis))

    return tot_var 