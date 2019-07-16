import os
import scipy.misc
import numpy as np

from model_non_linear_3DMM import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 5000000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 64, "The size of batch samples images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 224, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_boolean("is_with_y", True, "True for with lable")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("samples_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_reduce", False, "True for 6k verteices, False for 50k vertices")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("gf_dim", 32, "")
flags.DEFINE_integer("gfc_dim", 512, "")
flags.DEFINE_integer("df_dim", 32, "")
flags.DEFINE_integer("dfc_dim", 512, "")
flags.DEFINE_integer("z_dim", 50, "")
flags.DEFINE_string("gpu", "1,2", "GPU to use [0]")

flags.DEFINE_boolean("is_pretrain", False, "Is in pretrain stage [False]")

flags.DEFINE_boolean("is_using_landmark", False, "Using landmark loss [False]")
flags.DEFINE_boolean("is_using_symetry", False, "Using symetry loss [False]")
flags.DEFINE_boolean("is_using_recon", False, "Using rescontruction loss [False]")
flags.DEFINE_boolean("is_using_frecon", False, "Using feature rescontruction loss [False]")
flags.DEFINE_boolean("is_using_graddiff", False, "Using gradient difference [False]")
flags.DEFINE_boolean("is_gt_m", False, "Using gt m [False]")
flags.DEFINE_boolean("is_partbase_albedo", False, "Using part based albedo decoder [False]")
flags.DEFINE_boolean("is_using_linear", False, "Using linear model supervision [False]")
flags.DEFINE_boolean("is_batchwise_white_shading", False, "Using batchwise white shading constraint [False]")
flags.DEFINE_boolean("is_const_albedo", False, "Using batchwise const albedo constraint [False]")
flags.DEFINE_boolean("is_const_local_albedo", False, "Using batchwise const albedo constraint [False]")
flags.DEFINE_boolean("is_smoothness", False, "Using pairwise loss [False]")


FLAGS = flags.FLAGS


def main(_):
    #pp.pprint(FLAGS.__flags)
    pp.pprint(tf.app.flags.FLAGS.flag_values_dict())


    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.samples_dir):
        os.makedirs(FLAGS.samples_dir)

    gpu_options = tf.GPUOptions(visible_device_list =FLAGS.gpu, per_process_gpu_memory_fraction = 0.8, allow_growth = True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:
        dcgan = DCGAN(sess, FLAGS)
            
        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)
            dcgan.test(FLAGS, True)
        '''
        if FLAGS.visualize:
            to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
                                          [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
                                          [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
                                          [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
                                          [dcgan.h4_w, dcgan.h4_b, None])

                # Below is codes for visualization
            OPTION = 2
            visualize(sess, dcgan, FLAGS, OPTION)'''

if __name__ == '__main__':
    tf.app.run()
