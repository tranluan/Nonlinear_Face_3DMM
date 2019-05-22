'''
Outline of the main training script
Part of data/input pipeline is ommitted

'''


from __future__ import division
import os
import time
import csv
import random
from random import randint
from math import floor
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from six.moves import xrange
#from progress.bar import Bar
from rendering_ops import *
from ops import *
from utils import *


TRI_NUM = 105840
VERTEX_NUM = 53215


class DCGAN(object):
    def __init__(self, sess, config, devices=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
        """
        self.sess = sess
        self.c_dim = config.c_dim
        self.gpu_num = len(config.gpu.split())

        
        
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.sample_size = config.sample_size
        self.image_size = 224 #config.image_size
        self.texture_size = [192, 224]
        self.z_dim = config.z_dim
        self.gf_dim = config.gf_dim
        self.df_dim = config.df_dim
        self.gfc_dim = config.gfc_dim
        self.dfc_dim = config.dfc_dim

        self.shape_loss = config.shape_loss if hasattr(config, 'shape_loss') else "l2"
        self.tex_loss   = config.tex_loss if hasattr(config, 'tex_loss') else "l1"
        
        self.mDim = 8
                
        self.vertexNum = VERTEX_NUM
        self.landmark_num = 68

        
        self.checkpoint_dir = config.checkpoint_dir
        self.samples_dir = config.samples_dir

        if not os.path.exists(self.samples_dir+"/"+self.model_dir):
            os.makedirs(self.samples_dir+"/"+self.model_dir)
        if not os.path.exists(self.checkpoint_dir+"/"+self.model_dir):
            os.makedirs(self.checkpoint_dir+"/"+self.model_dir)

        self.setupParaStat()
        self.setupValData()
        self.build_model()
    
    def build_model(self):
                
        self.m_labels       = tf.placeholder(tf.float32, [self.batch_size, self.mDim], name='m_labels')
        self.shape_labels   = tf.placeholder(tf.float32, [self.batch_size, self.vertexNum * 3], name='shape_labels')
        self.texture_labels = tf.placeholder(tf.float32, [self.batch_size, self.texture_size[0], self.texture_size[1], self.c_dim], name='tex_labels')
         
        
        self.input_images   = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='input_images')

        self.shape_fx, self.tex_fx, self.m = self.generator_encoder( self.input_images, is_reuse=False)
        self.shape = self.generator_decoder_shape(shape_fx, is_reuse=False, is_training=True)
        self.texture = self.generator_decoder_texture(tex_fx, is_reuse=False, is_training=True)

        # Here we estimate the whitenning projection matrix m and shape S
        # In order to do rendering/ calculate landmark we convert it back using mean, std
        self.m_full = self.m * self.std_m_tf + self.mean_m_tf
        self.m_labels_full = self.m_labels * self.std_m_tf + self.mean_m_tf
        self.shape_full = self.shape * self.std_shape_tf + self.mean_shape_tf
        self.shape_labels_full = self.shape_labels * self.std_shape_tf + self.mean_shape_tf
        

        # Rendering
        self.G_images, self.G_images_mask = warp_texture(self.texture, self.m_full, self.shape_full, output_size=self.image_size)

        self.G_images_mask = tf.expand_dims(self.G_images_mask, -1)
        self.G_images = tf.multiply(self.G_images, self.G_images_mask) + tf.multiply(self.input_images, 1 - self.G_images_mask)

        self.landmark_u, self.landmark_v = compute_landmarks(self.m_full, self.shape_full, output_size=self.image_size)

        
       
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.g_en_vars = [var for var in t_vars if 'g_k' in var.name]
        self.g_tex_de_vars = [var for var in t_vars if 'g_h' in var.name]
        self.g_shape_de_vars = [var for var in t_vars if 'g_s' in var.name]

        #self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep = 10)

    def setupLossFunctions(self):
        # Losses
        self.texture_vis_mask = tf.cast(tf.not_equal(self.texture_labels, tf.ones_like(self.texture_labels)*(-1)), tf.float32)
        self.texture_ratio = tf.reduce_sum(self.texture_vis_mask)  / (self.batch_size* self.texture_size[0] * self.texture_size[1] * self.c_dim)
        self.img_ratio     = tf.reduce_sum(self.G_images_mask)/ (self.batch_size* self.output_size  * self.output_size)


        # Pretrain only losses
        self.G_loss_m = norm_loss(self.m, self.m_labels,     loss_type = 'l2')
        self.G_loss_shape   = norm_loss(self.shape, self.shape_labels, loss_type = self.shape_loss)
        self.G_loss_texture = norm_loss(self.texture, self.texture_labels, mask = texture_vis_mask, loss_type = self.tex_loss)  / texture_ratio

        self.G_loss_rescon  = 10*norm_loss(G_images, input_images, loss_type = self.tex_loss ) / img_ratio
        self.G_landmark_loss = (tf.reduce_mean(tf.nn.l2_loss(landmark_u - landmark_u_labels )) +  tf.reduce_mean(tf.nn.l2_loss(landmark_v - landmark_v_labels ))) / self.landmark_num / self.batch_size / 80
        
        if self.is_pretrain:
            self.G_loss = self.G_loss_m + self.G_loss_shape + self.G_loss_texture
        else:
            self.G_loss = self.G_loss_rescon + self.G_landmark_loss #+ self.G_loss_adversarial

        self.G_loss_wlandmark = self.G_loss + self.G_landmark_loss

    
    def setupParaStat(self):

        ## Compute mean, std of m and shape of the training set 
        ## to whiten the data

        self.mean_shape_tf = tf.constant(self.mean_shape, tf.float32)
        self.std_shape_tf = tf.constant(self.std_shape, tf.float32)

        self.mean_m_tf = tf.constant(self.mean_m, tf.float32)
        self.std_m_tf = tf.constant(self.std_m, tf.float32)
        


    def setupTrainingData(self):
        # Training data - 300W

        dataset = ['AFW', 'AFW_Flip', 'HELEN', 'HELEN_Flip', 'IBUG', 'IBUG_Flip', 'LFPW', 'LFPW_Flip']
        dataset_num = len(dataset)


        images = [0] * dataset_num
        pid = [0] * dataset_num
        m = [0] * dataset_num
        pose = [0] * dataset_num
        shape = [0] * dataset_num
        exp = [0] * dataset_num
        tex_para = [0] * dataset_num
        tex = [0] * dataset_num
        il = [0] * dataset_num
        alb = [0] * dataset_num
        mask = [0] * dataset_num

        for i in range(dataset_num):
            images[i], pid[i], m[i], pose[i], shape[i], exp[i], tex_para[i], _ = load_300W_LP_dataset(dataset[i])


        self.image_filenames   = np.concatenate(images, axis=0)
        images = None

        all_m = np.concatenate(m, axis=0)

        all_shape_para    = np.concatenate(shape, axis=0)
        all_exp_para      = np.concatenate(exp, axis=0)
        self.all_tex_para = np.concatenate(tex_para, axis=0)
        self.pids_300W    = np.concatenate(pid, axis=0)
        #self.all_il       = np.concatenate(il, axis=0)


        self.all_m  = np.divide(np.subtract(all_m, self.mean_m), self.std_m)

        self.mean_shape_para = np.mean(all_shape_para, axis=0)
        self.std_shape_para  = np.std(all_shape_para, axis=0)
        self.all_shape_para  = all_shape_para #np.divide(np.subtract(all_shape_para, self.mean_shape_para), self.std_shape_para)


        self.mean_exp_para = np.mean(all_exp_para, axis=0)
        self.std_exp_para  = np.std(all_exp_para, axis=0)
        self.all_exp_para  = all_exp_para #np.divide(np.subtract(all_exp_para, self.mean_exp_para), self.std_exp_para)

        return

    
 


    def train(self, config):
        # Using 2 separated optim for with and withou landmark losses
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.G_loss, var_list=self.g_vars, colocate_gradients_with_ops=True)
        g_en_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.G_loss_wlandmark, var_list=self.g_en_vars, colocate_gradients_with_ops=True)
        tf.global_variables_initializer().run()
        

        
        """Train DCGAN"""
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch0 = checkpoint_counter + 1
            print(" [*] Load SUCCESS")
        else:
            epoch0 = 1
            print(" [!] Load failed...")


        start_time = time.time()
        
        for epoch in xrange(epoch0, config.epoch):
            
            batch_idxs = min(len(valid_idx), config.train_size) // config.batch_size
            
            for idx in xrange(0, batch_idxs):
                '''
                Data processing. Create feed_dict
                .
                .
                .
                .
                .

                '''
                

                if np.mod(idx, 2) == 0:
                    # Update G
                    self.sess.run([g_optim], feed_dict=ffeed_dict)
                else:
                    # Update G encoder only
                    self.sess.run([g_en_optim], feed_dict=ffeed_dict)
                


            self.save(config.checkpoint_dir, epoch)

                
            

    def generator_encoder(self, image,  is_reuse=False, is_training = True):   

        ''' 
        Creating a encoder network

        Output: shape_fx, tex_fc, m, il

        '''

        
        if not is_reuse:
            self.g_bn0_0 = batch_norm(name='g_k_bn0_0')
            self.g_bn0_1 = batch_norm(name='g_k_bn0_1')
            self.g_bn0_2 = batch_norm(name='g_k_bn0_2')
            self.g_bn0_3 = batch_norm(name='g_k_bn0_3')
            self.g_bn1_0 = batch_norm(name='g_k_bn1_0')
            self.g_bn1_1 = batch_norm(name='g_k_bn1_1')
            self.g_bn1_2 = batch_norm(name='g_k_bn1_2')
            self.g_bn1_3 = batch_norm(name='g_k_bn1_3')
            self.g_bn2_0 = batch_norm(name='g_k_bn2_0')
            self.g_bn2_1 = batch_norm(name='g_k_bn2_1')
            self.g_bn2_2 = batch_norm(name='g_k_bn2_2')
            self.g_bn2_3 = batch_norm(name='g_k_bn2_3')
            self.g_bn3_0 = batch_norm(name='g_k_bn3_0')
            self.g_bn3_1 = batch_norm(name='g_k_bn3_1')
            self.g_bn3_2 = batch_norm(name='g_k_bn3_2')
            self.g_bn3_3 = batch_norm(name='g_k_bn3_3')
            self.g_bn4_0 = batch_norm(name='g_k_bn4_0')
            self.g_bn4_1 = batch_norm(name='g_k_bn4_1')
            self.g_bn4_2 = batch_norm(name='g_k_bn4_2')
            self.g_bn4_c = batch_norm(name='g_h_bn4_c')
            self.g_bn5   = batch_norm(name='g_k_bn5')
            self.g_bn5_m     = batch_norm(name='g_k_bn5_m')
            self.g_bn5_il    = batch_norm(name='g_k_bn5_il')
            self.g_bn5_shape = batch_norm(name='g_k_bn5_shape')
            self.g_bn5_shape_linear = batch_norm(name='g_k_bn5_shape_linear')
            self.g_bn5_tex   = batch_norm(name='g_k_bn5_tex')

       

        k0_1 = elu(self.g_bn0_1(conv2d(image, self.gf_dim*1, k_h=7, k_w=7, d_h=2, d_w =2, use_bias = False, name='g_k01_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k0_2 = elu(self.g_bn0_2(conv2d(k0_1, self.gf_dim*2, d_h=1, d_w =1, use_bias = False, name='g_k02_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))

        k1_0 = elu(self.g_bn1_0(conv2d(k0_2, self.gf_dim*2, d_h=2, d_w =2, use_bias = False, name='g_k10_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k1_1 = elu(self.g_bn1_1(conv2d(k1_0, self.gf_dim*2, d_h=1, d_w =1, use_bias = False, name='g_k11_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k1_2 = elu(self.g_bn1_2(conv2d(k1_1, self.gf_dim*4, d_h=1, d_w =1, use_bias = False, name='g_k12_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        #k1_3 =               maxpool2d(k1_2, k=2, padding='VALID')
        k2_0 = elu(self.g_bn2_0(conv2d(k1_2, self.gf_dim*4, d_h=2, d_w =2, use_bias = False, name='g_k20_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k2_1 = elu(self.g_bn2_1(conv2d(k2_0, self.gf_dim*3, d_h=1, d_w =1, use_bias = False, name='g_k21_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k2_2 = elu(self.g_bn2_2(conv2d(k2_1, self.gf_dim*6, d_h=1, d_w =1, use_bias = False, name='g_k22_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        #k2_3 =               maxpool2d(k2_2, k=2, padding='VALID')
        k3_0 = elu(self.g_bn3_0(conv2d(k2_2, self.gf_dim*6, d_h=2, d_w =2, use_bias = False, name='g_k30_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k3_1 = elu(self.g_bn3_1(conv2d(k3_0, self.gf_dim*4, d_h=1, d_w =1, use_bias = False, name='g_k31_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k3_2 = elu(self.g_bn3_2(conv2d(k3_1, self.gf_dim*8, d_h=1, d_w =1, use_bias = False, name='g_k32_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        #k3_3 =               maxpool2d(k3_2, k=2, padding='VALID')
        k4_0 = elu(self.g_bn4_0(conv2d(k3_2, self.gf_dim*8, d_h=2, d_w =2, use_bias = False, name='g_k40_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k4_1 = elu(self.g_bn4_1(conv2d(k4_0, self.gf_dim*5, d_h=1, d_w =1, use_bias = False, name='g_k41_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        

        # M
        k51_m = self.g_bn5_m(    conv2d(k4_1, int(self.gfc_dim/5),  d_h=1, d_w =1, name='g_k5_m_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k51_shape_ = get_shape(k51_m)
        k52_m = tf.nn.avg_pool(k51_m, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_m = tf.reshape(k52_m, [-1, int(self.gfc_dim/5)])
        k6_m = linear(k52_m, self.mDim, 'g_k6_m_lin', reuse = is_reuse)
        
        # Il
        k51_il = self.g_bn5_il(    conv2d(k4_1, int(self.gfc_dim/5),  d_h=1, d_w =1, name='g_k5_il_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k52_il = tf.nn.avg_pool(k51_il, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_il = tf.reshape(k52_il, [-1, int(self.gfc_dim/5)])
        k6_il = linear(k52_il, self.ilDim, 'g_k6_il_lin', reuse = is_reuse)

        # Shape
        k51_shape = self.g_bn5_shape(conv2d(k4_1, self.gfc_dim/2,  d_h=1, d_w =1, name='g_k5_shape_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k52_shape = tf.nn.avg_pool(k51_shape, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_shape = tf.reshape(k52_shape, [-1, int(self.gfc_dim/2)])

        # Albedo
        k51_tex   = self.g_bn5_tex(  conv2d(k4_1, self.gfc_dim/2,  d_h=1, d_w =1, name='g_k5_tex_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k52_tex = tf.nn.avg_pool(k51_tex, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_tex = tf.reshape(k52_tex, [-1, int(self.gfc_dim/2)])
        
        return k52_shape, k52_tex, k6_m, k6_il

    def generator_decoder_shape(self, k52_shape, is_reuse=False, is_training=True):
        if False:  ## This is for shape decoder as fully connected network (NOT FULLY COMPATIBLE WITH THE REST OF THE CODE)
            return self.generator_decoder_shape_1d(k52_shape, is_reuse, is_training)
        else: 

            n_size = get_shape(k52_shape)
            n_size = n_size[0]

            vt2pixel_u, vt2pixel_v = load_3DMM_vt2pixel()


            #Vt2pix
            vt2pixel_u_const = tf.constant(vt2pixel_u[:-1], tf.float32)
            vt2pixel_v_const = tf.constant(vt2pixel_v[:-1], tf.float32)

            if self.is_partbase_albedo:
                shape_2d = self.generator_decoder_shape_2d_partbase(k52_shape, is_reuse, is_training)
            else:
                shape_2d = self.generator_decoder_shape_2d_v1(k52_shape, is_reuse, is_training) 

            vt2pixel_v_const_ = tf.tile(tf.reshape(vt2pixel_v_const, shape =[1,1,-1]), [n_size, 1,1])
            vt2pixel_u_const_ = tf.tile(tf.reshape(vt2pixel_u_const, shape =[1,1,-1]), [n_size, 1,1])

            shape_1d = tf.reshape(bilinear_sampler( shape_2d, vt2pixel_v_const_, vt2pixel_u_const_), shape=[n_size, -1])

            return shape_1d, shape_2d


    def generator_decoder_shape_1d(self, shape_fx, is_reuse=False, is_training=True):
        s6 = elu(self.g1_bn6(linear(k52_shape, 1000, scope= 'g_s6_lin', reuse = is_reuse), train=is_training, reuse = is_reuse), name="g_s6_prelu")
        s7 = linear(s6, self.vertexNum*3, scope= 'g_s7_lin', reuse = is_reuse)

        return s7


    def generator_decoder_shape_2d(self, shape_fx, is_reuse=False, is_training=True):
        '''
        Create shape decoder network
        Output: 3d_shape [N, (self.vertexNum*3)]
        '''

        if not is_reuse:
            self.g2_bn0_0 = batch_norm(name='g_s_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_s_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_s_bn0_2')        
            self.g2_bn1_0 = batch_norm(name='g_s_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_s_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_s_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_s_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_s_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_s_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_s_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_s_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_s_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_s_bn4_0')
            self.g2_bn4   = batch_norm(name='g_s_bn4')
            self.g2_bn5   = batch_norm(name='g_s_bn5')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(k52_tex, self.gfc_dim*s32_h*s32_w, scope= 'g_s5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, self.gfc_dim])
        h5 = elu(self.g2_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_s4', reuse = is_reuse)
        h4_1 = elu(self.g2_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, self.gf_dim*8, strides=[1,1], name='g_s40', reuse = is_reuse)
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, self.gf_dim*8, strides=[2,2], name='g_s32', reuse = is_reuse)
        h3_2 = elu(self.g2_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, self.gf_dim*4, strides=[1,1], name='g_s31', reuse = is_reuse)
        h3_1 = elu(self.g2_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, self.gf_dim*6, strides=[1,1], name='g_s30', reuse = is_reuse)
        h3_0 = elu(self.g2_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, self.gf_dim*6, strides=[2,2], name='g_s22', reuse = is_reuse)
        h2_2 = elu(self.g2_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, self.gf_dim*3, strides=[1,1], name='g_s21', reuse = is_reuse)
        h2_1 = elu(self.g2_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, self.gf_dim*4, strides=[1,1], name='g_s20', reuse = is_reuse)
        h2_0 = elu(self.g2_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, self.gf_dim*4, strides=[2,2], name='g_s12', reuse = is_reuse)
        h1_2 = elu(self.g2_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, self.gf_dim*2, strides=[1,1], name='g_s11', reuse = is_reuse)
        h1_1 = elu(self.g2_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,self.gf_dim*2, strides=[1,1], name='g_s10', reuse = is_reuse)
        h1_0 = elu(self.g2_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, self.gf_dim*2, strides=[2,2], name='g_s02', reuse = is_reuse)
        h0_2 = elu(self.g2_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, self.gf_dim, strides=[1,1], name='g_s01', reuse = is_reuse)
        h0_1 = elu(self.g2_bn0_1(h0_1, train=is_training, reuse = is_reuse))
           
        h0 = 2*tf.nn.tanh(deconv2d(h0_1, self.c_dim, strides=[1,1], name='g_s0', reuse = is_reuse))
            
        return h0



    def generator_decoder_albedo(self, tex_fx, is_reuse=False, is_training=True):
        '''
        Create texture decoder network
        Output: uv_texture [N, self.texture_sz[0], self.texture_sz[1], self.c_dim]
        '''

        if not is_reuse:
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')        
            self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
            self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
            self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
            self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
            self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
            self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
            self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
            self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
            self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
            self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
            self.g1_bn4   = batch_norm(name='g_h_bn4')
            self.g1_bn5   = batch_norm(name='g_h_bn5')
            #self.g1_bn6   = batch_norm(name='g_s_bn6')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)

        df = int(self.gf_dim)
                    
        # project `z` and reshape
        h5 = linear(k52_tex, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g1_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse = is_reuse)
        h4_1 = elu(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_h40', reuse = is_reuse)
        h4_0 = elu(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_h32', reuse = is_reuse)
        h3_2 = elu(self.g1_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
        h3_1 = elu(self.g1_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
        h3_0 = elu(self.g1_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
        h2_2 = elu(self.g1_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
        h2_1 = elu(self.g1_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
        h2_0 = elu(self.g1_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
        h1_2 = elu(self.g1_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
        h1_1 = elu(self.g1_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
        h1_0 = elu(self.g1_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
        h0_2 = elu(self.g1_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
        h0_1 = elu(self.g1_bn0_1(h0_1, train=is_training, reuse = is_reuse))
           
        h0 = tf.nn.tanh(deconv2d(h0_1, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))
            
        return h0

              
    @property
    def model_dir(self):
        return "" # "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
      
    def save(self, checkpoint_dir, step):
        model_name = "Nonlinear3DMM.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        print(" Saved checkpoint %s-%d" % (os.path.join(checkpoint_dir, model_name), step))

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))


            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")

            return False, 0




