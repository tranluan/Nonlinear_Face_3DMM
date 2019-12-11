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
from progress.bar import Bar
from rendering_ops import *
from ops import *
from utils import *

#SUBJECT_NUM_MTPIE = 200
#SUBJECT_NUM_MTPIE_FULL = 347
#SUBJECT_NUM_CASIA = 10575

TRI_NUM = 105840
VERTEX_NUM = 53215

TRI_NUM_REMESH6k = 12281
VERTEX_NUM_REMESH6k = 6248

#TRI_NUM_REDUCE = 12281
VERTEX_NUM_REDUCE = 39111

WARPING = True

CONST_PIXELS_NUM = 20

HISTRORY_SZ = 1000


class DCGAN(object):
    def __init__(self, sess, config, devices=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.c_dim = config.c_dim
        print('------------------------------')
        print(config.gpu.split(','))
        self.gpu_num = len(config.gpu.split(','))
        print(self.gpu_num)

        
        self.is_using_landmark = config.is_using_landmark
        self.is_using_symetry = config.is_using_symetry
        self.is_using_res_symetry = config.is_using_res_symetry
        self.is_using_recon = config.is_using_recon
        self.is_using_frecon = config.is_using_frecon
        self.is_using_graddiff = config.is_using_graddiff
        self.is_gt_m = config.is_gt_m
        self.is_using_linear = False #config.is_using_linear
        self.is_batchwise_white_shading = config.is_batchwise_white_shading
        self.is_const_albedo = config.is_const_albedo
        self.is_const_local_albedo = config.is_const_local_albedo
        self.is_smoothness = config.is_smoothness

        self.is_using_GAN = config.is_using_GAN
        self.is_using_L2_GAN = config.is_using_L2_GAN
        self.is_using_patchGAN = config.is_using_patchGAN
        self.is_random_gan_labels = config.is_random_gan_labels
        self.is_combine = config.is_combine

        self.is_2d_normalize = True




        self.is_partbase_albedo = config.is_partbase_albedo
        self.is_2d_shape = True #config.is_2d_shape


        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.sample_size = config.sample_size
        self.output_size = config.output_size
        self.texture_size = [192, 224]
        self.z_dim = config.z_dim
        self.gf_dim = config.gf_dim
        self.df_dim = config.df_dim
        self.gfc_dim = config.gfc_dim
        self.dfc_dim = config.dfc_dim

        self.shape_loss = config.shape_loss if hasattr(config, 'shape_loss') else "l2"
        self.tex_loss   = 'l1' #config.tex_loss if hasattr(config, 'tex_loss') else "l1"
        
        self.mDim = 8
        self.poseDim = 7
        self.shapeDim = 199
        self.expDim = 29
        self.texDim = 40
        self.ilDim = 9 * 3
        self.is_reduce = config.is_reduce
        if self.is_reduce:
            self.vertexNum = VERTEX_NUM_REDUCE
        else:
            self.vertexNum = VERTEX_NUM

        self.landmark_num = 68

        # batch normalization : deals with poor initialization helps gradient flow
        self.bns = {}

        self.d_bn0_0 = batch_norm(name='d_k_bn0_0')
        self.d_bn0_1 = batch_norm(name='d_k_bn0_1')
        self.d_bn0_2 = batch_norm(name='d_k_bn0_2')
        self.d_bn1_0 = batch_norm(name='d_k_bn1_0')
        self.d_bn1_1 = batch_norm(name='d_k_bn1_1')
        self.d_bn1_2 = batch_norm(name='d_k_bn1_2')
        self.d_bn1_3 = batch_norm(name='d_k_bn1_3')
        self.d_bn2_0 = batch_norm(name='d_k_bn2_0')
        self.d_bn2_1 = batch_norm(name='d_k_bn2_1')
        self.d_bn2_2 = batch_norm(name='d_k_bn2_2')
        self.d_bn3_0 = batch_norm(name='d_k_bn3_0')
        self.d_bn3_1 = batch_norm(name='d_k_bn3_1')
        self.d_bn3_2 = batch_norm(name='d_k_bn3_2')
        self.d_bn3_3 = batch_norm(name='d_k_bn3_3')
        self.d_bn4_0 = batch_norm(name='d_k_bn4_0')
        self.d_bn4_1 = batch_norm(name='d_k_bn4_1')
        self.d_bn4_2 = batch_norm(name='d_k_bn4_2')
        self.d_bn4_1_l = batch_norm(name='d_k_bn4_1_l')
        self.d_bn4_2_l = batch_norm(name='d_k_bn4_2_l')
        self.d_bn4_1_r = batch_norm(name='d_k_bn4_1_r')
        self.d_bn4_2_r = batch_norm(name='d_k_bn4_2_r')
        self.d_bn4_1_p = batch_norm(name='d_k_bn4_1_p')
        self.d_bn4_2_p = batch_norm(name='d_k_bn4_2_P')
        self.d_bn4_1_a = batch_norm(name='d_k_bn4_1_a')
        self.d_bn4_2_a = batch_norm(name='d_k_bn4_2_a')                
        self.d_bn5   = batch_norm(name='d_k_bn5')
        
        
        
        #self.g2_bn0_0 = batch_norm(name='g_h2_bn0_0')
        #self.g2_bn0_1 = batch_norm(name='g_h2_bn0_1')
        #self.g2_bn0_2 = batch_norm(name='g_h2_bn0_2')

        
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.samples_dir = config.samples_dir
        #model_dir = "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
        if not os.path.exists(self.samples_dir+"/"+self.model_dir):
            os.makedirs(self.samples_dir+"/"+self.model_dir)
        if not os.path.exists(self.checkpoint_dir+"/"+self.model_dir):
            os.makedirs(self.checkpoint_dir+"/"+self.model_dir)
        self.devices = devices
        self.setupParaStat()
        self.setupValData()
        self.build_model()
    
    def build_model(self):
        def filename2image(input_filenames, offset_height = None, offset_width = None, target_height=None, target_width=None):
            batch_size = len(input_filenames)
            if offset_height != None:
                offset_height = tf.split(offset_height, batch_size)
                offset_width = tf.split(offset_width, batch_size)

            images = []          
            for i in range(batch_size):
                file_contents = tf.read_file(input_filenames[i])
                image = tf.image.decode_png(file_contents, channels=3)
                if offset_height != None:
                    image = tf.image.crop_to_bounding_box(image, tf.reshape(offset_height[i], []), tf.reshape(offset_width[i], []), target_height, target_width)

                images.append(image)
            return tf.cast(tf.stack(images), tf.float32)

        #self.subject_num = SUBJECT_NUM_CASIA
            
        #self.id_labels_CASIA  = tf.placeholder(tf.float32, [self.batch_size, self.subject_num], name='id_labels_CASIA')
        self.m_300W_labels       = tf.placeholder(tf.float32, [self.batch_size, self.mDim], name='m_300W_labels')
        self.shape_300W_labels   = tf.placeholder(tf.float32, [self.batch_size, self.vertexNum * 3], name='shape_300W_labels')
        self.texture_300W_labels = tf.placeholder(tf.float32, [self.batch_size, self.texture_size[0], self.texture_size[1], self.c_dim], name='tex_300W_labels')
        #self.exp_300W_labels     = tf.placeholder(tf.float32, [self.batch_size, self.expDim], name='exp_300W_labels')
        #self.il_300W_labels      = tf.placeholder(tf.float32, [self.batch_size, self.ilDim], name='lighting_300W_labels')

        self.input_offset_height  = tf.placeholder(tf.int32, [self.batch_size], name='input_offset_height')
        self.input_offset_width   = tf.placeholder(tf.int32, [self.batch_size], name='input_offset_width')

        self.input_images_fn_300W = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]
        self.input_masks_fn_300W  = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]
        self.texture_labels_fn_300W = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]
        self.texture_masks_fn_300W  = [tf.placeholder(dtype=tf.string) for _ in range(self.batch_size)]


        
        #self.input_images_300W   = filename2image(self.input_images_fn_300W, offset_height = self.input_offset_height, offset_width = self.input_offset_width, target_height=self.output_size, target_width=self.output_size)
        #self.input_images_300W   = self.input_images_300W /127.5 - 1
        #self.input_masks_300W    = filename2image(self.input_masks_fn_300W,  offset_height = self.input_offset_height, offset_width = self.input_offset_width, target_height=self.output_size, target_width=self.output_size)
        #self.input_masks_300W    = self.input_masks_300W / 255.0
        

        #self.texture_300W_labels    = filename2image(self.texture_labels_fn_300W)
        #self.texture_300W_labels    = self.texture_300W_labels / 127.5 - 1

        #self.texture_mask_300W_labels = filename2image(self.texture_masks_fn_300W)
        #self.texture_mask_300W_labels = self.texture_mask_300W_labels / 255.0

        # For const alb loss
        self.albedo_indexes_x1 = tf.placeholder(tf.int32, [self.batch_size, CONST_PIXELS_NUM, 1], name='idexes_x1')
        self.albedo_indexes_y1 = tf.placeholder(tf.int32, [self.batch_size, CONST_PIXELS_NUM, 1], name='idexes_y1')

        self.albedo_indexes_x2 = tf.placeholder(tf.int32, [self.batch_size, CONST_PIXELS_NUM, 1], name='idexes_x2')
        self.albedo_indexes_y2 = tf.placeholder(tf.int32, [self.batch_size, CONST_PIXELS_NUM, 1], name='idexes_y2')

        self.const_alb_mask = load_const_alb_mask()

        

        def rendering(texture, m ,shape, background=None, extra_mask=None):
            image, mask = warp_texture(texture, m, shape, output_size=self.output_size, is_reduce = self.is_reduce)
            mask = tf.expand_dims(mask, -1)

            if extra_mask != None:
                mask = tf.multiply(extra_mask, mask)

            if background != None:
                image = tf.multiply(image, mask) + tf.multiply(background, 1 - mask)

            return image, mask

        def rendering_wflow(texture, flow_u, flow_v, flow_mask, background=None, extra_mask=None):
            image = bilinear_sampler(texture, flow_v, flow_u)
            mask = tf.expand_dims(flow_mask, -1)

            if extra_mask != None:
                mask = tf.multiply(extra_mask, mask)

            if background != None:
                image = tf.multiply(image, mask) + tf.multiply(background, 1 - mask)

            return image, mask

        def alb2tex(albedo, shade):
            return 2.0*tf.multiply( (albedo + 1.0)/2.0, shade) - 1


        def model_and_loss(input_images_fn_300W, input_masks_fn_300W, texture_labels_fn_300W, texture_masks_fn_300W, input_offset_height, input_offset_width, m_300W_labels, shape_300W_labels, albedo_indexes_x1, albedo_indexes_y1, albedo_indexes_x2, albedo_indexes_y2):
            batch_size = self.batch_size / self.gpu_num
            input_images_300W_   = filename2image(input_images_fn_300W, offset_height = input_offset_height, offset_width = input_offset_width, target_height=self.output_size, target_width=self.output_size)
            input_images_300W    = input_images_300W_ /127.5 - 1

            input_masks_300W    = filename2image(input_masks_fn_300W,  offset_height = input_offset_height, offset_width = input_offset_width, target_height=self.output_size, target_width=self.output_size)
            input_masks_300W    = input_masks_300W / 255.0

            texture_300W_labels    = filename2image(texture_labels_fn_300W)
            texture_300W_labels    = texture_300W_labels / 127.5 - 1

            texture_mask_300W_labels = filename2image(texture_masks_fn_300W)
            texture_mask_300W_labels = texture_mask_300W_labels / 255.0


            ## ------------------------- Network ---------------------------
            shape_fx_300W, alb_fx_300W, m_300W, il_300W  = self.generator_encoder( input_images_300W, is_reuse=False)
            
            if False:   #v2
                shape_base_300W, shape_2d_base_300W, shape_res_300W, shape_2d_res_300W = self.generator_decoder_shape(shape_fx_300W, is_reuse=False, is_training=True)
                shape_2d_300W = shape_2d_base_300W + shape_2d_res_300W
                shape_300W = shape_base_300W + shape_res_300W

                albedo_base_300W, albedo_res_300W  = self.generator_decoder_albedo(alb_fx_300W, is_reuse=False, is_training=True)
                albedo_300W = albedo_base_300W + albedo_res_300W
            else:       #v3

                shape_base_300W, shape_2d_base_300W, shape_300W, shape_2d_300W = self.generator_decoder_shape(shape_fx_300W, is_reuse=False, is_training=True)
                shape_2d_res_300W = shape_2d_300W - shape_2d_base_300W
                shape_res_300W = shape_300W - shape_base_300W

                albedo_base_300W, albedo_300W  = self.generator_decoder_albedo(alb_fx_300W, is_reuse=False, is_training=True)
                albedo_res_300W = albedo_300W - albedo_base_300W


            m_300W_full         = self.m2full(m_300W)
            m_300W_labels_full  = self.m2full(m_300W_labels)
            shape_base_300W_full    = self.shape2full(shape_base_300W)
            shape_300W_full         = self.shape2full(shape_300W)
            shape_300W_labels_full  = self.shape2full(shape_300W_labels)
            

            if self.is_gt_m: 
                m_4syn = m_300W_labels_full
            else:
                m_4syn = m_300W_full

            # Shade
            shade_base_300W, rotated_normal_2d  = generate_shade(il_300W, m_4syn, shape_base_300W_full, self.texture_size, is_reduce= self.is_reduce, is_with_normal=True)
            shade_300W       = generate_shade(il_300W, m_4syn, shape_300W_full,      self.texture_size, is_reduce= self.is_reduce)

            # Texture
            texture_base_300W     = alb2tex(albedo_base_300W, shade_base_300W)
            texture_mix_s0a1_300W = alb2tex(albedo_300W,      shade_base_300W)
            texture_mix_s1a0_300W = alb2tex(albedo_base_300W, shade_300W)
            texture_300W          = alb2tex(albedo_300W,      shade_300W)
            

            # Rendering
            G_flow_base_u, G_flow_base_v, G_flow_base_300W_mask = warping_flow(m_4syn, shape_base_300W_full, output_size=self.output_size, is_reduce = self.is_reduce)
            G_flow_u,      G_flow_v,      G_flow_300W_mask      = warping_flow(m_4syn, shape_300W_full,      output_size=self.output_size, is_reduce = self.is_reduce)

            G_images_base_300W, G_images_base_300W_mask = rendering_wflow(texture_base_300W,     G_flow_base_u, G_flow_base_v, G_flow_base_300W_mask, background=input_images_300W, extra_mask=input_masks_300W)
            G_images_mix_s0a1_300W, _                   = rendering_wflow(texture_mix_s0a1_300W, G_flow_base_u, G_flow_base_v, G_flow_base_300W_mask, background=input_images_300W, extra_mask=input_masks_300W)
            G_images_mix_s1a0_300W, _                   = rendering_wflow(texture_mix_s1a0_300W, G_flow_u,      G_flow_v,      G_flow_300W_mask,      background=input_images_300W, extra_mask=input_masks_300W)
            G_images_300W, G_images_300W_mask           = rendering_wflow(texture_300W,          G_flow_u,      G_flow_v,      G_flow_300W_mask,      background=input_images_300W, extra_mask=input_masks_300W)
            


            # Landmarks
            landmark_u_300W, landmark_v_300W                = compute_landmarks(m_4syn,             shape_300W_full,        output_size=self.output_size, is_reduce = self.is_reduce)
            landmark_u_base_300W, landmark_v_base_300W      = compute_landmarks(m_300W_full,        shape_base_300W_full,   output_size=self.output_size, is_reduce = self.is_reduce)
            landmark_u_base2_300W, landmark_v_base2_300W    = compute_landmarks(m_300W_full,        shape_300W_full,        output_size=self.output_size, is_reduce = self.is_reduce)
            landmark_u_300W_labels, landmark_v_300W_labels  = compute_landmarks(m_300W_labels_full, shape_300W_labels_full, output_size=self.output_size, is_reduce = self.is_reduce)

            

            ## ------------------------- Losses ---------------------------
            g_loss = tf.zeros(1)

            G_loss_shape   = 10*norm_loss(shape_base_300W, shape_300W_labels, loss_type = self.shape_loss)
            G_loss_m       = 5*norm_loss(m_300W,        m_300W_labels,     loss_type = 'l2')
            g_loss  += G_loss_m + G_loss_shape


            texture_vis_mask = tf.cast(tf.not_equal(texture_300W_labels, tf.ones_like(texture_300W_labels)*(-1)), tf.float32)
            texture_vis_mask = tf.multiply(texture_vis_mask, texture_mask_300W_labels)
            texture_ratio = tf.reduce_sum(texture_vis_mask)  / (batch_size* self.texture_size[0] * self.texture_size[1] * self.c_dim)

            

            uv_mask_tf = tf.expand_dims(tf.expand_dims(tf.constant( self.uv_mask, dtype = tf.float32 ), 0), -1)
            if self.is_batchwise_white_shading:
                

                mean_shade = tf.reduce_mean( tf.multiply(shade_base_300W, uv_mask_tf) , axis=[0,1,2]) * 16384 / 10379
                G_loss_white_shading = 10*norm_loss(mean_shade,  0.99*tf.ones([1, 3], dtype=tf.float32), loss_type = "l2")
            else:
                G_loss_white_shading = tf.zeros(1)

            g_loss += G_loss_white_shading

            
            G_loss_texture = (norm_loss(texture_base_300W,      texture_300W_labels, mask = texture_vis_mask, loss_type = self.tex_loss) +
                              #+norm_loss(texture_300W,           texture_300W_labels, mask = texture_vis_mask, loss_type = self.tex_loss) +
                              norm_loss(texture_mix_s0a1_300W,  texture_300W_labels, mask = texture_vis_mask, loss_type = self.tex_loss) +
                              norm_loss(texture_mix_s1a0_300W,  texture_300W_labels, mask = texture_vis_mask, loss_type = self.tex_loss)) / 2 / texture_ratio

            G_loss_recon_base     = 10*norm_loss(G_images_base_300W,     input_images_300W, loss_type = self.tex_loss ) / (tf.reduce_sum(G_images_base_300W_mask)/ (batch_size* self.output_size  * self.output_size))
            G_loss_recon_mix_s0a1 = 10*norm_loss(G_images_mix_s0a1_300W, input_images_300W, loss_type = self.tex_loss ) / (tf.reduce_sum(G_images_base_300W_mask)/ (batch_size* self.output_size  * self.output_size))
            G_loss_recon_mix_s1a0 = 10*norm_loss(G_images_mix_s1a0_300W, input_images_300W, loss_type = self.tex_loss ) / (tf.reduce_sum(G_images_300W_mask)     / (batch_size* self.output_size  * self.output_size))
            G_loss_recon          = 10*norm_loss(G_images_300W,          input_images_300W, loss_type = self.tex_loss ) / (tf.reduce_sum(G_images_300W_mask)     / (batch_size* self.output_size  * self.output_size))

            
            if self.is_smoothness:
                G_loss_smoothness = 5e5*norm_loss( (shape_2d_base_300W[:, :-2, 1:-1, :] + shape_2d_base_300W[:, 2:, 1:-1, :] + shape_2d_base_300W[:, 1:-1, :-2, :] + shape_2d_base_300W[:, 1:-1, 2:, :])/4.0,
                                                    shape_2d_base_300W[:, 1:-1, 1:-1, :], loss_type = 'l2') + \
                                        norm_loss( (shape_2d_300W[:, :-2, 1:-1, :] + shape_2d_300W[:, 2:, 1:-1, :] + shape_2d_300W[:, 1:-1, :-2, :] + shape_2d_300W[:, 1:-1, 2:, :])/4.0,
                                                    shape_2d_300W[:, 1:-1, 1:-1, :], loss_type = 'l2')
            else:
                G_loss_smoothness = tf.zeros(1)
            g_loss +=  G_loss_smoothness

            G_landmark_loss = (tf.reduce_mean(tf.nn.l2_loss(landmark_u_base_300W - landmark_u_300W_labels )) +  tf.reduce_mean(tf.nn.l2_loss(landmark_v_base_300W - landmark_v_300W_labels ))) / self.landmark_num / batch_size / 50 + \
                              (tf.reduce_mean(tf.nn.l2_loss(landmark_u_base2_300W - landmark_u_300W_labels )) +  tf.reduce_mean(tf.nn.l2_loss(landmark_v_base2_300W - landmark_v_300W_labels ))) / self.landmark_num / batch_size / 50 + \
                              (tf.reduce_mean(tf.nn.l2_loss(landmark_u_300W - landmark_u_300W_labels )) +  tf.reduce_mean(tf.nn.l2_loss(landmark_v_300W - landmark_v_300W_labels ))) / self.landmark_num / batch_size / 50


            #G_landmark_loss = ((tf.reduce_mean(tf.abs(landmark_u_base_300W - landmark_u_300W_labels )) +  tf.reduce_mean(tf.abs(landmark_v_base_300W - landmark_v_300W_labels ))) / self.landmark_num / batch_size  + \
            #                  (tf.reduce_mean(tf.abs(landmark_u_base2_300W - landmark_u_300W_labels )) +  tf.reduce_mean(tf.abs(landmark_v_base2_300W - landmark_v_300W_labels ))) / self.landmark_num / batch_size + \
            #                  (tf.reduce_mean(tf.abs(landmark_u_300W - landmark_u_300W_labels )) +  tf.reduce_mean(tf.abs(landmark_v_300W - landmark_v_300W_labels ))) / self.landmark_num / batch_size )* 1e2

            if self.is_using_symetry:
                albedo_base_300W_flip = tf.map_fn(lambda img: tf.image.flip_left_right(img), albedo_base_300W)
                G_loss_symetry = 10*norm_loss(tf.maximum(tf.abs(albedo_base_300W-albedo_base_300W_flip), 0.01), 0, loss_type = self.tex_loss)
            else:
                G_loss_symetry = tf.zeros(1)
            g_loss +=  G_loss_symetry

            if self.is_using_res_symetry:
                _, _, shape_res_300W_z = tf.split(shape_2d_res_300W, num_or_size_splits=3, axis=3)
                _, _, rotated_normal_z = tf.split(rotated_normal_2d, num_or_size_splits=3, axis=3)



                
                shape_res_300W_z_flip = tf.map_fn(lambda img: tf.image.flip_left_right(img), shape_res_300W_z)
                G_loss_res_symetry = 100*norm_loss(shape_res_300W_z, tf.stop_gradient(shape_res_300W_z_flip), mask= tf.cast( tf.less(rotated_normal_z, 0), tf.float32), loss_type = 'l1')
            else:
                G_loss_res_symetry = tf.zeros(1)
            g_loss +=  G_loss_res_symetry


            G_loss_res = norm_loss(shape_res_300W, 0, loss_type = 'l1') + norm_loss(albedo_res_300W, 0, loss_type = 'l1')
            g_loss += G_loss_res

            if self.is_const_albedo:

                albedo_1 = get_pixel_value(albedo_base_300W, albedo_indexes_x1, albedo_indexes_y1)
                albedo_2 = get_pixel_value(albedo_base_300W, albedo_indexes_x2, albedo_indexes_y2)

                G_loss_albedo_const = 10*norm_loss( tf.maximum(tf.abs(albedo_1- albedo_2), 0.01), 0, loss_type = self.tex_loss)
            else:
                G_loss_albedo_const = tf.zeros(1)
            g_loss += G_loss_albedo_const

            if self.is_const_local_albedo:
                local_albedo_alpha = 0.9
                texture_300W_labels_chromaticity = (texture_300W_labels + 1.0)/2.0
                texture_300W_labels_chromaticity = tf.divide(texture_300W_labels_chromaticity, tf.reduce_sum(texture_300W_labels_chromaticity, axis=[-1], keep_dims=True) + 1e-6)

                w_u = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :-1, :, :] - texture_300W_labels_chromaticity[:, 1:, :, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :-1, :, :] )
                G_loss_local_albedo_const_u = tf.reduce_mean(norm_loss( albedo_base_300W[:, :-1, :, :], albedo_base_300W[:, 1:, :, :], loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_u) / tf.reduce_sum(w_u+1e-6)

                    
                w_v = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :, :-1, :] - texture_300W_labels_chromaticity[:, :, 1:, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :, :-1, :] )
                G_loss_local_albedo_const_v = tf.reduce_mean(norm_loss( albedo_base_300W[:, :, :-1, :], albedo_base_300W[:, :, 1:, :],  loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_v) / tf.reduce_sum(w_v+1e-6)

                G_loss_local_albedo_const = (G_loss_local_albedo_const_u + G_loss_local_albedo_const_v)*10
            else:
                G_loss_local_albedo_const = tf.zeros(1)
            g_loss += G_loss_local_albedo_const

            if self.is_using_recon:
                g_loss +=  (G_loss_recon_base + 2*G_loss_recon_mix_s0a1 + 2*G_loss_recon_mix_s1a0)/2 #G_loss_recon +
            else:
                g_loss += G_loss_texture*4
            
            g_loss += G_loss_texture

            G_loss_frecon = tf.zeros(1)
            if self.is_using_frecon:
                from vgg_face import vgg_face


                #G_images_300W_vgg = (tf.multiply(G_images_300W, G_images_300W_mask)+ 1.0)*127.5
                G_images_mix_s1a0_300W_vgg = (G_images_mix_s1a0_300W + 1.0)*127.5
                G_images_mix_s0a1_300W_vgg = (G_images_mix_s0a1_300W + 1.0)*127.5
                input_images_300W_vgg      = (input_images_300W      + 1.0)*127.5

                input_features, _, _  = vgg_face('vgg-face.mat', input_images_300W_vgg)
                recon_mix_s1a0_features, _, _  = vgg_face('vgg-face.mat', G_images_mix_s1a0_300W_vgg)
                recon_mix_s0a1_features, _, _  = vgg_face('vgg-face.mat', G_images_mix_s0a1_300W_vgg)
                
                layer_names = ['conv1_1', 'conv2_1',  'conv3_1',  'conv4_1', 'conv5_1'] #['conv1_2', 'conv2_2',  'conv3_3',  'conv4_3', 'conv5_3']
                layer_weights = [50, 500, 2000, 3000, 250]

                for i in range(len(layer_names)):
                    layer_name = layer_names[i]
                    layer_weight = layer_weights[i]
                    G_loss_frecon += norm_loss(input_features[layer_name], recon_mix_s1a0_features[layer_name], loss_type='l2') / (layer_weight*layer_weight * len(layer_names))*2
                    G_loss_frecon += norm_loss(input_features[layer_name], recon_mix_s0a1_features[layer_name], loss_type='l2') / (layer_weight*layer_weight * len(layer_names))*2

            g_loss = g_loss + G_loss_frecon



            G_loss_graddiff = tf.zeros(1)
            if self.is_using_graddiff:
                G_images_300W_grad_u = G_images_300W[:, :-1, :, :] - G_images_300W[:, 1:, :, :]
                input_images_300W_grad_u = input_images_300W[:, :-1, :, :] - input_images_300W[:, 1:, :, :]


                G_images_300W_grad_v = G_images_300W[:, :, :-1, :] - G_images_300W[:, :, 1:, :]
                input_images_300W_grad_v = input_images_300W[:, :, :-1, :] - input_images_300W[:, :, 1:, :]

                G_loss_graddiff = 10* ( norm_loss(G_images_300W_grad_u, input_images_300W_grad_u, loss_type = 'l1' ) + 
                                        norm_loss(G_images_300W_grad_u, input_images_300W_grad_u, loss_type = 'l1' ) ) #/ (batch_size* self.output_size  * self.output_size)

            g_loss = g_loss + G_loss_graddiff

            G_loss_shade_mag = norm_loss(shade_base_300W, 1, mask= tf.cast(tf.greater(texture_base_300W , 1), dtype=tf.float32) , loss_type = 'l1')
            g_loss += G_loss_shade_mag






            '''
            # GAN loss
            if self.is_using_GAN:
                if not self.is_using_L2_GAN:

                    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.images_D_logit, labels= 0.99*tf.ones_like(self.images_D_logit)))
                    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G_images_D_logit, labels= tf.zeros_like(self.G_images_D_logit)))
                    self.d_loss_fake_his = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G_images_D_logit, labels= tf.zeros_like(self.G_images_D_logit)))
                    self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_fake_his

                    self.G_loss_adversarial = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G_images_D_logit, labels= tf.ones_like(self.G_images_D_logit)))
                else:

                    if self.is_random_gan_labels:
                        self.d_loss_real     = norm_loss(self.images_D_logit,  tf.ones_like(self.images_D_logit) + tf.random_uniform(shape = self.images_D_logit.get_shape(),       minval=-0.1, maxval=0.1, dtype=tf.float32), loss_type = 'l2')
                        #self.d_loss_fake     = norm_loss(self.G_images_D_logit,                                    tf.random_uniform(shape = self.G_images_D_logit.get_shape(),     minval=-0.2, maxval=0.2, dtype=tf.float32), loss_type = 'l2')
                        #self.d_loss_fake_his = norm_loss(self.G_images_his_D_logit,                                tf.random_uniform(shape = self.G_images_his_D_logit.get_shape(), minval=-0.2, maxval=0.2, dtype=tf.float32), loss_type = 'l2')
                        
                    else:
                        self.d_loss_real = norm_loss(self.images_D_logit,       tf.ones_like(self.images_D_logit),        loss_type = 'l2')
                    self.d_loss_fake     = norm_loss(self.G_images_D_logit,     tf.zeros_like(self.G_images_D_logit),     loss_type = 'l2')
                    self.d_loss_fake_his = norm_loss(self.G_images_his_D_logit, tf.zeros_like(self.G_images_his_D_logit), loss_type = 'l2')
                
                    self.d_loss = (self.d_loss_real*2 + self.d_loss_fake + self.d_loss_fake_his)/4
                    self.G_loss_adversarial = norm_loss(self.G_images_D_logit, tf.ones_like(self.G_images_D_logit), loss_type = 'l2')

                self.g_loss = self.g_loss + self.G_loss_adversarial
            '''




            if self.is_using_landmark:
                g_loss_wlandmark = G_landmark_loss + g_loss
            else:
                g_loss_wlandmark = g_loss


            return g_loss, g_loss_wlandmark, G_loss_m, G_loss_shape, G_loss_texture, G_loss_recon, G_loss_recon_base, G_loss_recon_mix_s1a0, G_loss_recon_mix_s0a1, G_loss_frecon, G_loss_graddiff, G_landmark_loss, \
                G_loss_symetry, G_loss_res_symetry, G_loss_white_shading, G_loss_albedo_const, G_loss_smoothness, G_loss_local_albedo_const, G_loss_res, G_loss_shade_mag, \
                G_images_300W, G_images_base_300W, G_images_mix_s1a0_300W, G_images_mix_s0a1_300W, texture_300W, texture_base_300W, texture_mix_s1a0_300W, texture_mix_s0a1_300W, albedo_300W, albedo_base_300W, shade_300W, shade_base_300W, texture_300W_labels, input_images_300W

        g_loss, g_loss_wlandmark, G_loss_m, G_loss_shape, G_loss_texture, G_loss_recon, G_loss_recon_base, G_loss_recon_mix_s1a0, G_loss_recon_mix_s0a1, G_loss_frecon, G_loss_graddiff, G_landmark_loss, \
            G_loss_symetry, G_loss_res_symetry, G_loss_white_shading, G_loss_albedo_const, G_loss_smoothness, G_loss_local_albedo_const, G_loss_res, G_loss_shade_mag, \
            G_images_300W, G_images_base_300W, G_images_mix_s1a0_300W, G_images_mix_s0a1_300W, texture_300W, texture_base_300W, texture_mix_s1a0_300W, texture_mix_s0a1_300W, albedo_300W, albedo_base_300W, shade_300W, shade_base_300W, texture_300W_labels, input_images_300W \
            = make_parallel(model_and_loss, self.gpu_num, 
                            input_images_fn_300W= self.input_images_fn_300W, input_masks_fn_300W=self.input_masks_fn_300W,
                            texture_labels_fn_300W=self.texture_labels_fn_300W, texture_masks_fn_300W=self.texture_masks_fn_300W,
                            input_offset_height=self.input_offset_height, input_offset_width=self.input_offset_width,
                            m_300W_labels = self.m_300W_labels, shape_300W_labels=self.shape_300W_labels, 
                            albedo_indexes_x1= self.albedo_indexes_x1, albedo_indexes_y1 = self.albedo_indexes_y1,
                            albedo_indexes_x2=self.albedo_indexes_x2, albedo_indexes_y2 = self.albedo_indexes_y2)


        #with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        #   var_name = 'w_u'
        self.w_u = tf.zeros([]) #get_variable(var_name)

        t_vars = tf.trainable_variables()
        weights_vars = [var for var in t_vars if ('conv/w' in var.name or 'Matrix' in var.name or 'kernel' in var.name) ]
        print('weights_vars')
        for var in weights_vars:
            print(var.name)


        self.regu_loss = tf.add_n([ tf.nn.l2_loss(var) for var in weights_vars ]) * 1e-6

        self.g_loss = tf.reduce_mean(g_loss) #+ self.regu_loss
        self.g_loss_wlandmark = tf.reduce_mean(g_loss_wlandmark)
        self.G_loss_m = tf.reduce_mean(G_loss_m)
        self.G_loss_shape =  tf.reduce_mean(G_loss_shape)
        self.G_loss_texture =  tf.reduce_mean(G_loss_texture)
        self.G_loss_recon =  tf.reduce_mean(G_loss_recon)
        self.G_loss_recon_base =  tf.reduce_mean(G_loss_recon_base)
        self.G_loss_recon_mix_s1a0 =  tf.reduce_mean(G_loss_recon_mix_s1a0)
        self.G_loss_recon_mix_s0a1 =  tf.reduce_mean(G_loss_recon_mix_s0a1)
        self.G_loss_frecon =  tf.reduce_mean(G_loss_frecon)
        self.G_loss_graddiff =  tf.reduce_mean(G_loss_graddiff)
        self.G_landmark_loss =  tf.reduce_mean(G_landmark_loss)
        self.G_loss_symetry =  tf.reduce_mean(G_loss_symetry)
        self.G_loss_res_symetry =  tf.reduce_mean(G_loss_res_symetry)
        self.G_loss_white_shading =  tf.reduce_mean(G_loss_white_shading)
        self.G_loss_albedo_const =  tf.reduce_mean(G_loss_albedo_const)
        self.G_loss_res =  tf.reduce_mean(G_loss_res)
        self.G_loss_local_albedo_const =  tf.reduce_mean(G_loss_local_albedo_const)
        self.G_loss_smoothness =  tf.reduce_mean(G_loss_smoothness)
        self.G_loss_shade_mag = tf.reduce_mean(G_loss_shade_mag)

        self.G_images_300W = tf.clip_by_value(tf.concat(G_images_300W, axis=0), -1, 1)
        self.G_images_mix_s1a0_300W = tf.clip_by_value(tf.concat(G_images_mix_s1a0_300W, axis=0), -1, 1)
        self.G_images_mix_s0a1_300W = tf.clip_by_value(tf.concat(G_images_mix_s0a1_300W, axis=0), -1, 1)        
        self.G_images_base_300W = tf.clip_by_value(tf.concat(G_images_base_300W, axis=0), -1, 1)

        self.texture_300W = tf.clip_by_value(tf.concat(texture_300W, axis=0), -1, 1)
        self.texture_base_300W = tf.clip_by_value(tf.concat(texture_base_300W, axis=0), -1, 1)
        self.texture_mix_s1a0_300W = tf.clip_by_value(tf.concat(texture_mix_s1a0_300W, axis=0), -1, 1)
        self.texture_mix_s0a1_300W = tf.clip_by_value(tf.concat(texture_mix_s0a1_300W, axis=0), -1, 1)

        self.albedo_300W = tf.concat(albedo_300W, axis=0)
        self.albedo_base_300W = tf.concat(albedo_base_300W, axis=0)
        self.shade_300W = tf.concat(shade_300W, axis=0)
        self.shade_base_300W = tf.concat(shade_base_300W, axis=0)
        self.texture_300W_labels = tf.concat(texture_300W_labels, axis=0)
        self.input_images_300W = tf.concat(input_images_300W, axis=0)





        '''
        if self.is_using_GAN:
            self.input_good_images_300W = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='input_good_images_300W')
            self.G_images_300W_his = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='history_images_300W')

            if self.is_combine:
                self.all_D_logits = self.discriminator( tf.concat( [self.input_good_images_300W, self.G_images_300W, self.G_images_300W_his], axis = 0 ), is_reuse = False )
                self.images_D_logit, self.G_images_D_logit, self.G_images_his_D_logit = tf.split(self.all_D_logits, num_or_size_splits = 3)
            else:
                self.images_D_logit  = self.discriminator(self.input_good_images_300W, is_reuse = False)
                self.G_images_D_logit     = self.discriminator(self.G_images_300W,     is_reuse = True)
                self.G_images_his_D_logit = self.discriminator(self.G_images_300W_his, is_reuse = True)
        '''



        # Sampler
        #self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim], name='sample_images')
        #self.sample_m_ph  = tf.placeholder(tf.float32, [self.sample_size, self.mDim], name='mm')
        #self.sample_shape_ph = tf.placeholder(tf.float32, [self.sample_size, self.vertexNum * 3], name='ssss')
        #self.sample_texture_ph = tf.placeholder(tf.float32, [self.sample_size, self.texture_size, self.texture_size, self.c_dim], name='tttt')

        #self.landmark_u, self.landmark_v = compute_landmarks(self.sample_m_ph, self.sample_shape_ph, output_size=self.output_size, is_reduce = self.is_reduce)

        #self.s_shape, self.s_shade, self.s_shaded_texture, self.s_texture, self.s_m, self.s_img, self.s_overlay_img = self.sampler(self.sample_images)
  
       
        
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.g_vars_v6 = [var for var in self.g_vars if 'v6' in var.name]

        self.g_en_vars = [var for var in t_vars if 'g_k' in var.name]
        self.g_tex_de_vars = [var for var in t_vars if 'g_h' in var.name]
        self.g_shape_de_vars = [var for var in t_vars if 'g_s' in var.name]

    
    def setupParaStat(self):

        if self.is_reduce:
            self.tri = load_3DMM_tri_reduce()
            self.vertex_tri = load_3DMM_vertex_tri_reduce()
            self.vt2pixel_u, self.vt2pixel_v = load_FaceAlignment_vt2pixel_reduce()
            self.uv_tri, self.uv_mask = load_FaceAlignment_tri_2d_reduce(with_mask = True)
        else:
            self.tri = load_3DMM_tri()
            self.vertex_tri = load_3DMM_vertex_tri()
            self.vt2pixel_u, self.vt2pixel_v = load_FaceAlignment_vt2pixel()
            self.uv_tri, self.uv_mask = load_FaceAlignment_tri_2d(with_mask = True)

        
        


        # Basis
        mu_shape, w_shape = load_FaceAlignment_basic('shape', is_reduce = self.is_reduce)
        mu_exp, w_exp = load_FaceAlignment_basic('exp', is_reduce = self.is_reduce)

        self.mean_shape = mu_shape + mu_exp
        if self.is_2d_normalize:
            #self.mean_shape = np.tile(np.array([0, 0, 6e4]), VERTEX_NUM)
            self.std_shape = np.tile(np.array([1e4, 1e4, 1e4]), self.vertexNum)
        else:
            #self.mean_shape = np.load('mean_shape.npy')
            self.std_shape  = np.load('std_shape.npy')

        self.mean_shape_tf = tf.constant(self.mean_shape, tf.float32)
        self.std_shape_tf = tf.constant(self.std_shape, tf.float32)

        self.mean_m = np.load('mean_m.npy')
        self.std_m = np.load('std_m.npy')

        self.mean_m_tf = tf.constant(self.mean_m, tf.float32)
        self.std_m_tf = tf.constant(self.std_m, tf.float32)
        
        self.w_shape = w_shape
        self.w_exp = w_exp

        #mu_tex, w_tex = load_FaceAlignment_2dbasic('tex')
        ##mu_tex = mu_tex.reshape(-1, 128, 128, 3)

        #self.mu_tex = mu_tex#/127.5-1
        
        #self.w_tex = w_tex#/127.5-1

        #self.mu_tex_const = tf.constant(mu_tex.reshape(-1, 128, 128, 3)/127.5-1 )

    def m2full(self, m):
        return m * self.std_m_tf + self.mean_m_tf

    def shape2full(self, shape):
        return shape * self.std_shape_tf + self.mean_shape_tf


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
            #images_AFLW2000, pid_AFLW2000, m_AFLW2000, pose_AFLW2000, shape_AFLW2000, exp_AFLW2000, tex_para_AFLW2000, _, = load_FaceAlignment_dataset_recrop_sz224('AFW', with_sh = False)

            images[i], pid[i], m[i], pose[i], shape[i], exp[i], tex_para[i], _, = load_FaceAlignment_dataset_recrop_sz224(dataset[i], False)  #_, tex[i], il[i], alb[i], mask[i]


        self.images_300W   = np.concatenate(images, axis=0)
        images = None
        #self.textures_300W = np.concatenate(tex, axis=0)
        #tex = None
        #self.albedos_300W = np.concatenate(alb, axis=0)
        #alb = None
        #self.masks_300W = np.concatenate(mask, axis=0)
        #mask = None


        all_m = np.concatenate(m, axis=0)

        #self.mean_m = np.mean(all_m, axis=0)
        #self.std_m  = np.std(all_m, axis=0)

        #np.save('mean_m.npy', self.mean_m)
        #np.save('std_m.npy', self.std_m)

        #print("Save mean m--------------------------------")


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

        '''
        # Basis
        if self.is_remesh:
            mu_shape, w_shape = load_FaceAlignment_basic_remesh6k('shape')
            mu_exp, w_exp = load_FaceAlignment_basic_remesh6k('exp')
        else:
            mu_shape, w_shape = load_FaceAlignment_basic('shape')
            mu_exp, w_exp = load_FaceAlignment_basic('exp')


        var_shape_para = np.var(all_shape_para, axis=0)
        w_shape_square = np.square(w_shape)

        var_exp_para = np.var(all_exp_para, axis=0)
        w_exp_square = np.square(w_exp)


        var_shape = np.matmul(w_shape_square, var_shape_para) + np.matmul(w_exp_square, var_exp_para)
        self.std_shape =np.sqrt(var_shape)
        self.mean_shape = mu_shape+mu_exp

        if self.is_remesh:
            np.save('std_shape_remesh6k.npy', self.std_shape)
            np.save('mean_shape_remesh6k.npy', self.mean_shape)
        '''


        

    def setupValData(self):
        return
        # Samples data - AFLW200
        self.images_AFLW2000, pid_AFLW2000, m_AFLW2000, pose_AFLW2000, shape_AFLW2000, exp_AFLW2000, _, _, _, self.tex_AFLW2000 = load_FaceAlignment_dataset_recrop('AFLW2000')
        self.AFLW2000_m = np.divide(np.subtract(m_AFLW2000, self.mean_m), self.std_m)
        self.AFLW2000_shape_para  = shape_AFLW2000
        self.AFLW2000_exp_para    = exp_AFLW2000

    
 


    def train(self, config):
        np.random.seed(0)
        tf.set_random_seed(0)

        # Training data
        self.setupTrainingData()
        

        valid_300W_idx = range(self.images_300W.shape[0])
        print("Valid images %d/%d" % ( len(valid_300W_idx), self.images_300W.shape[0] ))



        np.random.shuffle(valid_300W_idx)


        # Pairwise preparing
        num_300W = len(self.pids_300W)
        self.is_using_pairwise = False
        if self.is_using_pairwise:
            id_dict = {}
            for i in range(num_300W):
                pid = int(self.pids_300W[i])
                if (pid not in id_dict):          
                    id_dict[pid] = []
                id_dict[pid].append(i)


        # Optimizer      
        #elf.setupLossFunctions()
        print('Optim start')
        history = np.zeros(1)    
        if self.is_using_GAN:    
            real_indices = load_FaceAlignment_datasets_good()
            real_id_dict = {}
            for i in range(num_300W):

                if real_indices[i]:
                    pid = int(self.pids_300W[i])
                    if (pid not in real_id_dict):          
                        real_id_dict[pid] = []
                    real_id_dict[pid].append(i)
            real_indices = [int(i) for i in range(real_indices.shape[0]) if ( real_indices[i])]
            

            # History
            if os.path.exists(self.history_file):
                history = np.load(self.history_file)
            else:
                history = np.zeros([HISTRORY_SZ, self.output_size, self.output_size, self.c_dim], dtype=np.float32)



            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars, colocate_gradients_with_ops=True)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars, colocate_gradients_with_ops=True)
        g_en_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss_wlandmark, var_list=self.g_en_vars, colocate_gradients_with_ops=True)

        tf.global_variables_initializer().run()
        #self.g_global_vars = [var for var in tf.global_variables() if not ('v6' in var.name) ] # and 'w_v' not in var.name]
        #self.saver = tf.train.Saver(self.g_global_vars, keep_checkpoint_every_n_hours=1, max_to_keep = 0)
        self.saver = tf.train.Saver(tf.trainable_variables(), keep_checkpoint_every_n_hours=1, max_to_keep = 10) #tf.trainable_variables(),
        #self.saver = tf.train.Saver([var for var in tf.global_variables() if 'res' not in var.name], keep_checkpoint_every_n_hours=1, max_to_keep = 10)

        
        """Train DCGAN"""
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch0 = checkpoint_counter + 1
            print(" [*] Load SUCCESS")
        else:
            epoch0 = 1
            print(" [!] Load failed...")



        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep = 10)

        
        



        start_time = time.time()
        
        for epoch in xrange(epoch0, config.epoch):
            
            batch_idxs = min(len(valid_300W_idx), config.train_size) // config.batch_size
            
            print('--------------------------------')
            for idx in xrange(0, batch_idxs):

                batch_idx = valid_300W_idx[idx*config.batch_size:(idx+1)*config.batch_size] #valid_300W_idx[idx*config.batch_size:(idx+1)*config.batch_size]
                if self.is_using_pairwise:
                    for i in range( int(config.batch_size/2)) :
                        pid = int(self.pids_300W[ batch_idx[i] ])
                        idx_list = id_dict.get(pid)
                        batch_idx[int(config.batch_size/2) + i] = idx_list[random.randint(0, len(idx_list)-1)]

                batch_300W_images_fn = [self.images_300W[batch_idx[i]] for i in range(config.batch_size)] 
                
                tx = np.random.random_integers(0, 32, size=config.batch_size)
                ty = np.random.random_integers(0, 32, size=config.batch_size)
               

                delta_m      = np.zeros([config.batch_size, 8]);
                delta_m[:,6] = np.divide(ty, self.std_m[6]);
                delta_m[:,7] = np.divide(32 - tx, self.std_m[7]);

                
                batch_m      = self.all_m[batch_idx,:] - delta_m;


                #batch_il     = self.all_il[batch_idx,:]

                batch_shape_para = self.all_shape_para[batch_idx,:]
                batch_exp_para   = self.all_exp_para[batch_idx,:]

                batch_tex_para   = self.all_tex_para[batch_idx,:]

                
                batch_shape  = np.divide( np.matmul(batch_shape_para, np.transpose(self.w_shape)) + np.matmul(batch_exp_para, np.transpose(self.w_exp)), self.std_shape)

                
                ffeed_dict={ self.m_300W_labels: batch_m, self.shape_300W_labels: batch_shape, self.input_offset_height: tx, self.input_offset_width: ty}
                for i in range(self.batch_size):
                    ffeed_dict[self.input_images_fn_300W[i]] = DATA_DIR + 'image/'+ batch_300W_images_fn[i]
                    ffeed_dict[self.input_masks_fn_300W[i]] = DATA_DIR + 'mask_img/'+ batch_300W_images_fn[i]
                    ffeed_dict[self.texture_labels_fn_300W[i]] = DATA_DIR + 'texture/'+ image2texture_fn(batch_300W_images_fn[i])
                    ffeed_dict[self.texture_masks_fn_300W[i]] = DATA_DIR + 'mask/'+ image2texture_fn(batch_300W_images_fn[i])

                if self.is_const_albedo:
                    indexes1 = np.random.randint(low=0, high=self.const_alb_mask.shape[0], size=[self.batch_size* CONST_PIXELS_NUM])
                    indexes2 = np.random.randint(low=0, high=self.const_alb_mask.shape[0], size=[self.batch_size* CONST_PIXELS_NUM])


                    ffeed_dict[self.albedo_indexes_x1] = np.reshape(self.const_alb_mask[indexes1, 1], [self.batch_size, CONST_PIXELS_NUM, 1])
                    ffeed_dict[self.albedo_indexes_y1] = np.reshape(self.const_alb_mask[indexes1, 0], [self.batch_size, CONST_PIXELS_NUM, 1])
                    ffeed_dict[self.albedo_indexes_x2] = np.reshape(self.const_alb_mask[indexes2, 1], [self.batch_size, CONST_PIXELS_NUM, 1])
                    ffeed_dict[self.albedo_indexes_y2] = np.reshape(self.const_alb_mask[indexes2, 0], [self.batch_size, CONST_PIXELS_NUM, 1])



                '''

                self.sess.run(g_en_optim, feed_dict=ffeed_dict)

                _, g_loss, G_loss_m, G_loss_shape, G_loss_texture, G_loss_recon, G_loss_recon_base, G_loss_frecon, G_loss_graddiff, G_landmark_loss, \
                    G_loss_symetry, G_loss_white_shading, G_loss_albedo_const, G_loss_local_albedo_const, G_loss_smoothness, G_loss_regu, G_loss_res, w_u, \
                    albedo_300W, albedo_base_300W, texture_300W, shade_300W, shade_base_300W, G_images_300W, G_images_base_300W, batch_300W_textures, batch_300W_images = \
                            self.sess.run([g_optim, self.g_loss_wlandmark, self.G_loss_m, self.G_loss_shape, self.G_loss_texture, self.G_loss_recon, self.G_loss_recon_base, self.G_loss_frecon, self.G_loss_graddiff, self.G_landmark_loss, \
                                                     self.G_loss_symetry, self.G_loss_white_shading, self.G_loss_albedo_const, self.G_loss_local_albedo_const, self.G_loss_smoothness, self.regu_loss, self.G_loss_res, self.w_u, \
                                                    self.albedo_300W, self.albedo_base_300W, self.texture_300W, self.shade_300W, self.shade_base_300W, self.G_images_300W, self.G_images_base_300W, self.texture_300W_labels, self.input_images_300W], feed_dict=ffeed_dict)
                            
                print("Epoch [%2d][%4d/%4d] time: %4.0fs, G_loss:%2.4f (m:%.4f, shape:%.4f, tex:%.4f, recon:%.4f, base_recon:%.4f , frecon:%.4f, graddiff:%.4f, land:%.4f, sym:%.4f, wshade:%.4f, alb_const:%.4f, alb_l_const:%.4f, smooth:%.4f, res:%.4f, regu:%.4f)"\
                    % (epoch, idx, batch_idxs, time.time() - start_time, g_loss, G_loss_m, G_loss_shape, G_loss_texture, G_loss_recon, G_loss_recon_base, G_loss_frecon, G_loss_graddiff, G_landmark_loss, G_loss_symetry, G_loss_white_shading, G_loss_albedo_const, G_loss_local_albedo_const, G_loss_smoothness, G_loss_res, G_loss_regu))

                

                if np.mod(idx+1, 2) == 0:
                    save_images(G_images_300W, [-1, 8], './{:s}/{:s}/pred_image_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(G_images_base_300W, [-1, 8], './{:s}/{:s}/pred_image_{:02d}_{:06d}_base.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    #save_images(batch_300W_textures, [-1, 8], './{:s}/{:s}/texture_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(batch_300W_images, [-1, 8], './{:s}/{:s}/pred_image_{:02d}_{:06d}_in.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(texture_300W, [-1, 8], './{:s}/{:s}/pred_shaded_texture_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    #save_images(texture_300W, [8, 8], './{:s}/{:s}/pred_texture_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(albedo_300W, [-1, 8], './{:s}/{:s}/pred_albedo_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(albedo_base_300W, [-1, 8], './{:s}/{:s}/pred_albedo_{:02d}_{:06d}_base.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(shade_300W/2, [-1, 8], './{:s}/{:s}/pred_shade_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx), inverse = False)
                    save_images(shade_base_300W/2, [-1, 8], './{:s}/{:s}/pred_shade_{:02d}_{:06d}_base.png'.format(self.samples_dir, self.model_dir, epoch, idx), inverse = False)

                if np.mod(idx+1, 100) == 0:
                    self.save(config.checkpoint_dir, epoch)
                '''

                
                

                if not self.is_using_GAN:
                    #Not using GAN

                    if np.mod(idx, 25) == 0:
                        _, g_loss, G_loss_m, G_loss_shape, G_loss_texture, G_loss_recon, G_loss_recon_base, G_loss_recon_mix_s1a0, G_loss_recon_mix_s0a1, G_loss_frecon, G_loss_graddiff, G_landmark_loss, \
                            G_loss_symetry, G_loss_res_symetry, G_loss_white_shading, G_loss_albedo_const, G_loss_local_albedo_const, G_loss_smoothness, G_loss_regu, G_loss_res, G_loss_shade_mag, w_u = \
                            self.sess.run([g_en_optim, self.g_loss_wlandmark, self.G_loss_m, self.G_loss_shape, self.G_loss_texture, self.G_loss_recon, self.G_loss_recon_base, self.G_loss_recon_mix_s1a0, self.G_loss_recon_mix_s0a1, self.G_loss_frecon, self.G_loss_graddiff, self.G_landmark_loss, \
                                                     self.G_loss_symetry, self.G_loss_res_symetry, self.G_loss_white_shading, self.G_loss_albedo_const, self.G_loss_local_albedo_const, self.G_loss_smoothness, self.regu_loss, self.G_loss_res, self.G_loss_shade_mag, self.w_u],
                            feed_dict=ffeed_dict)
                        print("Epoch [%2d][%4d/%4d] time: %4.0fs, G_loss:%2.4f (m:%.4f, shape:%.4f, tex:%.4f, recon:%.4f, base_r:%.4f, mix_s1a0_r:%.4f, mix_s0a1_r:%.4f, frecon:%.4f, graddiff:%.4f, land:%.4f, sym:%.4f, res_sym:%.4f, wshade:%.4f, magshade:%.4f, alb_const:%.4f, alb_l_const:%.4f, smooth:%.4f, res:%.4f, regu:%.4f)"\
                    % (epoch, idx, batch_idxs, time.time() - start_time, g_loss, G_loss_m, G_loss_shape, G_loss_texture, G_loss_recon, G_loss_recon_base, G_loss_recon_mix_s1a0, G_loss_recon_mix_s0a1, G_loss_frecon, G_loss_graddiff, G_landmark_loss, G_loss_symetry, G_loss_res_symetry, G_loss_white_shading, G_loss_shade_mag, G_loss_albedo_const, G_loss_local_albedo_const, G_loss_smoothness, G_loss_res, G_loss_regu))

                    self.sess.run([g_en_optim], feed_dict=ffeed_dict)
                    self.sess.run([g_optim], feed_dict=ffeed_dict)
                    
                else:
                    # Good images for GAN
                    real_batch_idx = np.zeros(config.batch_size)
                    for i in range( int(config.batch_size)) :
                        pid = int(self.pids_300W[ batch_idx[i] ])
                        idx_list = real_id_dict.get(pid)
                        real_batch_idx[i] = idx_list[random.randint(0, len(idx_list)-1)]
                        

                    batch_300W_real_images = [crop(self.images_300W[ int(real_batch_idx[i]),:,:,:], self.output_size, self.output_size, tx[i], ty[i]) for i in range(config.batch_size)]
                    batch_300W_real_images = np.array(batch_300W_real_images).astype(np.float32)/127.5-1

                    his_batch_idx = np.random.random_integers(low = 0, high=HISTRORY_SZ - 1, size=config.batch_size)
                    batch_300W_history_images = history[his_batch_idx, :,:,:]

                    ffeed_dict[self.input_good_images_300W]=batch_300W_real_images
                    ffeed_dict[self.G_images_300W_his] = batch_300W_history_images

                    if np.mod(idx, 10) == 0:
                        _, G_images_300W, d_loss, d_loss_real, d_loss_fake, g_loss, G_loss_adversarial, G_loss_m, G_loss_il, G_loss_shape, G_loss_texture, G_loss_recon, G_landmark_loss, G_loss_symetry, G_loss_albedo, G_loss_white_shading, G_loss_albedo_const, G_loss_local_albedo_const, G_loss_smoothness, w_u = \
                            self.sess.run([g_en_optim, self.G_images_300W, self.d_loss, self.d_loss_real, self.d_loss_fake, \
                             self.g_loss_wlandmark,  self.G_loss_adversarial, self.G_loss_m, self.G_loss_il, self.G_loss_shape, self.G_loss_texture, self.G_loss_recon, self.G_landmark_loss, self.G_loss_symetry, self.G_loss_albedo, self.G_loss_local_albedo_const, self.G_loss_white_shading, self.G_loss_albedo_const, self.G_loss_smoothness, self.w_u],
                            feed_dict=ffeed_dict)
                        print(w_u[50][50:51])
                        print("Epoch [%2d][%4d/%4d] time: %4.0fs, D_loss: %.4f (r: %.4f, f: %.4f), G_loss: %.4f (ad: %.4f, m: %.4f, il:%.4f, shape: %.4f, alb: %.4f, tex: %.4f, recon: %.4f, land: %4f, sym: %.4f, wshade: %.4f, alb_const: %.4f, alb_l_const: %.4f, smooth: %.4f)" \
                            % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, d_loss_real, d_loss_fake,  g_loss, G_loss_adversarial,  G_loss_m, G_loss_il, G_loss_shape, G_loss_albedo, G_loss_texture, G_loss_recon, G_landmark_loss, G_loss_symetry, G_loss_white_shading, G_loss_albedo_const, G_loss_local_albedo_const, G_loss_smoothness))

                    elif np.mod(idx, 2) == 0:
                        # Update D, G
                        _                = self.sess.run([d_optim], feed_dict=ffeed_dict)
                        _, G_images_300W = self.sess.run([g_optim, self.G_images_300W], feed_dict=ffeed_dict)
                    else:
                        # Update D, G encoder only
                        _                = self.sess.run([d_optim], feed_dict=ffeed_dict)
                        _, G_images_300W = self.sess.run([g_en_optim, self.G_images_300W], feed_dict=ffeed_dict)

                    history[his_batch_idx,:,:,:] = G_images_300W


                
                    
                
                if np.mod(idx+1, 500) == 0:
                    self.save(config.checkpoint_dir, epoch*10000+idx+1)

                    albedo_300W, albedo_base_300W, texture_300W, texture_base_300W, texture_mix_s0a1_300W, texture_mix_s1a0_300W, shade_300W, shade_base_300W, G_images_300W, G_images_base_300W, G_images_mix_s1a0_300W, G_images_mix_s0a1_300W, batch_300W_textures, batch_300W_images= \
                        self.sess.run([self.albedo_300W, self.albedo_base_300W, self.texture_300W, self.texture_base_300W, self.texture_mix_s0a1_300W, self.texture_mix_s1a0_300W, self.shade_300W, self.shade_base_300W, self.G_images_300W, self.G_images_base_300W, self.G_images_mix_s1a0_300W, self.G_images_mix_s0a1_300W, self.texture_300W_labels, self.input_images_300W], feed_dict=ffeed_dict)

                    save_images(G_images_300W, [-1, 8], './{:s}/{:s}/pred_image_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(G_images_base_300W, [-1, 8], './{:s}/{:s}/pred_image_{:02d}_{:06d}_base.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(G_images_mix_s1a0_300W, [-1, 8], './{:s}/{:s}/pred_image_{:02d}_{:06d}_base_s1a0.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(G_images_mix_s0a1_300W, [-1, 8], './{:s}/{:s}/pred_image_{:02d}_{:06d}_base_s0a1.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(batch_300W_textures, [-1, 8], './{:s}/{:s}/texture_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(batch_300W_images, [-1, 8], './{:s}/{:s}/pred_image_{:02d}_{:06d}_in.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(texture_300W, [-1, 8], './{:s}/{:s}/pred_shaded_texture_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(texture_base_300W, [-1, 8], './{:s}/{:s}/pred_shaded_texture_{:02d}_{:06d}_base.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(texture_mix_s1a0_300W, [-1, 8], './{:s}/{:s}/pred_shaded_texture_{:02d}_{:06d}_base_s1a0.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(texture_mix_s0a1_300W, [-1, 8], './{:s}/{:s}/pred_shaded_texture_{:02d}_{:06d}_base_s0a1.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(albedo_300W, [-1, 8], './{:s}/{:s}/pred_albedo_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(albedo_base_300W, [-1, 8], './{:s}/{:s}/pred_albedo_{:02d}_{:06d}_base.png'.format(self.samples_dir, self.model_dir, epoch, idx))
                    save_images(shade_300W/2, [-1, 8], './{:s}/{:s}/pred_shade_{:02d}_{:06d}.png'.format(self.samples_dir, self.model_dir, epoch, idx), inverse = False)
                    save_images(shade_base_300W/2, [-1, 8], './{:s}/{:s}/pred_shade_{:02d}_{:06d}_base.png'.format(self.samples_dir, self.model_dir, epoch, idx), inverse = False)

                    np.save(self.history_file, history)
                

                    


            self.save(config.checkpoint_dir, epoch)

                
            #print('Start warping')
            '''
            s_shape, s_shade, s_shaded_texture, s_texture, s_m, s_img  = self.sess.run( [ self.s_shape, self.s_shade, self.s_shaded_texture, self.s_texture, self.s_m, self.s_overlay_img], feed_dict={ self.sample_images: sample_images})
            save_images(s_texture, [8, 8], './{:s}/{:s}/pred_texture_{:02d}.png'.format(self.samples_dir, self.model_dir, epoch))
            save_images(s_shaded_texture, [8, 8], './{:s}/{:s}/pred_shaded_texture_{:02d}.png'.format(self.samples_dir, self.model_dir, epoch))
            save_images(s_shade/2, [8, 8], './{:s}/{:s}/pred_shade_{:02d}.png'.format(self.samples_dir, self.model_dir, epoch), inverse = False)
            save_images(s_img, [8, 8], './{:s}/{:s}/pred_warped_texture_{:02d}.png'.format(self.samples_dir, self.model_dir, epoch))

            landmark_val_loss, ssim_val_loss = self.evaluation()
            print('  |')
            print('  |   Evaluation: SSIM: %2.2f ' % (ssim_val_loss))
            print('  |   Evaluation: landmark_NME: %2.2f ' % (landmark_val_loss * 100))
            print('  |')
            print('  |')
            '''
                            

    

    def generator_encoder(self, image,  is_reuse=False, is_training = True):
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

        k2_0 = elu(self.g_bn2_0(conv2d(k1_2, self.gf_dim*4, d_h=2, d_w =2, use_bias = False, name='g_k20_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k2_1 = elu(self.g_bn2_1(conv2d(k2_0, self.gf_dim*3, d_h=1, d_w =1, use_bias = False, name='g_k21_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k2_2 = elu(self.g_bn2_2(conv2d(k2_1, self.gf_dim*6, d_h=1, d_w =1, use_bias = False, name='g_k22_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))

        k3_0 = elu(self.g_bn3_0(conv2d(k2_2, self.gf_dim*6, d_h=2, d_w =2, use_bias = False, name='g_k30_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k3_1 = elu(self.g_bn3_1(conv2d(k3_0, self.gf_dim*4, d_h=1, d_w =1, use_bias = False, name='g_k31_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k3_2 = elu(self.g_bn3_2(conv2d(k3_1, self.gf_dim*8, d_h=1, d_w =1, use_bias = False, name='g_k32_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))

        k4_0 = elu(self.g_bn4_0(conv2d(k3_2, self.gf_dim*8, d_h=2, d_w =2, use_bias = False, name='g_k40_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k4_1 = elu(self.g_bn4_1(conv2d(k4_0, self.gf_dim*5, d_h=1, d_w =1, use_bias = False, name='g_k41_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        

        # M
        k51_m = self.g_bn5_m(    conv2d(k4_1, int(self.gfc_dim/5),  d_h=1, d_w =1, name='g_k5_m_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k51_shape_ = get_shape(k51_m)
        k52_m = tf.nn.avg_pool(k51_m, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_m = tf.reshape(k52_m, [-1, int(self.gfc_dim/5)])
        #if (is_training):
        #    k52_m = tf.nn.dropout(k52_m, keep_prob = 0.6)
        k6_m = linear(k52_m, self.mDim, 'g_k6_m_lin', reuse = is_reuse)
        
        # Il
        k51_il = self.g_bn5_il(    conv2d(k4_1, int(self.gfc_dim/5),  d_h=1, d_w =1, name='g_k5_il_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k52_il = tf.nn.avg_pool(k51_il, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_il = tf.reshape(k52_il, [-1, int(self.gfc_dim/5)])
        #if (is_training):
        #    k52_il = tf.nn.dropout(k52_il, keep_prob = 0.6)
        k6_il = linear(k52_il, self.ilDim, 'g_k6_il_lin', reuse = is_reuse)

        # Shape
        k51_shape = self.g_bn5_shape(conv2d(k4_1, self.gfc_dim/2,  d_h=1, d_w =1, name='g_k5_shape_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k52_shape = tf.nn.avg_pool(k51_shape, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_shape = tf.reshape(k52_shape, [-1, int(self.gfc_dim/2)])
        #if (is_training):
        #    k52_shape = tf.nn.dropout(k52_shape, keep_prob = 0.6)

        k51_tex   = self.g_bn5_tex(  conv2d(k4_1, self.gfc_dim/2,  d_h=1, d_w =1, name='g_k5_tex_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k52_tex = tf.nn.avg_pool(k51_tex, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_tex = tf.reshape(k52_tex, [-1, int(self.gfc_dim/2)])
        #if (is_training):
        #    k52_tex = tf.nn.dropout(k52_tex, keep_prob = 0.6)

        '''
        if self.is_using_linear:
            k51_shape_linear = self.g_bn5_shape_linear(conv2d(k4_1, int(self.gfc_dim/2),  d_h=1, d_w =1, name='g_k5_shape_linear_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
            k52_shape_linear = tf.nn.avg_pool(k51_shape_linear, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
            k52_shape_linear = tf.reshape(k52_shape_linear, [-1, int(self.gfc_dim/2)])

            k6_shape_linear = linear(k52_shape_linear, self.shapeDim + self.expDim, 'g_k6_shape_linear_lin', reuse = is_reuse)
        else:
            k6_shape_linear = 0
        '''


        return k52_shape, k52_tex, k6_m, k6_il#, k6_shape_linear


    '''
    def generator_encoder_v1(self, image,  is_reuse=False, is_training = True):
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

        k2_0 = elu(self.g_bn2_0(conv2d(k1_2, self.gf_dim*4, d_h=2, d_w =2, use_bias = False, name='g_k20_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k2_1 = elu(self.g_bn2_1(conv2d(k2_0, self.gf_dim*3, d_h=1, d_w =1, use_bias = False, name='g_k21_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k2_2 = elu(self.g_bn2_2(conv2d(k2_1, self.gf_dim*6, d_h=1, d_w =1, use_bias = False, name='g_k22_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        
        k3_0 = elu(self.g_bn3_0(conv2d(k2_2, self.gf_dim*6, d_h=2, d_w =2, use_bias = False, name='g_k30_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k3_1 = elu(self.g_bn3_1(conv2d(k3_0, self.gf_dim*4, d_h=1, d_w =1, use_bias = False, name='g_k31_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        k3_2 = elu(self.g_bn3_2(conv2d(k3_1, self.gf_dim*8, d_h=1, d_w =1, use_bias = False, name='g_k32_conv', reuse = is_reuse), train=is_training, reuse = is_reuse))
        
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
        k6_il = linear(k52_m, self.ilDim, 'g_k6_il_lin', reuse = is_reuse)

        # Shape
        k51_shape = self.g_bn5_shape(conv2d(k4_1, self.gfc_dim/2,  d_h=1, d_w =1, name='g_k5_shape_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k52_shape = tf.nn.avg_pool(k51_shape, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_shape = tf.reshape(k52_shape, [-1, int(self.gfc_dim/2)])

        k51_tex   = self.g_bn5_tex(  conv2d(k4_1, self.gfc_dim/2,  d_h=1, d_w =1, name='g_k5_tex_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
        k52_tex = tf.nn.avg_pool(k51_tex, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
        k52_tex = tf.reshape(k52_tex, [-1, int(self.gfc_dim/2)])

        if self.is_using_linear:
            k51_shape_linear = self.g_bn5_shape_linear(conv2d(k4_1, int(self.gfc_dim/2),  d_h=1, d_w =1, name='g_k5_shape_linear_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
            k52_shape_linear = tf.nn.avg_pool(k51_shape_linear, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
            k52_shape_linear = tf.reshape(k52_shape_linear, [-1, int(self.gfc_dim/2)])

            k6_shape_linear = linear(k52_shape_linear, self.shapeDim + self.expDim, 'g_k6_shape_linear_lin', reuse = is_reuse)
        else:
            k6_shape_linear = 0


        return k52_shape, k52_tex, k6_m, k6_il, k6_shape_linear
    '''

    def generator_decoder_albedo(self, k52_tex, is_reuse=False, is_training=True):
        #return tf.zeros(shape = [self.batch_size, self.texture_size[0], self.texture_size[1], 3])

        if self.is_partbase_albedo:
            return self.generator_decoder_albedo_part_based_v2_relu(k52_tex, is_reuse, is_training)
        else:
            return self.generator_decoder_albedo_v1(k52_tex, is_reuse, is_training)


    def generator_decoder_shape(self, k52_shape, is_reuse=False, is_training=True, is_remesh=False):
        if False:
            return self.generator_decoder_shape_1d(k52_shape, is_reuse, is_training)
        else: 

            n_size = get_shape(k52_shape)
            n_size = n_size[0]

            if self.is_reduce:
                #tri = load_3DMM_tri_remesh6k()
                vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel_reduce()
            else:
                #tri = load_3DMM_tri()
                vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel()


            #tri2vt1_const = tf.constant(tri[0,:], tf.int32)
            #tri2vt2_const = tf.constant(tri[1,:], tf.int32)
            #tri2vt3_const = tf.constant(tri[2,:], tf.int32)

            #Vt2pix
            vt2pixel_u_const = tf.constant(vt2pixel_u[:-1], tf.float32)
            vt2pixel_v_const = tf.constant(vt2pixel_v[:-1], tf.float32)

            if self.is_partbase_albedo:
                shape_2d, shape_2d_res = self.generator_decoder_shape_2d_partbase_v2_relu(k52_shape, is_reuse, is_training)
                print('get_shape(shape_2d)')
                print(get_shape(shape_2d))
            else:
                shape_2d, shape_2d_res = self.generator_decoder_shape_2d_v1(k52_shape, is_reuse, is_training) 

            vt2pixel_v_const_ = tf.tile(tf.reshape(vt2pixel_v_const, shape =[1,1,-1]), [n_size, 1,1])
            vt2pixel_u_const_ = tf.tile(tf.reshape(vt2pixel_u_const, shape =[1,1,-1]), [n_size, 1,1])

            shape_1d = tf.reshape(bilinear_sampler( shape_2d, vt2pixel_v_const_, vt2pixel_u_const_), shape=[n_size, -1])

            shape_1d_res = tf.reshape(bilinear_sampler( shape_2d_res, vt2pixel_v_const_, vt2pixel_u_const_), shape=[n_size, -1])

            return shape_1d, shape_2d, shape_1d_res, shape_2d_res


    def generator_decoder_shape_1d(self, k52_shape, is_reuse=False, is_training=True):
        s6 = elu(self.g1_bn6(linear(k52_shape, 1000, scope= 'g_s6_lin', reuse = is_reuse), train=is_training, reuse = is_reuse), name="g_s6_prelu")
        s7 = linear(s6, self.vertexNum*3, scope= 'g_s7_lin', reuse = is_reuse)

        return s7

    def generator_decoder_shape_2d_v1(self, k52_tex, is_reuse=False, is_training=True):
        if not is_reuse:
            self.g2_bn0_1_res = batch_norm(name='g_l_bn0_1_res')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')        
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4   = batch_norm(name='g_l_bn4')
            self.g2_bn5   = batch_norm(name='g_l_bn5')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(k52_tex, self.gfc_dim*s32_h*s32_w, scope= 'g_l5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, self.gfc_dim])
        h5 = elu(self.g2_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_l4', reuse = is_reuse)
        h4_1 = elu(self.g2_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, self.gf_dim*8, strides=[1,1], name='g_l40', reuse = is_reuse)
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, self.gf_dim*8, strides=[2,2], name='g_l32', reuse = is_reuse)
        h3_2 = elu(self.g2_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, self.gf_dim*4, strides=[1,1], name='g_l31', reuse = is_reuse)
        h3_1 = elu(self.g2_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, self.gf_dim*6, strides=[1,1], name='g_l30', reuse = is_reuse)
        h3_0 = elu(self.g2_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, self.gf_dim*6, strides=[2,2], name='g_l22', reuse = is_reuse)
        h2_2 = elu(self.g2_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, self.gf_dim*3, strides=[1,1], name='g_l21', reuse = is_reuse)
        h2_1 = elu(self.g2_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, self.gf_dim*4, strides=[1,1], name='g_l20', reuse = is_reuse)
        h2_0 = elu(self.g2_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, self.gf_dim*4, strides=[2,2], name='g_l12', reuse = is_reuse)
        h1_2 = elu(self.g2_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, self.gf_dim*2, strides=[1,1], name='g_l11', reuse = is_reuse)
        h1_1 = elu(self.g2_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,self.gf_dim*2, strides=[1,1], name='g_l10', reuse = is_reuse)
        h1_0 = elu(self.g2_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, self.gf_dim*2, strides=[2,2], name='g_l02', reuse = is_reuse)
        h0_2 = elu(self.g2_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, self.gf_dim, strides=[1,1], name='g_l01', reuse = is_reuse)
        h0_1 = elu(self.g2_bn0_1(h0_1, train=is_training, reuse = is_reuse))
           
        h0 = 2*tf.nn.tanh(deconv2d(h0_1, self.c_dim, strides=[1,1], name='g_l0', reuse = is_reuse))

        h0_1_res = deconv2d(h0_2, self.gf_dim, strides=[1,1], name='g_l01_res', reuse = is_reuse)
        h0_1_res = elu(self.g2_bn0_1_res(h0_1_res, train=is_training, reuse = is_reuse))
           
        h0_res = 2*tf.nn.tanh(deconv2d(h0_1_res, self.c_dim, strides=[1,1], name='g_l0_res', reuse = is_reuse))
            
        return h0, h0_res

    def generator_decoder_shape_2d_partbase(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):
            

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1", "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w= int(s_w/8)
                s8_h= int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_l4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_l_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
                h3_1 = elu(self.bns[name + "/" + "g_l_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
                h3_0 = elu(self.bns[name + "/" + "g_l_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
                h2_2 = elu(self.bns[name + "/" + "g_l_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
                h2_1 = elu(self.bns[name + "/" "g_l_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
                h2_0 = elu(self.bns[name + "/" + "g_l_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
                h1_2 = elu(self.bns[name + "/" + "g_l_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
                h1_1 = elu(self.bns[name + "/" + "g_l_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
                h1_0 = elu(self.bns[name + "/" + "g_l_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
                h0_2 = elu(self.bns[name + "/" + "g_l_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
                h0_1 = elu(self.bns[name + "/" + "g_l_bn0_1"](h0_1, train=is_training, reuse = is_reuse))

            return h0_1

         # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)




        if not is_reuse:
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')        
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4   = batch_norm(name='g_l_bn4')
            self.g2_bn5   = batch_norm(name='g_l_bn5')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_l5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g2_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*10, name='g_l4', reuse = is_reuse)
        h4_1 = elu(self.g2_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_l40', reuse = is_reuse)
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_l32', reuse = is_reuse)
        h3_2 = elu(self.g2_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
        h3_1 = elu(self.g2_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
        h3_0 = elu(self.g2_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
        h2_2 = elu(self.g2_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
        h2_1 = elu(self.g2_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
        h2_0 = elu(self.g2_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
        h1_2 = elu(self.g2_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
        h1_1 = elu(self.g2_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
        h1_0 = elu(self.g2_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
        h0_2 = elu(self.g2_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
        h0_1 = elu(self.g2_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00', reuse = is_reuse)
        h0_0 = elu(self.g2_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_l0', reuse = is_reuse))

        return h0



    def generator_decoder_shape_2d_partbase_v2(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):
            

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1", "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w= int(s_w/8)
                s8_h= int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_l4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_l_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
                h3_1 = elu(self.bns[name + "/" + "g_l_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
                h3_0 = elu(self.bns[name + "/" + "g_l_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
                h2_2 = elu(self.bns[name + "/" + "g_l_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
                h2_1 = elu(self.bns[name + "/" "g_l_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
                h2_0 = elu(self.bns[name + "/" + "g_l_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
                h1_2 = elu(self.bns[name + "/" + "g_l_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
                h1_1 = elu(self.bns[name + "/" + "g_l_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
                h1_0 = elu(self.bns[name + "/" + "g_l_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
                h0_2 = elu(self.bns[name + "/" + "g_l_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
                h0_1 = elu(self.bns[name + "/" + "g_l_bn0_1"](h0_1, train=is_training, reuse = is_reuse))

            return h0_1

         # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)




        if not is_reuse:
            self.g2_bn0_0_res = batch_norm(name='g_l_bn0_0_res')
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')        
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4   = batch_norm(name='g_l_bn4')
            self.g2_bn5   = batch_norm(name='g_l_bn5')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_l5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g2_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_l4', reuse = is_reuse)
        h4_1 = elu(self.g2_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_l40', reuse = is_reuse)
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_l32', reuse = is_reuse)
        h3_2 = elu(self.g2_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
        h3_1 = elu(self.g2_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
        h3_0 = elu(self.g2_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
        h2_2 = elu(self.g2_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
        h2_1 = elu(self.g2_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
        h2_0 = elu(self.g2_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
        h1_2 = elu(self.g2_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
        h1_1 = elu(self.g2_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
        h1_0 = elu(self.g2_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
        h0_2 = elu(self.g2_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
        h0_1 = elu(self.g2_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00', reuse = is_reuse)
        h0_0 = elu(self.g2_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_l0', reuse = is_reuse))

        #Final res
        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00_res', reuse = is_reuse)
        h0_0_res = elu(self.g2_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = 2*tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_l0_res', reuse = is_reuse))

        return h0, h0_res


    def generator_decoder_shape_2d_partbase_v2_relu(self, input_feature, is_reuse=False, is_training=True):
        activ = relu

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):
            

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1", "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w= int(s_w/8)
                s8_h= int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_l4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_l_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_l_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_l_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_l_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_l_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_l_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_l_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_l_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_l_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_l_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
                h0_1 = activ(self.bns[name + "/" + "g_l_bn0_1"](h0_1, train=is_training, reuse = is_reuse))

            return h0_1

         # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)




        if not is_reuse:
            self.g2_bn0_0_res = batch_norm(name='g_l_bn0_0_res')
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')        
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4   = batch_norm(name='g_l_bn4')
            self.g2_bn5   = batch_norm(name='g_l_bn5')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_l5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g2_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_l4', reuse = is_reuse)
        h4_1 = activ(self.g2_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_l40', reuse = is_reuse)
        h4_0 = activ(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_l32', reuse = is_reuse)
        h3_2 = activ(self.g2_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
        h3_1 = activ(self.g2_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
        h3_0 = activ(self.g2_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
        h2_2 = activ(self.g2_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
        h2_1 = activ(self.g2_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
        h2_0 = activ(self.g2_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
        h1_2 = activ(self.g2_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
        h1_1 = activ(self.g2_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
        h1_0 = activ(self.g2_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
        h0_2 = activ(self.g2_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
        h0_1 = activ(self.g2_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00', reuse = is_reuse)
        h0_0 = activ(self.g2_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_l0', reuse = is_reuse))

        #Final res
        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00_res', reuse = is_reuse)
        h0_0_res = activ(self.g2_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = 2*tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_l0_res', reuse = is_reuse))

        return h0, h0_res



    def generator_decoder_shape_2d_partbase_v3(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):
            

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1", "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w= int(s_w/8)
                s8_h= int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_l4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_l_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_l31', reuse = is_reuse, use_bias = False)
                h3_1 = elu(self.bns[name + "/" + "g_l_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse, use_bias = False)
                h3_0 = elu(self.bns[name + "/" + "g_l_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse, use_bias = False)
                h2_2 = elu(self.bns[name + "/" + "g_l_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse, use_bias = False)
                h2_1 = elu(self.bns[name + "/" "g_l_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse, use_bias = False)
                h2_0 = elu(self.bns[name + "/" + "g_l_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse, use_bias = False)
                h1_2 = elu(self.bns[name + "/" + "g_l_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse, use_bias = False)
                h1_1 = elu(self.bns[name + "/" + "g_l_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_l10', reuse = is_reuse, use_bias = False)
                h1_0 = elu(self.bns[name + "/" + "g_l_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse, use_bias = False)
                h0_2 = elu(self.bns[name + "/" + "g_l_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse, use_bias = False)
                h0_1 = elu(self.bns[name + "/" + "g_l_bn0_1"](h0_1, train=is_training, reuse = is_reuse))

            return h0_1

         # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)




        if not is_reuse:
            self.g2_bn0_0_res = batch_norm(name='g_l_bn0_0_res')
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')        
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4   = batch_norm(name='g_l_bn4')
            self.g2_bn5   = batch_norm(name='g_l_bn5')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s16_h= int(s_h/16)
        s16_w= int(s_w/16)
                    
        # project `z` and reshape
        
        '''
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_l5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g2_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df_dim*5, strides=[2,2], name='g_l4', reuse = is_reuse)
        h4_1 = elu(self.g2_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_l40', reuse = is_reuse)
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))
        '''

        h4_0 = linear(input_feature, df*4*s16_h*s16_w, scope= 'g_l40_lin', reuse = is_reuse)
        h4_0 = tf.reshape(h4_0, [-1, s16_h, s16_w, df*4])
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))



        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_l32', reuse = is_reuse, use_bias = False)
        h3_2 = elu(self.g2_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_l31', reuse = is_reuse, use_bias = False)
        h3_1 = elu(self.g2_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse, use_bias = False)
        h3_0 = elu(self.g2_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse, use_bias = False)
        h2_2 = elu(self.g2_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse, use_bias = False)
        h2_1 = elu(self.g2_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse, use_bias = False)
        h2_0 = elu(self.g2_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse, use_bias = False)
        h1_2 = elu(self.g2_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse, use_bias = False)
        h1_1 = elu(self.g2_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_l10', reuse = is_reuse, use_bias = False)
        h1_0 = elu(self.g2_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse, use_bias = False)
        h0_2 = elu(self.g2_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse, use_bias = False)
        h0_1 = elu(self.g2_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00', reuse = is_reuse, use_bias = False)
        h0_0 = elu(self.g2_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_l0', reuse = is_reuse))

        #Final res
        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00_res', reuse = is_reuse, use_bias = False)
        h0_0_res = elu(self.g2_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = 2*tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_l0_res', reuse = is_reuse))

        return h0, h0_res


    def generator_decoder_shape_2d_partbase_v5_relu(self, input_feature, is_reuse=False, is_training=True):
        activ = relu

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):
            

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1", "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w= int(s_w/8)
                s8_h= int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_l4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_l_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_l_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_l_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_l_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_l_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_l_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_l_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_l_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_l_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_l_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
                #h0_1 = activ(self.bns[name + "/" + "g_l_bn0_1"](h0_1, train=is_training, reuse = is_reuse))

            return h0_1

         # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)




        if not is_reuse:
            self.g2_bn0_0_res = batch_norm(name='g_l_bn0_0_res')
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_1_local = batch_norm(name='g_l_bn0_1_local')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')        
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4   = batch_norm(name='g_l_bn4')
            self.g2_bn5   = batch_norm(name='g_l_bn5')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_l5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g2_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_l4', reuse = is_reuse)
        h4_1 = activ(self.g2_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_l40', reuse = is_reuse)
        h4_0 = activ(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_l32', reuse = is_reuse)
        h3_2 = activ(self.g2_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
        h3_1 = activ(self.g2_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
        h3_0 = activ(self.g2_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
        h2_2 = activ(self.g2_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
        h2_1 = activ(self.g2_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
        h2_0 = activ(self.g2_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
        h1_2 = activ(self.g2_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
        h1_1 = activ(self.g2_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
        h1_0 = activ(self.g2_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
        h0_2 = activ(self.g2_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
        h0_1 = activ(self.g2_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        local_bg = deconv2d(h0_2, df, strides=[1,1], name='g_l01_local', reuse = is_reuse)
        local = tf.maximum(local, local_bg)
        local = activ(self.g2_bn0_1_local(local, train=is_training, reuse = is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00', reuse = is_reuse)
        h0_0 = activ(self.g2_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_l0', reuse = is_reuse))

        #Final res
        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00_res', reuse = is_reuse)
        h0_0_res = activ(self.g2_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = 2*tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_l0_res', reuse = is_reuse))

        return h0, h0_res


    def generator_decoder_shape_2d_partbase_v6_elu(self, input_feature, is_reuse=False, is_training=True):
        #v2_elu_nose_bg
        activ = elu

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):
            

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1", "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w= int(s_w/8)
                s8_h= int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_l4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_l_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_l_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_l_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_l_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_l_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_l_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_l_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_l_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_l_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_l_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
                h0_1 =       self.bns[name + "/" + "g_l_bn0_1"](h0_1, train=is_training, reuse = is_reuse)

            return h0_1

        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_shape(self, input_feature, [bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)




        if not is_reuse:
            self.g2_bn0_0_res = batch_norm(name='g_l_bn0_0_res')
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_1_local = batch_norm(name='g_l_bn0_1_local_v6')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')        
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4   = batch_norm(name='g_l_bn4')
            self.g2_bn5   = batch_norm(name='g_l_bn5')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_l5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g2_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_l4', reuse = is_reuse)
        h4_1 = activ(self.g2_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_l40', reuse = is_reuse)
        h4_0 = activ(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_l32', reuse = is_reuse)
        h3_2 = activ(self.g2_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_l31', reuse = is_reuse)
        h3_1 = activ(self.g2_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_l30', reuse = is_reuse)
        h3_0 = activ(self.g2_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_l22', reuse = is_reuse)
        h2_2 = activ(self.g2_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_l21', reuse = is_reuse)
        h2_1 = activ(self.g2_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_l20', reuse = is_reuse)
        h2_0 = activ(self.g2_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_l12', reuse = is_reuse)
        h1_2 = activ(self.g2_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_l11', reuse = is_reuse)
        h1_1 = activ(self.g2_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_l10', reuse = is_reuse)
        h1_0 = activ(self.g2_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_l02', reuse = is_reuse)
        h0_2 = activ(self.g2_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_l01', reuse = is_reuse)
        h0_1 = activ(self.g2_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        local_bg = deconv2d(h0_2, df, strides=[1,1], name='g_l01_local_v6', reuse = is_reuse, use_bias=False)
        local_bg = self.g2_bn0_1_local(local_bg, train=is_training, reuse = is_reuse)
        local = activ(tf.maximum(local, local_bg))
        
        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00', reuse = is_reuse)
        h0_0 = activ(self.g2_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_l0', reuse = is_reuse))

        #Final res
        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_l00_res', reuse = is_reuse)
        h0_0_res = activ(self.g2_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = 2*tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_l0_res', reuse = is_reuse))

        return h0, h0_res

            

    def generator_decoder_albedo_v1(self, k52_tex, is_reuse=False, is_training=True):
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
            self.g1_bn0_0_res = batch_norm(name='g_s_bn6_res')
        
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

        h0_1_res = deconv2d(h0_2, df, strides=[1,1], name='g_h01_res', reuse = is_reuse)
        h0_1_res = elu(self.g1_bn0_0_res(h0_1_res, train=is_training, reuse = is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_1_res, self.c_dim, strides=[1,1], name='g_h0_res', reuse = is_reuse))
            
        return h0, h0_res


    def generator_decoder_albedo_part_based(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1", "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w= int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h= int(s_h/2), int(s_h/4), int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_h4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_h_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
                h3_1 = elu(self.bns[name + "/" + "g_h_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
                h3_0 = elu(self.bns[name + "/" + "g_h_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
                h2_2 = elu(self.bns[name + "/" + "g_h_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
                h2_1 = elu(self.bns[name + "/" "g_h_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
                h2_0 = elu(self.bns[name + "/" + "g_h_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
                h1_2 = elu(self.bns[name + "/" + "g_h_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
                h1_1 = elu(self.bns[name + "/" + "g_h_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
                h1_0 = elu(self.bns[name + "/" + "g_h_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
                h0_2 = elu(self.bns[name + "/" + "g_h_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
                h0_1 = elu(self.bns[name + "/" + "g_h_bn0_1"](h0_1, train=is_training, reuse = is_reuse))


            return h0_1

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


        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        


        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
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

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00', reuse = is_reuse)
        h0_0 = elu(self.g1_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))
            
        return h0

    def generator_decoder_albedo_part_based_v2(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1", "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w= int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h= int(s_h/2), int(s_h/4), int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_h4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_h_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
                h3_1 = elu(self.bns[name + "/" + "g_h_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
                h3_0 = elu(self.bns[name + "/" + "g_h_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
                h2_2 = elu(self.bns[name + "/" + "g_h_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
                h2_1 = elu(self.bns[name + "/" "g_h_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
                h2_0 = elu(self.bns[name + "/" + "g_h_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
                h1_2 = elu(self.bns[name + "/" + "g_h_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
                h1_1 = elu(self.bns[name + "/" + "g_h_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
                h1_0 = elu(self.bns[name + "/" + "g_h_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
                h0_2 = elu(self.bns[name + "/" + "g_h_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
                h0_1 = elu(self.bns[name + "/" + "g_h_bn0_1"](h0_1, train=is_training, reuse = is_reuse))


            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
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


        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        


        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
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

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00', reuse = is_reuse)
        h0_0 = elu(self.g1_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))

        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00_res', reuse = is_reuse)
        h0_0_res = elu(self.g1_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_h0_res', reuse = is_reuse))
            
        return h0, h0_res

    def generator_decoder_albedo_part_based_v3(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1", "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w= int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h= int(s_h/2), int(s_h/4), int(s_h/8)

                
                #s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_h4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_h_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_h31', reuse = is_reuse, use_bias = False)
                h3_1 = elu(self.bns[name + "/" + "g_h_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse, use_bias = False)
                h3_0 = elu(self.bns[name + "/" + "g_h_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse, use_bias = False)
                h2_2 = elu(self.bns[name + "/" + "g_h_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse, use_bias = False)
                h2_1 = elu(self.bns[name + "/" "g_h_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse, use_bias = False)
                h2_0 = elu(self.bns[name + "/" + "g_h_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse, use_bias = False)
                h1_2 = elu(self.bns[name + "/" + "g_h_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse, use_bias = False)
                h1_1 = elu(self.bns[name + "/" + "g_h_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_h10', reuse = is_reuse, use_bias = False)
                h1_0 = elu(self.bns[name + "/" + "g_h_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse, use_bias = False)
                h0_2 = elu(self.bns[name + "/" + "g_h_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse, use_bias = False)
                h0_1 = elu(self.bns[name + "/" + "g_h_bn0_1"](h0_1, train=is_training, reuse = is_reuse))


            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
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


        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)
        


        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s16_h= int(s_h/16)
        s16_w= int(s_w/16)
                    
        # project `z` and reshape
        '''
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g1_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse = is_reuse)
        h4_1 = elu(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_h40', reuse = is_reuse)
        h4_0 = elu(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))
        '''

        h4_0 = linear(input_feature, df*4*s16_h*s16_w, scope= 'g_h40_lin', reuse = is_reuse)
        h4_0 = tf.reshape(h4_0, [-1, s16_h, s16_w, df*4])
        h4_0 = elu(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_h32', reuse = is_reuse, use_bias = False)
        h3_2 = elu(self.g1_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_h31', reuse = is_reuse, use_bias = False)
        h3_1 = elu(self.g1_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse, use_bias = False)
        h3_0 = elu(self.g1_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse, use_bias = False)
        h2_2 = elu(self.g1_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse, use_bias = False)
        h2_1 = elu(self.g1_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse, use_bias = False)
        h2_0 = elu(self.g1_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse, use_bias = False)
        h1_2 = elu(self.g1_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse, use_bias = False)
        h1_1 = elu(self.g1_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_h10', reuse = is_reuse, use_bias = False)
        h1_0 = elu(self.g1_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse, use_bias = False)
        h0_2 = elu(self.g1_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse, use_bias = False)
        h0_1 = elu(self.g1_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00', reuse = is_reuse, use_bias = False)
        h0_0 = elu(self.g1_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))

        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00_res', reuse = is_reuse, use_bias = False)
        h0_0_res = elu(self.g1_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_h0_res', reuse = is_reuse))
            
        return h0, h0_res


    def generator_decoder_albedo_part_based_v2_relu(self, input_feature, is_reuse=False, is_training=True):
        #v2 + RELU

        activ = relu

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):


            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1", "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w= int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h= int(s_h/2), int(s_h/4), int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_h4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_h_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_h_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_h_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_h_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_h_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_h_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_h_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_h_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_h_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_h_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
                h0_1 = activ(self.bns[name + "/" + "g_h_bn0_1"](h0_1, train=is_training, reuse = is_reuse))


            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
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


        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        


        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g1_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse = is_reuse)
        h4_1 = activ(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_h40', reuse = is_reuse)
        h4_0 = activ(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_h32', reuse = is_reuse)
        h3_2 = activ(self.g1_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
        h3_1 = activ(self.g1_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
        h3_0 = activ(self.g1_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
        h2_2 = activ(self.g1_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
        h2_1 = activ(self.g1_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
        h2_0 = activ(self.g1_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
        h1_2 = activ(self.g1_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
        h1_1 = activ(self.g1_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
        h1_0 = activ(self.g1_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
        h0_2 = activ(self.g1_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
        h0_1 = activ(self.g1_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00', reuse = is_reuse)
        h0_0 = activ(self.g1_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))

        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00_res', reuse = is_reuse)
        h0_0_res = activ(self.g1_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_h0_res', reuse = is_reuse))
            
        return h0, h0_res

    def generator_decoder_albedo_part_based_v4_relu(self, input_feature, is_reuse=False, is_training=True):
        #v2 + RELU + nose

        activ = relu

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):


            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1", "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w= int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h= int(s_h/2), int(s_h/4), int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_h4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_h_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_h_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_h_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_h_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_h_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_h_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_h_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_h_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_h_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_h_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
                h0_1 = activ(self.bns[name + "/" + "g_h_bn0_1"](h0_1, train=is_training, reuse = is_reuse))


            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
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


        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g1_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse = is_reuse)
        h4_1 = activ(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_h40', reuse = is_reuse)
        h4_0 = activ(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_h32', reuse = is_reuse)
        h3_2 = activ(self.g1_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
        h3_1 = activ(self.g1_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
        h3_0 = activ(self.g1_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
        h2_2 = activ(self.g1_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
        h2_1 = activ(self.g1_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
        h2_0 = activ(self.g1_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
        h1_2 = activ(self.g1_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
        h1_1 = activ(self.g1_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
        h1_0 = activ(self.g1_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
        h0_2 = activ(self.g1_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
        h0_1 = activ(self.g1_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00', reuse = is_reuse)
        h0_0 = activ(self.g1_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))

        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00_res', reuse = is_reuse)
        h0_0_res = activ(self.g1_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_h0_res', reuse = is_reuse))
            
        return h0, h0_res



    def generator_decoder_albedo_part_based_v5_relu(self, input_feature, is_reuse=False, is_training=True):
        #v2 + RELU + nose + local_bg

        activ = relu

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):


            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1", "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w= int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h= int(s_h/2), int(s_h/4), int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_h4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_h_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_h_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_h_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_h_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_h_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_h_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_h_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_h_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_h_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_h_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
                #h0_1 = activ(self.bns[name + "/" + "g_h_bn0_1"](h0_1, train=is_training, reuse = is_reuse))


            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_1_local = batch_norm(name='g_h_bn0_1_local')
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


        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g1_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse = is_reuse)
        h4_1 = activ(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_h40', reuse = is_reuse)
        h4_0 = activ(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_h32', reuse = is_reuse)
        h3_2 = activ(self.g1_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
        h3_1 = activ(self.g1_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
        h3_0 = activ(self.g1_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
        h2_2 = activ(self.g1_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
        h2_1 = activ(self.g1_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
        h2_0 = activ(self.g1_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
        h1_2 = activ(self.g1_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
        h1_1 = activ(self.g1_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
        h1_0 = activ(self.g1_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
        h0_2 = activ(self.g1_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
        h0_1 = activ(self.g1_bn0_1(h0_1, train=is_training, reuse = is_reuse))


        local_bg = deconv2d(h0_2, df, strides=[1,1], name='g_h01_local', reuse = is_reuse)
        local = tf.maximum(local, local_bg)
        local = activ(self.g1_bn0_1_local(local, train=is_training, reuse = is_reuse))



        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00', reuse = is_reuse)
        h0_0 = activ(self.g1_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))

        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00_res', reuse = is_reuse)
        h0_0_res = activ(self.g1_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_h0_res', reuse = is_reuse))
            
        return h0, h0_res



    def generator_decoder_albedo_part_based_v6_elu(self, input_feature, is_reuse=False, is_training=True):
        #v2 + ELU + nose + local_bg
        activ = elu

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):


            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1", "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            #print(self.bns.keys())
            #print("----------------")


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w= int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h= int(s_h/2), int(s_h/4), int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_h4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_h_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_h_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_h_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_h_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_h_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_h_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_h_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_h_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_h_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_h_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
                h0_1 =       self.bns[name + "/" + "g_h_bn0_1"](h0_1, train=is_training, reuse = is_reuse)


            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_1_local = batch_norm(name='g_h_bn0_1_local_v6')
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


        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48] #nose
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='nose_v6', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g1_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse = is_reuse)
        h4_1 = activ(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_h40', reuse = is_reuse)
        h4_0 = activ(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_h32', reuse = is_reuse)
        h3_2 = activ(self.g1_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
        h3_1 = activ(self.g1_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
        h3_0 = activ(self.g1_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
        h2_2 = activ(self.g1_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
        h2_1 = activ(self.g1_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
        h2_0 = activ(self.g1_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
        h1_2 = activ(self.g1_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
        h1_1 = activ(self.g1_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
        h1_0 = activ(self.g1_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
        h0_2 = activ(self.g1_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
        h0_1 = activ(self.g1_bn0_1(h0_1, train=is_training, reuse = is_reuse))


        local_bg = deconv2d(h0_2, df, strides=[1,1], name='g_h01_local_v6', reuse = is_reuse, use_bias=False)
        local_bg = self.g1_bn0_1_local(local_bg, train=is_training, reuse = is_reuse)
        local = activ(tf.maximum(local, local_bg))
        

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00', reuse = is_reuse)
        h0_0 = activ(self.g1_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))

        h0_0_res = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00_res', reuse = is_reuse)
        h0_0_res = activ(self.g1_bn0_0_res(h0_0_res, train=is_training, reuse = is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[1,1], name='g_h0_res', reuse = is_reuse))
            
        return h0, h0_res







    def discriminator(self, image,  is_reuse=False, is_training = True):
        if self.is_using_patchGAN:
            return self.discriminator_patchGAN(image,  is_reuse, is_training)


    def discriminator_patchGAN(self, image,  is_reuse=False, is_training = True):
        if not is_reuse:
            self.d_bn1_0 = batch_norm(name='d_k_bn1_0')
            self.d_bn2_0 = batch_norm(name='d_k_bn2_0')
            self.d_bn3_0 = batch_norm(name='d_k_bn3_0')
            self.d_bn4_0 = batch_norm(name='d_k_bn4_0')


        s16 = int(self.output_size/16)
        #if self.is_combine:
        #    img = tf.reshape(image, shape = [self.batch_size*4, self.output_size, self.output_size, self.c_dim])
        #else:
        #    img = tf.reshape(image, shape = [self.batch_size*2, self.output_size, self.output_size, self.c_dim])
        k0  = elu(          conv2d(image, self.df_dim*1, d_h=1, d_w =1, name='d_k01_conv', reuse = is_reuse),                                       name='d_k01_prelu')

        k1  = elu(self.d_bn1_0(conv2d(k0, self.df_dim*2, d_h=2, d_w =2, name='d_k10_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k10_prelu')
        #k1_1 = elu(self.d_bn1_1(conv2d(k1_0, self.df_dim*2, d_h=1, d_w =1, name='d_k11_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k11_prelu')
        #k1_2 = elu(self.d_bn1_2(conv2d(k1_1, self.df_dim*4, d_h=1, d_w =1, name='d_k12_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k12_prelu')
        #k1_3 =               maxpool2d(k1_2, k=2, padding='VALID')
        k2  = elu(self.d_bn2_0(conv2d(k1, self.df_dim*4, d_h=2, d_w =2, name='d_k20_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k20_prelu')
        #k2_1 = elu(self.d_bn2_1(conv2d(k2_0, self.df_dim*3, d_h=1, d_w =1, name='d_k21_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k21_prelu')
        #k2_2 = elu(self.d_bn2_2(conv2d(k2_1, self.df_dim*6, d_h=1, d_w =1, name='d_k22_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k22_prelu')
        #k2_3 =               maxpool2d(k2_2, k=2, padding='VALID')
        k3  = elu(self.d_bn3_0(conv2d(k2, self.df_dim*6, d_h=2, d_w =2, name='d_k30_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k30_prelu')
        #k3_1 = elu(self.d_bn3_1(conv2d(k3_0, self.df_dim*4, d_h=1, d_w =1, name='d_k31_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k31_prelu')
        #k3_2 = elu(self.d_bn3_2(conv2d(k3_1, self.df_dim*8, d_h=1, d_w =1, name='d_k32_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k32_prelu')
        #k3_3 =               maxpool2d(k3_2, k=2, padding='VALID')
        k4 = elu(self.d_bn4_0(conv2d(k3, self.df_dim*8, d_h=2, d_w =2, name='d_k40_conv', reuse = is_reuse), train=is_training, reuse = is_reuse), name='d_k40_prelu')
        k5 = conv2d(k4, 1,  d_h=1, d_w =1, name='d_k42_conv_ad', reuse = is_reuse)

        print('k5.get_shape()')
        print(k5.get_shape())

        return k5

            
    def sampler(self, input_images, with_landmark = True):
                        
        shape_fx, tex_fx, m, il,_ = self.generator_encoder( input_images, is_reuse=True, is_training=False)
        shape, _ = self.generator_decoder_shape(shape_fx, is_reuse=True, is_training=False)
        albedo = self.generator_decoder_albedo(tex_fx, is_reuse=True, is_training=False)
        albedo = albedo[0] + albedo[1]

        shape_full = shape * self.std_shape_tf + self.mean_shape_tf
        m_full = m * self.std_m_tf + self.mean_m_tf

        shade = generate_shade(il, m_full, shape_full, texture_size=self.texture_size, is_reduce = self.is_reduce)
        texture = 2.0*tf.multiply( (albedo + 1.0)/2.0, shade) - 1
        texture = tf.clip_by_value(texture, -1, 1)   
        warped_img, mask = warp_texture(texture, m_full, shape_full, output_size=self.output_size, is_reduce = self.is_reduce)

        mask = tf.expand_dims(mask, -1)

        overlay_img = tf.multiply(warped_img, mask) + tf.multiply(input_images, 1 - mask)

        return shape_full, shade, texture, albedo, m, warped_img, overlay_img

  
    @property
    def history_file(self):
        return os.path.join(self.checkpoint_dir, "history.npy")
              
    @property
    def model_dir(self):
        return "" # "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
      
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
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

            #self.d_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            #self.g_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))


            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")

            return False, 0

    def load_checkpoint(self, ckpt_file):
        if os.path.isfile(ckpt_file):
            self.saver.restore(self.sess, ckpt_file)
            print(" [*] Success to read {}".format(ckpt_file))
        else:
            self.load(ckpt_file)


#------------------------------------------------------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    import argparse
    parser = argparse.ArgumentParser(description="My parser")

    parser.add_argument("--epoch", type=int, default=1000, help="Epoch to train [25]")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate of for adam [0.0002]")
    parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of adam [0.5]")
    parser.add_argument("--train_size", type=int, default=5000000, help="The size of train images [np.inf]")
    parser.add_argument("--batch_size", type=int, default=64, help="The size of batch images [64]")
    parser.add_argument("--sample_size", type=int, default=64, help="The size of batch samples images [64]")
    parser.add_argument("--image_size", type=int, default=108, help="The size of image to use (will be center cropped) [108]")
    parser.add_argument("--output_size", type=int, default=224, help="The size of the output images to produce [64]")
    parser.add_argument("--c_dim", type=int, default=3, help="Dimension of image color. [3]")
    parser.add_argument("--dataset", default="celebA", help="The name of dataset [celebA, mnist, lsun]")
    parser.add_argument("--checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
    parser.add_argument("--samples_dir", default="samples", help="Directory name to save the image samples [samples]")
    parser.add_argument("--is_train", type=str2bool, nargs='?', const=True, default=False, help="True for training, False for testing [False]")
    parser.add_argument("--is_reduce", type=str2bool, nargs='?', const=True, default=False, help="True for 6k verteices, False for 50k vertices")
    parser.add_argument("--is_crop", type=str2bool, nargs='?', const=True, default=False, help="True for training, False for testing [False]")
    parser.add_argument("--visualize", type=str2bool, nargs='?', const=True, default=False, help="True for visualizing, False for nothing [False]")
    parser.add_argument("--gf_dim", type=int, default=32)
    parser.add_argument("--gfc_dim", type=int, default=512)
    parser.add_argument("--gpu", default="0", help="GPU to use [0]")

    parser.add_argument("--is_using_landmark", type=str2bool, nargs='?', const=True, default=False, help="Using landmark loss [False]")
    parser.add_argument("--is_using_sym", type=str2bool, nargs='?', const=True, default=False, help="Using sym loss [False]")
    parser.add_argument("--is_using_res_sym", type=str2bool, nargs='?', const=True, default=False, help="Using sym loss [False]")
    parser.add_argument("--is_using_recon", type=str2bool, nargs='?', const=True, default=False, help="Using rescontruction loss [False]")
    parser.add_argument("--is_using_frecon", type=str2bool, nargs='?', const=True, default=False, help="Using feature rescontruction loss [False]")
    parser.add_argument("--is_using_graddiff", type=str2bool, nargs='?', const=True, default=False, help="Using gradient difference [False]")
    parser.add_argument("--is_gt_m", type=str2bool, nargs='?', const=True, default=False, help="Using gt m [False]")
    parser.add_argument("--is_partbase_albedo", type=str2bool, nargs='?', const=True, default=False, help="Using part based albedo decoder [False]")
    parser.add_argument("--is_using_linear", type=str2bool, nargs='?', const=True, default=False, help="Using linear model supervision [False]")
    parser.add_argument("--is_batchwise_white_shading", type=str2bool, nargs='?', const=True, default=False, help="Using batchwise white shading constraint [False]")
    parser.add_argument("--is_const_albedo", type=str2bool, nargs='?', const=True, default=False, help="Using batchwise const albedo constraint [False]")
    parser.add_argument("--is_const_local_albedo", type=str2bool, nargs='?', const=True, default=False, help="Using batchwise const albedo constraint [False]")
    parser.add_argument("--is_smoothness", type=str2bool, nargs='?', const=True, default=False, help="Using pairwise loss [False]")

    parser.add_argument("--is_using_GAN", type=str2bool, nargs='?', const=True, default=False, help="Using is_using_GAN [False]")
    parser.add_argument("--is_using_L2_GAN", type=str2bool, nargs='?', const=True, default=False, help="Using L2 GAN [False]")
    parser.add_argument("--is_using_patchGAN", type=str2bool, nargs='?', const=True, default=False, help="Using pathc _GAN [False]")
    parser.add_argument("--is_random_gan_labels", type=str2bool, nargs='?', const=True, default=False, help="Using is_using_perceptual [True]")
    parser.add_argument("--is_combine", type=str2bool, nargs='?', const=True, default=False, help="Using is_using_perceptual [False]")

    parser.add_argument("--shape_loss", default="l2")
    parser.add_argument("--tex_loss", default="l1")

    args = parser.parse_args()
    print(args)


    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.samples_dir):
        os.makedirs(args.samples_dir)

    gpu_options = tf.GPUOptions(visible_device_list =args.gpu, per_process_gpu_memory_fraction = 0.99, allow_growth = True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:
        dcgan = DCGAN(sess, args)
            
        if args.is_train:
            dcgan.train(args)
        else:
            dcgan.load(args.checkpoint_dir)
            dcgan.test(args, True)


if __name__ == '__main__':
    main()


