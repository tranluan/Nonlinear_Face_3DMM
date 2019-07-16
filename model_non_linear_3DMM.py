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
CONST_PIXELS_NUM = 20



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

        self.is_using_landmark = config.is_using_landmark
        self.is_using_symetry = config.is_using_symetry
        self.is_using_recon = config.is_using_recon
        self.is_using_frecon = config.is_using_frecon
        self.is_batchwise_white_shading = config.is_batchwise_white_shading
        self.is_const_albedo = config.is_const_albedo
        self.is_const_local_albedo = config.is_const_local_albedo
        self.is_smoothness = config.is_smoothness
        
        self.mDim = 8
        self.ilDim = 27
                
        self.vertexNum = VERTEX_NUM
        self.landmark_num = 68

        
        self.checkpoint_dir = config.checkpoint_dir
        self.samples_dir = config.samples_dir

        if not os.path.exists(self.samples_dir+"/"+self.model_dir):
            os.makedirs(self.samples_dir+"/"+self.model_dir)
        if not os.path.exists(self.checkpoint_dir+"/"+self.model_dir):
            os.makedirs(self.checkpoint_dir+"/"+self.model_dir)

        self.setupParaStat()
        #self.setupValData()
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


        # For const alb loss
        self.albedo_indexes_x1 = tf.placeholder(tf.int32, [self.batch_size, CONST_PIXELS_NUM, 1], name='idexes_x1')
        self.albedo_indexes_y1 = tf.placeholder(tf.int32, [self.batch_size, CONST_PIXELS_NUM, 1], name='idexes_y1')

        self.albedo_indexes_x2 = tf.placeholder(tf.int32, [self.batch_size, CONST_PIXELS_NUM, 1], name='idexes_x2')
        self.albedo_indexes_y2 = tf.placeholder(tf.int32, [self.batch_size, CONST_PIXELS_NUM, 1], name='idexes_y2')

        self.const_alb_mask = load_const_alb_mask()

        def model_and_loss(input_images_fn_300W, input_masks_fn_300W, texture_labels_fn_300W, texture_masks_fn_300W, input_offset_height, input_offset_width, m_300W_labels, shape_300W_labels, albedo_indexes_x1, albedo_indexes_y1, albedo_indexes_x2, albedo_indexes_y2):
            batch_size = self.batch_size / self.gpu_num
            input_images_300W_   = filename2image(input_images_fn_300W, offset_height = input_offset_height, offset_width = input_offset_width, target_height=self.image_size, target_width=self.image_size)
            input_images_300W    = input_images_300W_ /127.5 - 1

            input_masks_300W    = filename2image(input_masks_fn_300W,  offset_height = input_offset_height, offset_width = input_offset_width, target_height=self.image_size, target_width=self.image_size)
            input_masks_300W    = input_masks_300W / 255.0

            texture_300W_labels    = filename2image(texture_labels_fn_300W)
            texture_300W_labels    = texture_300W_labels / 127.5 - 1

            texture_mask_300W_labels = filename2image(texture_masks_fn_300W)
            texture_mask_300W_labels = texture_mask_300W_labels / 255.0


            ## ------------------------- Network ---------------------------
            shape_fx_300W, tex_fx_300W, m_300W, il_300W = self.generator_encoder( input_images_300W, is_reuse=False)
            shape_300W, shape_2d_300W = self.generator_decoder_shape(shape_fx_300W, is_reuse=False, is_training=True)
            albedo_300W = self.generator_decoder_albedo(tex_fx_300W, is_reuse=False, is_training=True)

            m_300W_full = m_300W * self.std_m_tf + self.mean_m_tf
            shape_300W_full = shape_300W * self.std_shape_tf + self.mean_shape_tf
            shape_300W_labels_full = shape_300W_labels * self.std_shape_tf + self.mean_shape_tf
            m_300W_labels_full = m_300W_labels * self.std_m_tf + self.mean_m_tf

            shape_for_synthesize = shape_300W_full
            m_for_synthesize = m_300W_full

            # Rendering
            shade_300W  = generate_shade(il_300W, m_for_synthesize, shape_for_synthesize, self.texture_size)
            texture_300W = 2.0*tf.multiply( (albedo_300W + 1.0)/2.0, shade_300W) - 1


            G_images_300W, G_images_300W_mask = warp_texture(texture_300W, m_for_synthesize, shape_for_synthesize, output_size=self.image_size)

            G_images_300W_mask = tf.multiply(input_masks_300W, tf.expand_dims(G_images_300W_mask, -1))
            G_images_300W = tf.multiply(G_images_300W, G_images_300W_mask) + tf.multiply(input_images_300W, 1 - G_images_300W_mask)

            landmark_u_300W, landmark_v_300W = compute_landmarks(m_300W_full, shape_300W_full, output_size=self.image_size)
            landmark_u_300W_labels, landmark_v_300W_labels = compute_landmarks(m_300W_labels_full, shape_300W_labels_full, output_size=self.image_size)


            

            ##---------------- Losses -------------------------
            g_loss = tf.zeros(1)

            G_loss_shape   = 10*norm_loss(shape_300W, shape_300W_labels, loss_type = self.shape_loss) #tf.zeros(1) 
            G_loss_m       = 5*norm_loss(m_300W,        m_300W_labels,     loss_type = 'l2')


            texture_vis_mask = tf.cast(tf.not_equal(texture_300W_labels, tf.ones_like(texture_300W_labels)*(-1)), tf.float32)
            texture_vis_mask = tf.multiply(texture_vis_mask, texture_mask_300W_labels)
            texture_ratio = tf.reduce_sum(texture_vis_mask)  / (batch_size* self.texture_size[0] * self.texture_size[1] * self.c_dim)

            

            if self.is_batchwise_white_shading:
                uv_mask_tf = tf.expand_dims(tf.expand_dims(tf.constant( self.uv_mask, dtype = tf.float32 ), 0), -1)

                mean_shade = tf.reduce_mean( tf.multiply(shade_300W, uv_mask_tf) , axis=[0,1,2]) * 16384 / 10379
                G_loss_white_shading = 10*norm_loss(mean_shade,  0.99*tf.ones([1, 3], dtype=tf.float32), loss_type = "l2")
            else:
                G_loss_white_shading = tf.zeros(1)

            

            G_loss_texture = norm_loss(texture_300W,  texture_300W_labels, mask = texture_vis_mask, loss_type = self.tex_loss)  / texture_ratio

            G_loss_recon  = 10*norm_loss(G_images_300W, input_images_300W, loss_type = self.tex_loss ) / (tf.reduce_sum(G_images_300W_mask)/ (batch_size* self.image_size  * self.image_size))

            g_loss += G_loss_m + G_loss_shape + G_loss_white_shading

            if self.is_smoothness:
                G_loss_smoothness = 1000*norm_loss( (shape_2d_300W[:, :-2, 1:-1, :] + shape_2d_300W[:, 2:, 1:-1, :] + shape_2d_300W[:, 1:-1, :-2, :] + shape_2d_300W[:, 1:-1, 2:, :])/4.0,
                                                    shape_2d_300W[:, 1:-1, 1:-1, :], loss_type = self.shape_loss)
            else:
                G_loss_smoothness = tf.zeros(1)
            g_loss = g_loss + G_loss_smoothness

            G_landmark_loss = (tf.reduce_mean(tf.nn.l2_loss(landmark_u_300W - landmark_u_300W_labels )) +  tf.reduce_mean(tf.nn.l2_loss(landmark_v_300W - landmark_v_300W_labels ))) / self.landmark_num / batch_size / 50

            if self.is_using_symetry:
                albedo_300W_flip = tf.map_fn(lambda img: tf.image.flip_left_right(img), albedo_300W)
                G_loss_symetry = norm_loss(tf.maximum(tf.abs(albedo_300W-albedo_300W_flip), 0.05), 0, loss_type = self.tex_loss)
            else:
                G_loss_symetry = tf.zeros(1)
            g_loss +=  G_loss_symetry

            if self.is_const_albedo:

                albedo_1 = get_pixel_value(albedo_300W, albedo_indexes_x1, albedo_indexes_y1)
                albedo_2 = get_pixel_value(albedo_300W, albedo_indexes_x2, albedo_indexes_y2)

                G_loss_albedo_const = 5*norm_loss( tf.maximum(tf.abs(albedo_1- albedo_2), 0.05), 0, loss_type = self.tex_loss)
            else:
                G_loss_albedo_const = tf.zeros(1)
            g_loss += G_loss_albedo_const

            if self.is_const_local_albedo:
                local_albedo_alpha = 0.9
                texture_300W_labels_chromaticity = (texture_300W_labels + 1.0)/2.0
                texture_300W_labels_chromaticity = tf.divide(texture_300W_labels_chromaticity, tf.reduce_sum(texture_300W_labels_chromaticity, axis=[-1], keep_dims=True) + 1e-6)

                
                w_u = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :-1, :, :] - texture_300W_labels_chromaticity[:, 1:, :, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :-1, :, :] )
                G_loss_local_albedo_const_u = tf.reduce_mean(norm_loss( albedo_300W[:, :-1, :, :], albedo_300W[:, 1:, :, :], loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_u) / tf.reduce_sum(w_u+1e-6)

                    
                w_v = tf.stop_gradient(tf.exp(-15*tf.norm( texture_300W_labels_chromaticity[:, :, :-1, :] - texture_300W_labels_chromaticity[:, :, 1:, :], ord='euclidean', axis=-1, keep_dims=True)) * texture_vis_mask[:, :, :-1, :] )
                G_loss_local_albedo_const_v = tf.reduce_mean(norm_loss( albedo_300W[:, :, :-1, :], albedo_300W[:, :, 1:, :],  loss_type = 'l2,1', reduce_mean=False, p=0.8) * w_v) / tf.reduce_sum(w_v+1e-6)

                G_loss_local_albedo_const = (G_loss_local_albedo_const_u + G_loss_local_albedo_const_v)*10
            else:
                G_loss_local_albedo_const = tf.zeros(1)
            g_loss += G_loss_local_albedo_const

            if self.is_using_recon:
                g_loss +=  G_loss_recon
            else:
                g_loss += G_loss_texture

            G_loss_frecon = tf.zeros(1)
            

            if self.is_using_landmark:
                g_loss_wlandmark = g_loss + G_landmark_loss
            else:
                g_loss_wlandmark = g_loss


            return g_loss, g_loss_wlandmark, G_loss_m, G_loss_shape, G_loss_texture, G_loss_recon, G_loss_frecon, G_landmark_loss, G_loss_symetry, G_loss_white_shading, G_loss_albedo_const, G_loss_smoothness, G_loss_local_albedo_const, \
                   G_images_300W, texture_300W, albedo_300W, shade_300W, texture_300W_labels, input_images_300W

        g_loss, g_loss_wlandmark, G_loss_m, G_loss_shape, G_loss_texture, G_loss_recon, G_loss_frecon, G_landmark_loss, G_loss_symetry, G_loss_white_shading, G_loss_albedo_const, G_loss_smoothness, G_loss_local_albedo_const, \
            G_images_300W, texture_300W, albedo_300W, shade_300W, texture_300W_labels, input_images_300W \
            = make_parallel(model_and_loss, self.gpu_num, 
                            input_images_fn_300W= self.input_images_fn_300W, input_masks_fn_300W=self.input_masks_fn_300W,
                            texture_labels_fn_300W=self.texture_labels_fn_300W, texture_masks_fn_300W=self.texture_masks_fn_300W,
                            input_offset_height=self.input_offset_height, input_offset_width=self.input_offset_width,
                            m_300W_labels = self.m_300W_labels, shape_300W_labels=self.shape_300W_labels, 
                            albedo_indexes_x1= self.albedo_indexes_x1, albedo_indexes_y1 = self.albedo_indexes_y1,
                            albedo_indexes_x2=self.albedo_indexes_x2, albedo_indexes_y2 = self.albedo_indexes_y2)

        self.G_loss = tf.reduce_mean(g_loss)
        self.G_loss_wlandmark = tf.reduce_mean(g_loss_wlandmark)
        self.G_loss_m = tf.reduce_mean(G_loss_m)
        self.G_loss_shape =  tf.reduce_mean(G_loss_shape)
        self.G_loss_texture =  tf.reduce_mean(G_loss_texture)
        self.G_loss_recon =  tf.reduce_mean(G_loss_recon)
        self.G_loss_frecon =  tf.reduce_mean(G_loss_frecon)
        self.G_landmark_loss =  tf.reduce_mean(G_landmark_loss)
        self.G_loss_symetry =  tf.reduce_mean(G_loss_symetry)
        self.G_loss_white_shading =  tf.reduce_mean(G_loss_white_shading)
        self.G_loss_albedo_const =  tf.reduce_mean(G_loss_albedo_const)
        self.G_loss_local_albedo_const =  tf.reduce_mean(G_loss_local_albedo_const)
        self.G_loss_smoothness =  tf.reduce_mean(G_loss_smoothness)

        self.G_images_300W = tf.clip_by_value(tf.concat(G_images_300W, axis=0), -1, 1)
        self.texture_300W = tf.clip_by_value(tf.concat(texture_300W, axis=0), -1, 1)
        self.albedo_300W = tf.concat(albedo_300W, axis=0)
        self.shade_300W = tf.concat(shade_300W, axis=0)
        self.texture_300W_labels = tf.concat(texture_300W_labels, axis=0)
        self.input_images_300W = tf.concat(input_images_300W, axis=0)


       
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.g_en_vars = [var for var in t_vars if 'g_k' in var.name]
        self.g_tex_de_vars = [var for var in t_vars if 'g_h' in var.name]
        self.g_shape_de_vars = [var for var in t_vars if 'g_s' in var.name]

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep = 10)
    

    def setupParaStat(self):
        self.tri = load_3DMM_tri()
        self.vertex_tri = load_3DMM_vertex_tri()
        self.vt2pixel_u, self.vt2pixel_v = load_3DMM_vt2pixel()
        self.uv_tri, self.uv_mask = load_3DMM_tri_2d(with_mask = True)

        
        


        # Basis
        mu_shape, w_shape = load_Basel_basic('shape')
        mu_exp, w_exp = load_Basel_basic('exp')

        self.mean_shape = mu_shape + mu_exp
        self.std_shape = np.tile(np.array([1e4, 1e4, 1e4]), self.vertexNum)
        #self.std_shape  = np.load('std_shape.npy')

        self.mean_shape_tf = tf.constant(self.mean_shape, tf.float32)
        self.std_shape_tf = tf.constant(self.std_shape, tf.float32)

        self.mean_m = np.load('mean_m.npy')
        self.std_m = np.load('std_m.npy')

        self.mean_m_tf = tf.constant(self.mean_m, tf.float32)
        self.std_m_tf = tf.constant(self.std_m, tf.float32)
        
        self.w_shape = w_shape
        self.w_exp = w_exp

        

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

         # Training data
        self.setupTrainingData()

        valid_idx = range(self.images_300W.shape[0])
        print("Valid images %d/%d" % ( len(valid_300W_idx), self.images_300W.shape[0] ))



        np.random.shuffle(valid_idx)


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
                '''

                # 300W
                batch_idx = valid_idx[idx*config.batch_size:(idx+1)*config.batch_size]
                
                
                tx = np.random.random_integers(0, 32, size=config.batch_size)
                ty = np.random.random_integers(0, 32, size=config.batch_size)

                batch_300W_images_fn = [self.image_filenames[batch_idx[i]] for i in range(config.batch_size)] 



                delta_m      = np.zeros([config.batch_size, 8])
                delta_m[:,6] = np.divide(ty, self.std_m[6])
                delta_m[:,7] = np.divide(32 - tx, self.std_m[7])

                
                batch_m      = self.all_m[batch_idx,:] - delta_m

                batch_shape_para = self.all_shape_para[batch_idx,:]
                batch_exp_para   = self.all_exp_para[batch_idx,:]

                batch_shape  = np.divide( np.matmul(batch_shape_para, np.transpose(self.w_shape)) + np.matmul(batch_exp_para, np.transpose(self.w_exp)), self.std_shape)

                ffeed_dict={ self.m_300W_labels: batch_m, self.shape_300W_labels: batch_shape, self.input_offset_height: tx, self.input_offset_width: ty}
                for i in range(self.batch_size):
                    ffeed_dict[self.input_images_fn_300W[i]] = _300W_LP_DIR + 'image/'+ batch_300W_images_fn[i]
                    ffeed_dict[self.input_masks_fn_300W[i]] = _300W_LP_DIR + 'mask_img/'+ batch_300W_images_fn[i]
                    ffeed_dict[self.texture_labels_fn_300W[i]] = _300W_LP_DIR + 'texture/'+ image2texture_fn(batch_300W_images_fn[i])
                    ffeed_dict[self.texture_masks_fn_300W[i]] = _300W_LP_DIR + 'mask/'+ image2texture_fn(batch_300W_images_fn[i])

                if self.is_const_albedo:
                    indexes1 = np.random.randint(low=0, high=self.const_alb_mask.shape[0], size=[self.batch_size* CONST_PIXELS_NUM])
                    indexes2 = np.random.randint(low=0, high=self.const_alb_mask.shape[0], size=[self.batch_size* CONST_PIXELS_NUM])


                    ffeed_dict[self.albedo_indexes_x1] = np.reshape(self.const_alb_mask[indexes1, 1], [self.batch_size, CONST_PIXELS_NUM, 1])
                    ffeed_dict[self.albedo_indexes_y1] = np.reshape(self.const_alb_mask[indexes1, 0], [self.batch_size, CONST_PIXELS_NUM, 1])
                    ffeed_dict[self.albedo_indexes_x2] = np.reshape(self.const_alb_mask[indexes2, 1], [self.batch_size, CONST_PIXELS_NUM, 1])
                    ffeed_dict[self.albedo_indexes_y2] = np.reshape(self.const_alb_mask[indexes2, 0], [self.batch_size, CONST_PIXELS_NUM, 1])
                

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

            #if self.is_partbase_albedo:
            #    shape_2d = self.generator_decoder_shape_2d_partbase(k52_shape, is_reuse, is_training)
            #else:
            #    shape_2d = self.generator_decoder_shape_2d_v1(k52_shape, is_reuse, is_training) 
            shape_2d = self.generator_decoder_shape_2d(k52_shape, is_reuse, is_training) 

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
        h5 = linear(shape_fx, self.gfc_dim*s32_h*s32_w, scope= 'g_s5_lin', reuse = is_reuse)
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
        h5 = linear(tex_fx, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
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




