import os
import scipy.misc
import numpy as np
#import h5py 
import math

#import skvideo.io

from glob import glob

#from model_non_linear_3DMM import DCGAN
from utils import *
from rendering_ops import *

from skimage import io


#import tensorflow as tf
#from ops import *

VERTEX_NUM = 53215
VERTEX_NUM_REDUCE = 39111

def unwarp_texture(image, m, mshape, output_size=224, is_reduce = False):


    n_size = get_shape(image)
    n_size = n_size[0]
    s = output_size   

     # Tri, tri2vt
    if is_reduce:
        tri = load_3DMM_tri_reduce()
        vertex_tri = load_3DMM_vertex_tri_reduce()
        vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel_reduce()

        tri_2d = load_FaceAlignment_tri_2d_reduce()
    else:
        tri = load_3DMM_tri()
        vertex_tri = load_3DMM_vertex_tri()
        vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel()

        tri_2d = load_FaceAlignment_tri_2d()
        
    tri2vt1_const = tf.constant(tri[0,:], tf.int32)
    tri2vt2_const = tf.constant(tri[1,:], tf.int32)
    tri2vt3_const = tf.constant(tri[2,:], tf.int32)

    tri_const = tf.constant(tri, tf.int32)
    tri_2d_const = tf.constant(tri_2d, tf.int32)
    vertex_tri_const = tf.constant(vertex_tri, tf.int32)


    #Vt2pix
    vt2pixel_u_const = tf.constant(vt2pixel_u, tf.float32)
    vt2pixel_v_const = tf.constant(vt2pixel_v, tf.float32)

    #indicies = np.zeros([s*s,2])
    #for i in range(s):
    #    for j in range(s):
    #        indicies[i*s+j ] = [i,j]

    #indicies_const = tf.constant(indicies, tf.float32)
    #[indicies_const_u, indicies_const_v] = tf.split(1, 2, indicies_const)

    ###########m = m * tf.constant(self.std_m) + tf.constant(self.mean_m)
    ###########mshape = mshape * tf.constant(self.std_shape) + tf.constant(self.mean_shape)

    m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
    shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)
    
    pixel_u = []
    pixel_v = []

    masks = []
    for i in range(n_size):

        m_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
        m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)
        m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)
        m_i_row3 = tf.concat([tf.reshape(tf.cross(m_i_row1, m_i_row2), shape = [1, 3]), tf.zeros([1, 1])], axis = 1)
                  
        m_i = tf.concat([m_i, m_i_row3], axis = 0)

        # Compute 2d vertex
        #vertex3d = tf.transpose(tf.reshape( mu_const + tf.matmul(w_shape_const, p_shape_single[i], False, True) + tf.matmul(w_exp_const, p_exp_single[i], False, True), shape = [-1, 3] ))

        vertex3d_rs = tf.transpose(tf.reshape( shape_single[i], shape = [-1, 3] ))

        normal, normalf = compute_normal(vertex3d_rs,tri_const, vertex_tri_const)
        normalf = tf.transpose(normalf)
        normalf4d = tf.concat(axis=0, values=[normalf, tf.ones([1, normalf.get_shape()[-1]], tf.float32)])
        rotated_normalf = tf.matmul(m_i, normalf4d, False, False)
        _, _, rotated_normalf_z = tf.split(axis=0, num_or_size_splits=3, value=rotated_normalf)
        visible_tri = tf.greater(rotated_normalf_z, 0)

        mask_i = tf.gather( tf.cast(visible_tri, dtype=tf.float32),  tri_2d_const ) 
        print("get_shape(mask_i)")
        print(get_shape(mask_i))


        vertex4d = tf.concat(axis = 0, values = [vertex3d_rs, tf.ones([1, vertex3d_rs.get_shape()[-1]], tf.float32)])
        
        vertex2d = tf.matmul(m_i, vertex4d, False, False)
        vertex2d = tf.transpose(vertex2d)
        
        [vertex2d_u, vertex2d_v, vertex2d_z]   = tf.split(axis=1, num_or_size_splits=3, value=vertex2d)
        vertex2d_u = tf.squeeze(vertex2d_u - 1)
        vertex2d_v = tf.squeeze(s - vertex2d_v)

        #vertex2d = tf.concat(axis=1, values=[vertex2d_v, vertex2d_u, vertex2d_z])
        #vertex2d = tf.transpose(vertex2d)

        #vertex2d_u = tf.transpose(vertex2d_u)
        #vertex2d_V = tf.transpose(vertex2d_v)



        vt1 = tf.gather( tri2vt1_const,  tri_2d_const ) 
        vt2 = tf.gather( tri2vt2_const,  tri_2d_const ) 
        vt3 = tf.gather( tri2vt3_const,  tri_2d_const )

        


        pixel1_u = tf.gather( vertex2d_u,  vt1 ) #tf.gather( vt2pixel_u_const,  vt1 ) 
        pixel2_u = tf.gather( vertex2d_u,  vt2 ) 
        pixel3_u = tf.gather( vertex2d_u,  vt3 )

        pixel1_v = tf.gather( vertex2d_v,  vt1 ) 
        pixel2_v = tf.gather( vertex2d_v,  vt2 ) 
        pixel3_v = tf.gather( vertex2d_v,  vt3 )

        pixel_u_i = tf.scalar_mul(scalar = 1.0/3.0, x = tf.add_n([pixel1_u, pixel2_u, pixel3_u]))
        pixel_v_i = tf.scalar_mul(scalar = 1.0/3.0, x = tf.add_n([pixel1_v, pixel2_v, pixel3_v]))

        pixel_u.append(pixel_u_i)
        pixel_v.append(pixel_v_i)

        masks.append(mask_i)


        
    texture = bilinear_sampler(image, pixel_u, pixel_v)
    masks = tf.stack(masks)

    return texture, masks

def main(_):

    #uv_mask = imread('uv_mask.png')
    #np.save('uv_mask.npy', uv_mask/255.0)

    '''
    uv_mask = np.load('uv_mask.npy')

    uv_mask_tf = tf.expand_dims(tf.expand_dims(tf.constant( uv_mask, dtype = tf.float32 ), 0), -1)

    fd = open('texture_const_mask.bin')
    texture_const_mask = np.fromfile(file=fd, dtype=np.uint8)
    fd.close()
    texture_const_mask = texture_const_mask.reshape((-1,2)).astype(np.uint8)

    print(texture_const_mask)
    print(texture_const_mask.shape)
    np.save('texture_const_mask.npy', texture_const_mask)

    texture_const_mask = np.load('texture_const_mask.npy')
    '''


    #print(uv_mask.shape)
    #print(np.max(np.max(uv_mask)))

    batch_size = 16
    output_size = 224
    mDim = 8
    ilDim = 9 * 3
    is_reduce = False
    if is_reduce:
        vertexNum = VERTEX_NUM_REDUCE
    else:
        vertexNum = VERTEX_NUM


    texture_size = [192, 224]
    c_dim = 3

    

    #a = load_FaceAlignment_dataset_recrop_sz224('AFLW2000', with_sh = False)


    
    images_AFLW2000, pid_AFLW2000, m_AFLW2000, pose_AFLW2000, shape_AFLW2000, exp_AFLW2000, tex_para_AFLW2000, _, = load_FaceAlignment_dataset_recrop_sz224('AFLW2000', with_sh = False)
    #mu_tex, w_tex = load_FaceAlignment_2dbasic('tex')
    mu_shape, w_shape = load_FaceAlignment_basic('shape', is_reduce = is_reduce)
    mu_exp, w_exp = load_FaceAlignment_basic('exp', is_reduce = is_reduce)

    '''
    mean_m = np.load('mean_m.npy')
    std_m  = np.load('std_m.npy')
    print(mean_m)
    print(std_m)
    m_AFLW2000 = np.divide(np.subtract(m_AFLW2000, mean_m), std_m)
    '''

    #sub_idxes, sub_idxes_ = load_3DMM_sub_idxes_reduce()



    #batch_tex_para = tex_para_AFLW2000[7:9, :]

    #batch_tex = np.matmul(batch_tex_para, np.transpose(w_tex)) + mu_tex
    #batch_tex = batch_tex.reshape(-1, 128, 128, 3)/127.5-1

    #sample_images = images_AFLW2000[0:64*10:10,:,:,:]
    #sample_images = np.array(sample_images).astype(np.float32)/127.5-1

    #save_images(sample_images, [8, 8], 'a_.png')

    """ Sample images """
    sample_files = range(0, batch_size*5, 5)
    #sample_files[0] = 0
    #sample_files[1] = 390
    #sample_files[3] = 480


    tx = 16*np.ones(batch_size, dtype=np.int)
    ty = 16*np.ones(batch_size, dtype=np.int)

    sample_images_in = [crop(imread(DATA_DIR + 'image/'+ images_AFLW2000[sample_files[i]]), output_size, output_size, tx[i], ty[i]) for i in range(batch_size)]
    sample_images_in = np.array(sample_images_in).astype(np.float32)/127.5-1

    #sample_masks = [crop(mask_AFLW2000[sample_files[i],:,:,:], output_size, output_size, tx[i], ty[i]) for i in range(batch_size)]
    #sample_masks = np.array(sample_masks).astype(np.float32)/255

    #print(np.max(np.max(sample_masks)))

    #sample_textures = [imread('../../data/300W_LP_crop/albedo/AFLW2000/image00019.jpg'), imread('../../data/300W_LP_crop/albedo/AFLW2000/image00020.jpg')]
    sample_textures = [imread(DATA_DIR + 'texture/'+ image2texture_fn(images_AFLW2000[sample_files[i]])) for i in range(batch_size)]
    sample_textures = np.array(sample_textures).astype(np.float32)/127.5-1

    #sample_albedos = [albedo_AFLW2000[sample_files[i],:,:,:] for i in range(batch_size)]
    #sample_albedos = np.array(sample_albedos).astype(np.float32)/127.5-1

    sample_albedos = sample_textures


    #sample_tex_para = tex_para_AFLW2000[sample_files, :]

    #sample_l_albedos = np.matmul(sample_tex_para, np.transpose(w_tex)) + mu_tex
    #sample_l_albedos = sample_l_albedos.reshape(-1, 128, 128, 3)/127.5-1

    

    delta_m      = np.zeros([batch_size, 8])

    delta_m[:,6] = delta_m[:,6] = ty #delta_m[:,6] = np.divide(ty, self.std_m[6]);
    delta_m[:,7] = delta_m[:,7] = 32 - tx #delta_m[:,7] = np.divide(14 - tx, self.std_m[7]);
    
    sample_m      = m_AFLW2000[sample_files,:] - delta_m;

    sample_shape_para = shape_AFLW2000[sample_files,:]
    sample_exp_para   = exp_AFLW2000[sample_files,:]

    #sample_shape  =  np.divide( np.matmul(sample_shape_para, np.transpose(self.w_shape)) + np.matmul(sample_exp_para, np.transpose(self.w_exp)), self.std_shape + 1e-6)
    sample_shape  =  mu_shape + np.matmul(sample_shape_para, np.transpose(w_shape)) + mu_exp + np.matmul(sample_exp_para, np.transpose(w_exp))

    '''
    mean_shape = np.load('mean_shape.npy')
    std_shape  = np.load('std_shape.npy')

    print(np.max(mean_shape))
    print(np.min(mean_shape))

    a = (sample_shape[1] - mean_shape)/std_shape
    print(a)
    print(np.min(a))
    print(np.max(a))
    '''


    sample_il   = np.ones([batch_size, ilDim], dtype=np.float32) #il_AFLW200[sample_files,:]
    


    
    gpu_options = tf.GPUOptions(visible_device_list ="0")

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:

        """ Graph """

        sample_mask_ph    =  tf.placeholder(tf.float32, [batch_size, output_size, output_size, c_dim], name='mask')
        sample_img_ph     =  tf.placeholder(tf.float32, [batch_size, output_size, output_size, c_dim], name='mask')
        
        sample_m_ph       = tf.placeholder(tf.float32, [batch_size, mDim], name='mm')
        #sample_m_ph_full  = sample_m_ph * tf.constant(std_m, tf.float32) + tf.constant(mean_m, tf.float32)
        sample_il_ph      = tf.placeholder(tf.float32, [batch_size, ilDim], name='ii')

        sample_shape_ph   = tf.placeholder(tf.float32, [batch_size, vertexNum*3], name='ssss')

        sample_texture_ph = tf.placeholder(tf.float32, [batch_size, texture_size[0], texture_size[1], c_dim], name='tttt')

        #sample_shade_300W = generate_shade(sample_il_ph, sample_m_ph, sample_shape_ph, texture_size=texture_size)
        #mean_shade = tf.reduce_mean( tf.multiply(sample_shade_300W, uv_mask_tf) , axis=[0,1,2]) * 16384 / 10379

        #shaded_texture_300W = 2.0*tf.multiply( (sample_texture_ph + 1.0)/2.0, sample_shade_300W) - 1
        #shaded_texture_300W = tf.clip_by_value(shaded_texture_300W, -1, 1)


        #sample_unwarp_textures, unwarping_mask = unwarp_texture(sample_img_ph, sample_m_ph, sample_shape_ph, output_size=96, is_remesh = False)
        #unwarping_mask = tf.expand_dims(unwarping_mask, -1)
        #sample_unwarp_textures = sample_unwarp_textures * unwarping_mask


        samples_images, warping_mask = warp_texture(sample_texture_ph, sample_m_ph, sample_shape_ph, output_size = 224)

        #mask = tf.expand_dims(warping_mask, -1) #tf.multiply(sample_mask_ph, tf.expand_dims(warping_mask, -1))

        #samples_images = samples_images * mask + sample_img_ph * (1-mask)

        np.savez('sample_data.npz', sample_texture = sample_textures, sample_shape=sample_shape, sample_m=sample_m)

        
        idx = 102

        s_img  = sess.run( samples_images, feed_dict={ sample_texture_ph: sample_albedos, sample_shape_ph:sample_shape, sample_m_ph:sample_m, sample_il_ph:sample_il, sample_img_ph:sample_images_in}) #, sample_mask_ph:sample_masks})
        save_images(s_img, [8, -1], './warped_texture_{:02d}.png'.format(idx))
        save_images(sample_images_in, [8, -1], './warped_texture_{:02d}_in.png'.format(idx))
        

        #####
        idx=0
        sample_unwarp_textures, unwarping_mask = unwarp_texture(sample_img_ph, sample_m_ph, sample_shape_ph, output_size=224)
        sample_unwarp_textures_, sample_shade_300W_  = sess.run( [sample_unwarp_textures, sample_shade_300W], feed_dict={ sample_texture_ph: sample_albedos, sample_shape_ph:sample_shape, sample_m_ph:sample_m, sample_il_ph:sample_il, sample_img_ph:sample_images_in})
        save_images(sample_unwarp_textures_, [8, -1], './unwarped_texture_{:02d}.png'.format(idx))

        sample_unwarp_albedo_ = ((sample_unwarp_textures_ + 1.0)/2.0) / (sample_shade_300W_)
        print(sample_shade_300W_[0])
        save_images(sample_shade_300W_, [8, -1], './unwarped_albedo_{:02d}.png'.format(idx))#, inverse=True)

        s_img  = sess.run( samples_images, feed_dict={ sample_texture_ph: sample_unwarp_textures_, sample_shape_ph:sample_shape, sample_m_ph:sample_m, sample_il_ph:sample_il, sample_img_ph:sample_images_in}) #, sample_mask_ph:sample_masks})
        save_images(s_img, [8, -1], './warped_texture_{:02d}.png'.format(idx))
        save_images(sample_images_in, [8, -1], './warped_texture_{:02d}_in.png'.format(idx))


        

        '''
        for j in range(batch_size):
            idx = idx + 1

            for i in range(batch_size):
                sample_shape[i]  =  mu_shape + np.matmul(sample_shape_para[i], np.transpose(w_shape)) + mu_exp + np.matmul(sample_exp_para[j], np.transpose(w_exp))

            s_img  = sess.run( samples_images, feed_dict={ sample_texture_ph: sample_unwarp_textures_, sample_shape_ph:sample_shape, sample_m_ph:sample_m, sample_il_ph:sample_il, sample_img_ph:sample_images_in}) #, sample_mask_ph:sample_masks})
            save_images(s_img, [8, -1], './warped_texture_{:02d}.png'.format(idx))
            save_images(sample_images_in, [8, -1], './warped_texture_{:02d}_in.png'.format(idx))
        '''



        #mean_shade_  = sess.run( mean_shade, feed_dict={ sample_texture_ph: sample_albedos, sample_shape_ph:sample_shape, sample_m_ph:sample_m, sample_il_ph:sample_il, sample_img_ph:sample_images_in, sample_mask_ph:sample_masks})

        '''
        sample_albedos = sample_unwarp_textures_

        out_images = np.zeros( [batch_size*8, output_size, output_size, 3], dtype=np.float32)
        for i in range(2,8):
            idxx = np.random.randint(0, 1000, size=[batch_size] )
            sample_m = m_AFLW2000[idxx,:] - delta_m
            out_images[i*batch_size:(i+1)*batch_size]  = sess.run( samples_images, feed_dict={ sample_texture_ph: sample_albedos, sample_shape_ph:sample_shape, sample_m_ph:sample_m, sample_il_ph:sample_il, sample_img_ph:sample_images_in, sample_mask_ph:sample_masks})

        for i in range(batch_size):
            out_images[i] = sample_images_in[i]
            out_images[batch_size+i] = scipy.misc.imresize(sample_unwarp_textures_[i], [output_size, output_size])/127.5-1
        save_images(out_images, [8, -1], './warped_texture_{:02d}_random.png'.format(idx))

        save_images(s_img, [8, -1], './warped_texture_{:02d}.png'.format(idx))
        save_images(sample_images_in, [8, -1], './warped_texture_{:02d}_in.png'.format(idx))
        '''
        

        '''
        background_const = tf.constant(sample_images_in, dtype=tf.float32)



        sample_m_var =  tf.get_variable("m", [batch_size, mDim], dtype=tf.float32, initializer=tf.zeros_initializer)
        sample_il_var =  tf.get_variable("il", [batch_size, ilDim], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
 
        sample_shape_var = tf.get_variable("shape", [batch_size, vertexNum*3], dtype=tf.float32, initializer=tf.zeros_initializer)
        sample_albedo_var = tf.get_variable("albedo", [batch_size, texture_size, texture_size, c_dim], dtype=tf.float32, initializer=tf.zeros_initializer)



        sample_shade_300W = generate_shade(sample_il_var, sample_m_var, sample_shape_var)

        texture_300W = 2.0*tf.multiply( (sample_albedo_var + 1.0)/2.0, sample_shade_300W) - 1


        samples_images, mask = warp_texture(texture_300W, sample_m_var, sample_shape_var)
        mask = tf.expand_dims(mask, -1)
        samples_images = samples_images * mask + background_const * (1-mask)
        samples_images_ = tf.clip_by_value(samples_images, -1, 1)

        sample_m_const = tf.constant(sample_m, dtype=tf.float32)
        sample_il_const = tf.constant(sample_il, dtype=tf.float32)
        sample_albedo_const = tf.constant(sample_l_albedos, dtype=tf.float32) #sample_albedos
        sample_shape_const = tf.constant(sample_shape, dtype=tf.float32)

        ## Losses
        loss = norm_loss(samples_images, background_const, mask = None, loss_type = 'l1')
        optim_il = tf.train.AdamOptimizer(0.02, beta1=0.9).minimize(loss, var_list=[sample_il_var])
        optim_shape = tf.train.AdamOptimizer(250, beta1=0.9).minimize(loss, var_list=[sample_shape_var])
        tf.global_variables_initializer().run()


        sess.run(sample_m_var.assign(sample_m_const))
        sess.run(sample_il_var.assign(sample_il_const))
        sess.run(sample_shape_var.assign(sample_shape_const))
        sess.run(sample_albedo_var.assign(sample_albedo_const))

        s_img  = sess.run( samples_images_)
        idx = 106
        save_images(s_img, [1, 8], './warped_texture_{:02d}.png'.format(idx))


        for i in range(100):
            _, loss_ = sess.run([optim_il, loss])

            if (i % 10) == 0:
                
                print("Idx %d, loss :%f" % (i, loss_ ))

        idx = idx+1
        s_img, pred_shape  = sess.run( [samples_images_, sample_shape_var])
        save_images(s_img, [1, 8], './warped_texture_{:02d}.png'.format(idx))
        for b_idx in range(batch_size):
            np.savetxt('./pred_shape_{:d}_{:d}.txt'.format(idx, b_idx), pred_shape[b_idx])        


        for i in range(100,200):
            _, loss_ = sess.run([optim_shape, loss])

            if (i % 10) == 0:
                
                print("Idx %d, loss :%f" % (i, loss_ ))

        idx = idx+1
        s_img, pred_shape  = sess.run( [samples_images_, sample_shape_var])
        save_images(s_img, [1, 8], './warped_texture_{:02d}.png'.format(idx))
        for b_idx in range(batch_size):
            np.savetxt('./pred_shape_{:d}_{:d}.txt'.format(idx, b_idx), pred_shape[b_idx])

        print('Saved to ./pred_shape_{:02d}.txt'.format(idx))    
        save_images(sample_images_in, [1, 8], './warped_texture_{:02d}_in.png'.format(idx))
        '''







    


    '''

        my_var = tf.get_variable("my_int_variable", [1, 3], dtype=tf.float32, initializer=tf.zeros_initializer)

        a = tf.constant([1.0, 2.0, 3.0])
        b_target = tf.constant([1.0, 6.0, 15.0])

        b = tf.multiply(a, my_var)

        loss = norm_loss(b, b_target, loss_type = 'l1')

        optim = tf.train.AdamOptimizer(0.01, beta1=0.9).minimize(loss, var_list=[my_var])
        
        tf.global_variables_initializer().run()


        my_var_ = sess.run(my_var)
        print(my_var_)

        for i in range(5000):
            _, loss_ = sess.run([optim, loss] )
            my_var_ = sess.run(my_var)
            print(my_var_)
            print("Loss %f" % (loss_))
    '''









    '''
        a = np.random.rand(2, 32,32,3)
        b = np.random.rand(2, 32,32,3)
        c = np.random.rand(2, 32,32,3)

        a_tf = tf.constant(a)
        b_tf = tf.constant(b)
        c_tf = tf.constant(c)

        l2_old = tf.reduce_mean(tf.nn.l2_loss( tf.multiply(a_tf   - b_tf , c_tf) )) / (2*32*32*3)
        l2_new = matrix_norm_loss(a_tf, b_tf, mask = c_tf, loss_type = 'l2')

        l1_old = tf.reduce_mean(tf.abs(a_tf - b_tf))
        l1_new = matrix_norm_loss(a_tf, b_tf, loss_type = 'l1')

        l2_old_, l2_new_, l1_old_, l1_new_ = sess.run([l2_old, l2_new, l1_old, l1_new])

        print([l2_old_ , l2_new_, l1_old_ , l1_new_])
    '''
    




if __name__ == '__main__':
    tf.app.run()
