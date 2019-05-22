from utils import *
from rendering_ops import *
import tensorflow as tf
import numpy as np
import time

VERTEX_NUM = 53215

def main(_):

    batch_size = 16
    output_size = 224
    texture_size = [192, 224]
    mDim = 8
    vertexNum = VERTEX_NUM
    channel_num = 3

    data = np.load('sample_data.npz')


    gpu_options = tf.GPUOptions(visible_device_list ="0", allow_growth = True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:

        """ Graph """
        m_ph       = tf.placeholder(tf.float32, [batch_size, mDim])
        shape_ph   = tf.placeholder(tf.float32, [batch_size, vertexNum*3])
        texture_ph = tf.placeholder(tf.float32, [batch_size]+texture_size +[channel_num])      

        image_ph = tf.placeholder(tf.float32, [batch_size, output_size, output_size, channel_num])



        '''
        normal_v1 = DEPRECATED_compute_normalf(m_ph, shape_ph, output_size=output_size)      
        normal_v2 = NEW_compute_normalf(m_ph, shape_ph, output_size=output_size)

        s_land1, s_land2 = sess.run( [normal_v1, normal_v2], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        
        print('s_land1')
        print(s_land1.shape)
        print(s_land1[10,0:3,100:103])

        print('s_land2')
        print(s_land2.shape)
        print(s_land2[10,0:3,100:103])
        '''


        



        #images_v1, foreground_mask_v1 = DEPRECATED_warp_texture(texture_ph, m_ph, shape_ph, output_size = output_size)
        images, foreground_mask = warp_texture(texture_ph, m_ph, shape_ph, output_size = output_size)
        
        #s_time = time.time()
        #for i in range(1):
        #    s_img_v1 = sess.run( images_v1, feed_dict={ texture_ph: data['sample_texture'], shape_ph:data['sample_shape'], m_ph:data['sample_m']})       
        #print("Time landmark v1: %f" % (time.time() - s_time))
        #save_images(s_img_v1, [4, -1], './rendered_img_v1.png')

        s_time = time.time()
        for i in range(1):
            s_img  = sess.run( images, feed_dict={ texture_ph: data['sample_texture'], shape_ph:data['sample_shape'], m_ph:data['sample_m']})       
        print("Time landmark v2: %f" % (time.time() - s_time))

        save_images(s_img, [4, -1], './rendered_img.png')
        

        


        #save_images(data['sample_texture'], [4, -1], './texture.png')
        

        
        '''
        upwarped_texture, texture_mask = unwarp_texture(image_ph, m_ph, shape_ph, output_size=output_size)
        s_texture, s_mask  = sess.run( [upwarped_texture, texture_mask], feed_dict={ image_ph: s_img, shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        save_images(s_mask, [4, -1], './texture_pred_mask.png')
        save_images(s_texture, [4, -1], './texture_pred.png')
        '''

        ''' 
        land1_x, land1_y = _DEPRECATED_compute_landmarks(m_ph, shape_ph, output_size=224)
        land2_x, land2_y = compute_landmarks(m_ph, shape_ph, output_size=224)
        s_land1, s_land2 = sess.run( [land1_y, land2_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        
        print('s_land1')
        print(s_land1.shape)
        print(s_land1[10,0:5])

        print('s_land2')
        print(s_land2.shape)
        print(s_land2[10,0:5])
        '''

        '''
        s_time = time.time()
        for i in range(100):
            sess.run( [land1_x, land1_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v1: %f" %  (time.time() - s_time))

        s_time = time.time()
        for i in range(100):
            sess.run( [land2_x, land2_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v2: %f" % (time.time() - s_time))

        s_time = time.time()
        for i in range(100):
            sess.run( [land1_x, land1_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v1: %f" % (time.time() - s_time))

        s_time = time.time()
        for i in range(100):
            sess.run( [land2_x, land2_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v2: %f" % (time.time() - s_time))

        s_time = time.time()
        for i in range(100):
            sess.run( [land1_x, land1_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v1: %f" % (time.time() - s_time))

        s_time = time.time()
        for i in range(100):
            sess.run( [land2_x, land2_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v2: %f" % (time.time() - s_time))

        s_time = time.time()
        for i in range(100):
            sess.run( [land1_x, land1_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v1: %f" % (time.time() - s_time))

        s_time = time.time()
        for i in range(100):
            sess.run( [land2_x, land2_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v2: %f" % (time.time() - s_time))

        s_time = time.time()
        for i in range(1000):
            sess.run( [land1_x, land1_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v1: %f" % (time.time() - s_time))

        s_time = time.time()
        for i in range(1000):
            sess.run( [land2_x, land2_y], feed_dict={ shape_ph:data['sample_shape'], m_ph:data['sample_m']})
        print("Time landmark v2: %f" % (time.time() - s_time))
        '''



        
if __name__ == '__main__':
    tf.app.run()
