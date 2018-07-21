from utils import *
from rendering_ops import *
import tensorflow as tf
import numpy as np

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
        images, foreground_mask = warp_texture(texture_ph, m_ph, shape_ph, output_size = output_size)

        s_img  = sess.run( images, feed_dict={ texture_ph: data['sample_texture'], shape_ph:data['sample_shape'], m_ph:data['sample_m']})
       
        save_images(s_img, [4, -1], './rendered_img.png')
        save_images(data['sample_texture'], [4, -1], './texture.png')
        
        

        
if __name__ == '__main__':
    tf.app.run()
