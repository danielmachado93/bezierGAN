import tensorflow as tf
import numpy as np
import os
from utils import*
import time
from glob import glob
from btp_utils import*
from six.moves import xrange
import datetime
import imageio
from tensorflow.python.tools import inspect_checkpoint as chkp

class bezier_to_pixel_transformer_net(object):
    def __init__(self, sess, output_height=128, output_width=128,
                 batch_size=64, b_dim = 8, nf_dim=64, nfd_dim = 64):

        self.sess = sess # Create session
        self.output_width = output_width # Pixel space width dimension
        self.output_height = output_height # Pixel space height dimension
        self.batch_size = batch_size # batch size
        self.b_dim = b_dim # Number of bezier coordinates. For cubic curves the b_dim is 4 * 2 = 8
        self.nf_dim = nf_dim # Number of filters of last layer  encoder network
        self.nfd_dim = nfd_dim # Number of filters of last layer of decoder network

        self.model_setup()

    def model_setup(self):
        # Setup Placeholders
        self.b_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.b_dim], name='b_placeholder')
        self.p_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.output_width, self.output_height, 1])

        # Evaluate Networks
        self.encoder, self.e1, self.e2, self.e3, self.e4 = self.btp_encoder(images=self.p_placeholder,
                                                                            reuse_variables=False, is_training=True)
        self.decoder = self.btp_decoder(bezier_points=self.b_placeholder,
                                        reuse_variables=False, is_training=True)
        self.encoder_decoder, self.fd1,  self.fd2, self.fd3, self.fd4 = self.btp_encoder(images=self.decoder,
                                                                                       reuse_variables=True, is_training=False)

        # Inference
        self.encoder_p, self.fp1, self.fp2, self.fp3, self.fp4 = self.btp_encoder(images=self.p_placeholder,
                                                                                  reuse_variables=True,
                                                                                  is_training=False)
        self.decoder_test= self.btp_decoder(bezier_points=self.b_placeholder,
                                                        reuse_variables=True,is_training=False)


        # Define Losses
        # BTP - Feature Reconstruction Loss

        self.L1 = tf.reduce_mean(tf.abs(self.fd1-self.fp1), name="L1")
        self.L2 = tf.reduce_mean(tf.abs(self.fd2-self.fp2), name="L2")
        self.L3 = tf.reduce_mean(tf.abs(self.fd3-self.fp3), name="L3")
        self.L4 = tf.reduce_mean(tf.abs(self.fd4-self.fp4), name="L4")

        self.encode_diff_loss = tf.reduce_mean(tf.abs(self.encoder_p-self.encoder_decoder),name="encode_diff_loss")
        self.pixel_wise_loss = tf.reduce_mean(tf.abs(self.p_placeholder-self.decoder))
        self.check_encoder = tf.reduce_mean(tf.abs(self.b_placeholder-self.encoder_p))
        self.decoder_loss = tf.add_n(inputs=(self.L1,self.L2,self.L3,self.L4,self.encode_diff_loss), name="loss")

        self.encoder_loss = tf.losses.absolute_difference(labels=self.b_placeholder, predictions=self.encoder)

        # Trainable variables
        t_vars = tf.trainable_variables()
        self.encoder_vars = [var for var in t_vars if 'btp_encoder' in var.name]
        self.decoder_vars = [var for var in t_vars if 'btp_decoder' in var.name]

        # Saver
        self.saver = tf.train.Saver()

    def btp_encoder(self, images, reuse_variables=None, is_training=None):
        with tf.variable_scope('btp_encoder', reuse=reuse_variables) as scope:
            w16, w8 , w4, w2, w1 = self.output_width//16, self.output_width//8, self.output_width//4, self.output_width//2, self.output_width
            h16, h8, h4, h2, h1 = self.output_height//16, self.output_height//8, self.output_height//4, self.output_height//2, self.output_height
            stddev = 0.02
            # First Block
            b1_conv1_w = tf.get_variable('b1_conv1_w', shape=[3, 3, 1, self.nf_dim], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
            b1_conv1_b = tf.get_variable('b1_conv1_b', shape=[self.nf_dim], initializer=tf.constant_initializer(0.0))
            b1_conv1 = tf.nn.conv2d(input=images,filter=b1_conv1_w, strides=[1,1,1,1], padding='SAME', name='b1_conv1')
            b1_conv1 = tf.nn.bias_add(value=b1_conv1,bias=b1_conv1_b,name='b1_conv1_bias_add')
            b1_conv1 = tf.contrib.layers.batch_norm(inputs=b1_conv1, decay=0.90, epsilon=1e-5, scope='b1_conv1_bn',
                                              is_training=is_training,
                                              scale=True, updates_collections=None)
            b1_conv1 = tf.nn.relu(features=b1_conv1,name='b1_conv1_ReLU')

            b1_conv2_w = tf.get_variable('b1_conv2_w', shape=[3, 3, self.nf_dim, self.nf_dim], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b1_conv2_b = tf.get_variable('b1_conv2_b', shape=[self.nf_dim], initializer=tf.constant_initializer(0.0))
            b1_conv2 = tf.nn.conv2d(input=b1_conv1, filter=b1_conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b1_conv2')
            b1_conv2 = tf.contrib.layers.batch_norm(inputs=b1_conv2, decay=0.90, epsilon=1e-5, scope='b1_conv2_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b1_conv2 = tf.nn.bias_add(value=b1_conv2, bias=b1_conv2_b, name='b1_conv2_bias_add')
            b1_conv2 = tf.nn.relu(features=b1_conv2, name='b1_conv2_ReLU')

            # 1º Down-scale
            ds1 = tf.nn.max_pool(value=b1_conv2,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME',name='ds1')

            # Second Block
            b2_conv1_w = tf.get_variable('b2_conv1_w', shape=[3, 3, self.nf_dim, self.nf_dim*2], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b2_conv1_b = tf.get_variable('b2_conv1_b', shape=[self.nf_dim*2], initializer=tf.constant_initializer(0.0))
            b2_conv1 = tf.nn.conv2d(input=ds1, filter=b2_conv1_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b2_conv1')
            b2_conv1 = tf.nn.bias_add(value=b2_conv1, bias=b2_conv1_b, name='b2_conv1_bias_add')
            b2_conv1 = tf.contrib.layers.batch_norm(inputs=b2_conv1, decay=0.90, epsilon=1e-5, scope='b2_conv1_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b2_conv1 = tf.nn.relu(features=b2_conv1, name='b1_conv1_ReLU')

            b2_conv2_w = tf.get_variable('b2_conv2_w', shape=[3, 3, self.nf_dim*2, self.nf_dim*2], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b2_conv2_b = tf.get_variable('b2_conv2_b', shape=[self.nf_dim*2], initializer=tf.constant_initializer(0.0))
            b2_conv2 = tf.nn.conv2d(input=b2_conv1, filter=b2_conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b1_conv2')
            b2_conv2 = tf.nn.bias_add(value=b2_conv2, bias=b2_conv2_b, name='b2_conv2_bias_add')
            b2_conv2 = tf.contrib.layers.batch_norm(inputs=b2_conv2, decay=0.90, epsilon=1e-5, scope='b2_conv2_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b2_conv2 = tf.nn.relu(features=b2_conv2, name='b2_conv2_ReLU')

            # 2º Down-scale
            ds2 = tf.nn.max_pool(value=b2_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='ds2')

            # Third Block
            b3_conv1_w = tf.get_variable('b3_conv1_w', shape=[3, 3, self.nf_dim*2, self.nf_dim * 4], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b3_conv1_b = tf.get_variable('b3_conv1_b', shape=[self.nf_dim * 4],
                                         initializer=tf.constant_initializer(0.0))
            b3_conv1 = tf.nn.conv2d(input=ds2, filter=b3_conv1_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b3_conv1')
            b3_conv1 = tf.nn.bias_add(value=b3_conv1, bias=b3_conv1_b, name='b3_conv1_bias_add')
            b3_conv1 = tf.contrib.layers.batch_norm(inputs=b3_conv1, decay=0.90, epsilon=1e-5, scope='b3_conv1_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b3_conv1 = tf.nn.relu(features=b3_conv1, name='b3_conv1_ReLU')

            b3_conv2_w = tf.get_variable('b3_conv2_w', shape=[3, 3, self.nf_dim * 4, self.nf_dim * 4], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b3_conv2_b = tf.get_variable('b3_conv2_b', shape=[self.nf_dim * 4],
                                         initializer=tf.constant_initializer(0.0))
            b3_conv2 = tf.nn.conv2d(input=b3_conv1, filter=b3_conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b3_conv2')
            b3_conv2 = tf.nn.bias_add(value=b3_conv2, bias=b3_conv2_b, name='b3_conv2_bias_add')
            b3_conv2 = tf.contrib.layers.batch_norm(inputs=b3_conv2, decay=0.90, epsilon=1e-5, scope='b3_conv2_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b3_conv2 = tf.nn.relu(features=b3_conv2, name='b3_conv2_ReLU')

            # 3º Down-scale
            ds3 = tf.nn.max_pool(value=b3_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='ds3')

            # Forth Block
            b4_conv1_w = tf.get_variable('b4_conv1_w', shape=[3, 3, self.nf_dim * 4, self.nf_dim * 8], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b4_conv1_b = tf.get_variable('b4_conv1_b', shape=[self.nf_dim * 8],
                                         initializer=tf.constant_initializer(0.0))
            b4_conv1 = tf.nn.conv2d(input=ds3, filter=b4_conv1_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b4_conv1')
            b4_conv1 = tf.nn.bias_add(value=b4_conv1, bias=b4_conv1_b, name='b4_conv1_bias_add')
            b4_conv1 = tf.contrib.layers.batch_norm(inputs=b4_conv1, decay=0.90, epsilon=1e-5, scope='b4_conv1_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b4_conv1 = tf.nn.relu(features=b4_conv1, name='b4_conv1_ReLU')

            b4_conv2_w = tf.get_variable('b4_conv2_w', shape=[3, 3, self.nf_dim * 8, self.nf_dim * 8], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b4_conv2_b = tf.get_variable('b4_conv2_b', shape=[self.nf_dim * 8],
                                         initializer=tf.constant_initializer(0.0))
            b4_conv2 = tf.nn.conv2d(input=b4_conv1, filter=b4_conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b4_conv2')
            b4_conv2 = tf.nn.bias_add(value=b4_conv2, bias=b4_conv2_b, name='b4_conv2_bias_add')
            b4_conv2 = tf.contrib.layers.batch_norm(inputs=b4_conv2, decay=0.90, epsilon=1e-5, scope='b4_conv2_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b4_conv2 = tf.nn.relu(features=b4_conv2, name='b4_conv2_ReLU')

            # 4º Down-scale
            ds4 = tf.nn.max_pool(value=b4_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='ds4')

            # Fifth Block

            flat = tf.reshape(tensor=ds4,shape=[self.batch_size, self.nf_dim*8*w16*h16],name='flat')
            fc_w = tf.get_variable(name='fc_w',shape=[self.nf_dim*8 * w16 * h16, 8])
            fc_b = tf.get_variable('fc_b', shape=[8],
                                         initializer=tf.constant_initializer(0.0))
            fc = tf.add(tf.matmul(flat, fc_w) , fc_b, name='fc')
            fc = tf.nn.sigmoid(fc, name='encoder_activation')

            return fc, b2_conv1, b2_conv2, b3_conv2, b4_conv2

    def btp_decoder(self, bezier_points, reuse_variables=None, is_training = None):
        with tf.variable_scope('btp_decoder', reuse=reuse_variables) as scope:
            w16, w8, w4, w2, w1 = self.output_width // 16, self.output_width // 8, self.output_width // 4, self.output_width // 2, self.output_width
            h16, h8, h4, h2, h1 = self.output_height // 16, self.output_height // 8, self.output_height // 4, self.output_height // 2, self.output_height
            stddev = 0.02

            # First Block
            fc_w = tf.get_variable('fc_w', shape=[8, w16*h16*self.nfd_dim*8], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            fc_b = tf.get_variable('fc_b', shape=[w16*h16*self.nfd_dim*8], initializer=tf.constant_initializer(0.0))
            fc = tf.nn.bias_add(tf.matmul(bezier_points,fc_w),fc_b,name='fc')
            fc = tf.reshape(tensor=fc,shape=[self.batch_size,w16,h16,8*self.nfd_dim],name='fc_reshape')

            # Up-sampling 1
            us1_w = tf.get_variable(name='us1_w',shape=[2,2,8*self.nfd_dim,8*self.nfd_dim], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            us1_b = tf.get_variable('us1_b', shape=[8 * self.nfd_dim],
                                   initializer=tf.constant_initializer(0.0))
            us1 = tf.nn.conv2d_transpose(value=fc, filter=us1_w, output_shape=[self.batch_size,w8,h8,self.nfd_dim*8],
                                         strides=[1,2,2,1], name='us1')
            us1 = tf.nn.bias_add(us1,us1_b,name='us1_add_bias')

            # Second Block
            b2_conv1_w = tf.get_variable('b2_conv1_w', shape=[3, 3, self.nfd_dim * 8, self.nfd_dim * 8], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b2_conv1_b = tf.get_variable('b2_conv1_b', shape=[self.nfd_dim * 8],
                                         initializer=tf.constant_initializer(0.0))
            b2_conv1 = tf.nn.conv2d(input=us1, filter=b2_conv1_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b2_conv1')
            b2_conv1 = tf.nn.bias_add(value=b2_conv1, bias=b2_conv1_b, name='b2_conv1_bias_add')
            b2_conv1 = tf.contrib.layers.batch_norm(inputs=b2_conv1, decay=0.90, epsilon=1e-5, scope='b21_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b2_conv1 = tf.nn.relu(features=b2_conv1, name='b1_conv1_ReLU')

            b2_conv2_w = tf.get_variable('b2_conv2_w', shape=[3, 3, self.nfd_dim * 8, self.nfd_dim * 4], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b2_conv2_b = tf.get_variable('b2_conv2_b', shape=[self.nfd_dim * 4],
                                         initializer=tf.constant_initializer(0.0))
            b2_conv2 = tf.nn.conv2d(input=b2_conv1, filter=b2_conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b1_conv2')
            b2_conv2 = tf.nn.bias_add(value=b2_conv2, bias=b2_conv2_b, name='b2_conv2_bias_add')
            b2_conv2 = tf.contrib.layers.batch_norm(inputs=b2_conv2, decay=0.90, epsilon=1e-5, scope='b22_bn',
                                               is_training=is_training,
                                               scale=True, updates_collections=None)
            b2_conv2 = tf.nn.relu(features=b2_conv2, name='b2_conv2_ReLU')

            # Up-sampling 2
            us2_w = tf.get_variable(name='us2_w', shape=[2, 2, 4 * self.nfd_dim, 4 * self.nfd_dim], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
            us2_b = tf.get_variable('us2_b', shape=[4 * self.nfd_dim],
                                    initializer=tf.constant_initializer(0.0))
            us2 = tf.nn.conv2d_transpose(value=b2_conv2, filter=us2_w, output_shape=[self.batch_size, w4, h4, self.nfd_dim * 4],
                                         strides=[1, 2, 2, 1], name='us2')
            us2 = tf.nn.bias_add(us2, us2_b, name='us2_add_bias')

            # Third Block
            b3_conv1_w = tf.get_variable('b3_conv1_w', shape=[3, 3, self.nfd_dim * 4, self.nfd_dim * 4], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b3_conv1_b = tf.get_variable('b3_conv1_b', shape=[self.nfd_dim * 4],
                                         initializer=tf.constant_initializer(0.0))
            b3_conv1 = tf.nn.conv2d(input=us2, filter=b3_conv1_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b3_conv1')
            b3_conv1 = tf.nn.bias_add(value=b3_conv1, bias=b3_conv1_b, name='b3_conv1_bias_add')
            b3_conv1 = tf.contrib.layers.batch_norm(inputs=b3_conv1, decay=0.90, epsilon=1e-5, scope='b31_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b3_conv1 = tf.nn.relu(features=b3_conv1, name='b3_conv1_ReLU')

            b3_conv2_w = tf.get_variable('b3_conv2_w', shape=[3, 3, self.nfd_dim * 4, self.nfd_dim * 2], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b3_conv2_b = tf.get_variable('b3_conv2_b', shape=[self.nfd_dim * 2],
                                         initializer=tf.constant_initializer(0.0))
            b3_conv2 = tf.nn.conv2d(input=b3_conv1, filter=b3_conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b3_conv2')
            b3_conv2 = tf.nn.bias_add(value=b3_conv2, bias=b3_conv2_b, name='b3_conv2_bias_add')
            b3_conv2 = tf.contrib.layers.batch_norm(inputs=b3_conv2, decay=0.90, epsilon=1e-5, scope='b32_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b3_conv2 = tf.nn.relu(features=b3_conv2, name='b3_conv2_ReLU')

            # Up-sampling 3
            us3_w = tf.get_variable(name='us3_w', shape=[2, 2, 2 * self.nfd_dim, 2 * self.nfd_dim], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
            us3_b = tf.get_variable('us3_b', shape=[2 * self.nfd_dim],
                                    initializer=tf.constant_initializer(0.0))
            us3 = tf.nn.conv2d_transpose(value=b3_conv2, filter=us3_w,
                                         output_shape=[self.batch_size, w2, h2, self.nfd_dim * 2],
                                         strides=[1, 2, 2, 1], name='us3')
            us3 = tf.nn.bias_add(us3, us3_b, name='us3_add_bias')

            # Forth Block
            b4_conv1_w = tf.get_variable('b4_conv1_w', shape=[3, 3, self.nfd_dim * 2, self.nfd_dim*2], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b4_conv1_b = tf.get_variable('b4_conv1_b', shape=[self.nfd_dim*2],
                                         initializer=tf.constant_initializer(0.0))
            b4_conv1 = tf.nn.conv2d(input=us3, filter=b4_conv1_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b4_conv1')
            b4_conv1 = tf.nn.bias_add(value=b4_conv1, bias=b4_conv1_b, name='b4_conv1_bias_add')
            b4_conv1 = tf.contrib.layers.batch_norm(inputs=b4_conv1, decay=0.90, epsilon=1e-5, scope='b41_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b4_conv1 = tf.nn.relu(features=b4_conv1, name='b4_conv1_ReLU')

            b4_conv2_w = tf.get_variable('b4_conv2_w', shape=[3, 3, self.nfd_dim*2, self.nfd_dim], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b4_conv2_b = tf.get_variable('b4_conv2_b', shape=[self.nfd_dim],
                                         initializer=tf.constant_initializer(0.0))
            b4_conv2 = tf.nn.conv2d(input=b4_conv1, filter=b4_conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b4_conv2')
            b4_conv2 = tf.nn.bias_add(value=b4_conv2, bias=b4_conv2_b, name='b4_conv2_bias_add')
            b4_conv2 = tf.contrib.layers.batch_norm(inputs=b4_conv2, decay=0.90, epsilon=1e-5, scope='b42_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b4_conv2 = tf.nn.relu(features=b4_conv2, name='b4_conv2_ReLU')

            # Up-sampling 4
            us4_w = tf.get_variable(name='us4_w', shape=[2, 2, self.nfd_dim, self.nfd_dim], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
            us4_b = tf.get_variable('us4_b', shape=[ self.nfd_dim],
                                    initializer=tf.constant_initializer(0.0))
            us4 = tf.nn.conv2d_transpose(value=b4_conv2, filter=us4_w,
                                         output_shape=[self.batch_size, w1, h1, self.nfd_dim],
                                         strides=[1, 2, 2, 1], name='us4')
            us4 = tf.nn.bias_add(us4, us4_b, name='us4_add_bias')

            # Fifth Block
            b5_conv1_w = tf.get_variable('b5_conv1_w', shape=[3, 3, self.nfd_dim, self.nfd_dim], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b5_conv1_b = tf.get_variable('b5_conv1_b', shape=[self.nfd_dim],
                                         initializer=tf.constant_initializer(0.0))
            b5_conv1 = tf.nn.conv2d(input=us4, filter=b5_conv1_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b5_conv1')
            b5_conv1 = tf.nn.bias_add(value=b5_conv1, bias=b5_conv1_b, name='b5_conv1_bias_add')
            b5_conv1 = tf.contrib.layers.batch_norm(inputs=b5_conv1, decay=0.90, epsilon=1e-5, scope='b51_bn',
                                                    is_training=is_training,
                                                    scale=True, updates_collections=None)
            b5_conv1 = tf.nn.relu(features=b5_conv1, name='b5_conv1_ReLU')

            b5_conv2_w = tf.get_variable('b5_conv2_w', shape=[3, 3, self.nfd_dim, 1], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            b5_conv2_b = tf.get_variable('b5_conv2_b', shape=[1],
                                         initializer=tf.constant_initializer(0.0))
            b5_conv2 = tf.nn.conv2d(input=b5_conv1, filter=b5_conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                    name='b5_conv2')
            b5_conv2 = tf.nn.bias_add(value=b5_conv2, bias=b5_conv2_b, name='b5_conv2_bias_add')

            return tf.sigmoid(b5_conv2, 'act')

    def train_encoder(self, config):
        # set optimizers
        with tf.variable_scope("optimizers"):
            self.encoder_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate,
                                                        beta1=config.beta1,epsilon=0.1, name='encoder_adam') \
                    .minimize(self.encoder_loss, var_list=self.encoder_vars)


        # initialize variables
        self.sess.run(tf.global_variables_initializer())

        # Load encoder Model
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load_encoder(config.btp_checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load Encoder SUCCESS")
        else:
            print(" [!] Load Encoder failed...")

        # Load Files
        image_paths = glob(os.path.join(config.btp_pixel_space_dataset, config.btp_image_file_format))
        image_paths_samples = glob(os.path.join(config.btp_pixel_space_dataset_test, config.btp_image_file_format))

        # Training
        for epoch in xrange(config.epoch):
            # Calculate number of iterations
            batch_idxs = len(image_paths) // config.batch_size
            random.shuffle(image_paths)
            random.shuffle(image_paths_samples)

            # For each iteration
            for idx in range(0, batch_idxs):
                # create b_batch and p_batch
                bs_batch, ps_batch = loadbatch(config.batch_size, start_index=idx*config.batch_size,
                                               paths=image_paths)
                ps_batch = btp_normalize_ps(ps_batch)
                bs_batch = btp_normalize_bs(bs_batch, 128.0)


                # Update encoder
                _,err_encoder = self.sess.run(fetches=[ self.encoder_optim, self.encoder_loss],
                                            feed_dict={self.b_placeholder: bs_batch, self.p_placeholder: ps_batch})

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, btp_loss: %.8f" \
                        % (epoch, config.epoch, idx, batch_idxs,
                        time.time() - start_time, err_encoder))

                # Save Training
                counter += 1
                if np.mod(counter, 500) == 2:
                    self.save_encoder(config.btp_checkpoint_dir, counter)
                    print("========SAVED==========")

    def train_decoder(self, config):
        # set optimizers
        with tf.variable_scope("optimizers"):
            self.decoder_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate,
                                                        beta1=config.beta1, epsilon=0.1, name='decoder_adam') \
                                .minimize(self.decoder_loss, var_list=self.decoder_vars)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

        # Load encoder-decoder Model
        counter = 1
        start_time = time.time()

        could_load, checkpoint_counter = self.load_encoder(config.btp_checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load Encoder SUCCESS")
        else:
            print(" [!] Load Encoder failed...")

        could_load, checkpoint_counter = self.load_decoder(config.btp_checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load Decoder SUCCESS")
        else:
            print(" [!] Load Decoder failed...")

        # Load Files
        image_paths = glob(os.path.join(config.btp_pixel_space_dataset, config.btp_image_file_format))
        image_paths_samples = glob(os.path.join(config.btp_pixel_space_dataset_test, config.btp_image_file_format))
        """
        # Send summary statistics to TensorBoard
        tf.summary.scalar('btp_loss', self.decoder_loss)
        tf.summary.image('pixel space generated', self.encoder_test);
        summary_merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir=logdir, graph=self.sess.graph)
        """
        # Training
        for epoch in xrange(config.epoch):
            # Calculate number of iterations
            batch_idxs = len(image_paths) // config.batch_size
            random.shuffle(image_paths)
            random.shuffle(image_paths_samples)

            # For each iteration
            for idx in range(0, batch_idxs):
                # create b_batch and p_batch
                bs_batch, ps_batch = loadbatch(config.batch_size, start_index=idx * config.batch_size,
                                               paths=image_paths)
                ps_batch = btp_normalize_ps(ps_batch)
                bs_batch = btp_normalize_bs(bs_batch, 128.0)

                # Update decoder
                _,err_decoder,ce = self.sess.run(
                                                fetches=[self.decoder_optim,self.decoder_loss, self.check_encoder],
                                                feed_dict={self.b_placeholder: bs_batch,
                                                            self.p_placeholder: ps_batch})

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, decoder_loss: %.6f"
                      ", check_encoder: %.6f" \
                        % (epoch, config.epoch, idx, batch_idxs,
                        time.time() - start_time, err_decoder, ce))


                # For each 20 iterations save some samples
                bs_samples, ps_samples = loadbatch(config.batch_size, start_index=0 * config.batch_size,
                                                       paths=image_paths)
                bs_samples = btp_normalize_bs(bs_samples, 128.0)

                if np.mod(counter, 20) == 1:
                    samples = self.sess.run(
                            [self.decoder_test],
                            feed_dict={
                                self.b_placeholder: bs_samples,

                        }
                    )
                    samples = np.array(samples)
                    samples = np.squeeze(samples, axis=0)
                    print(samples.shape)
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

                # Update TensorBoard with summary statistics
                """
                bs_batch_val, ps_batch_val = loadbatch(config.batch_size, start_index=0 * config.batch_size,
                                                        paths=image_paths)
                ps_batch_val = btp_normalize_ps(ps_batch_val)
                bs_batch_val = btp_normalize_bs(bs_batch_val, 128.0)
                if np.mod(counter, 10) == 1:
                    summary = self.sess.run(summary_merged,
                                            feed_dict={self.b_placeholder: ps_batch_val,
                                                        self.p_placeholder: bs_batch_val})
                    writer.add_summary(summary, counter)

                """
                """
                # For each 20 iterations save some samples
                if np.mod(counter, 20) == 1:
                    samples = self.sess.run(
                        [self.btp_sampler],
                        feed_dict={
                            self.b_placeholder: bs_samples,

                        }
                    )
                    samples = np.array(samples)

                    samples = np.squeeze(samples, axis=0)
                    print(samples.shape)
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
"""

                # Save Training
                counter += 1
                if np.mod(counter, 500) == 2:
                    self.save_decoder(config.btp_checkpoint_dir, counter)
                    print("========SAVED==========")

    def test_encoder(self, config):
        # Create test batch
        image_paths = glob(os.path.join(config.btp_pixel_space_dataset_test, config.btp_image_file_format))
        random.shuffle(image_paths)
        for i in range(2):
            bs_batch, ps_batch = loadbatch(config.batch_size, start_index=i * config.batch_size,
                                           paths=image_paths)
            ps_batch = btp_normalize_ps(ps_batch)
            bs_batch = btp_normalize_bs(bs_batch, 128.0)

            err_encoder, encoder = self.sess.run(fetches=[self.encoder_loss, self.encoder_p],
                                                    feed_dict={self.b_placeholder: bs_batch, self.p_placeholder: ps_batch})
            """
            for i in range(self.batch_size):
                print(encoder[i,:])
                print(bs_batch[i,:])
                print("==========")"""
            print(err_encoder)
            print(bs_batch[0,:])
            print(encoder[0,:])
            print("======")

    def test_decoder(self, config):
        # Create test batch
        image_paths = glob(os.path.join(config.btp_pixel_space_dataset_test, config.btp_image_file_format))
        for i in range(3):
            bs_batch, ps_batch = loadbatch(config.batch_size, start_index=i * config.batch_size,
                                           paths=image_paths)
            ps_batch = btp_normalize_ps(ps_batch)
            bs_batch = btp_normalize_bs(bs_batch, 128.0)

            samples = self.sess.run(fetches=[self.decoder_test],
                                                    feed_dict={self.b_placeholder: bs_batch, self.p_placeholder: ps_batch})
            samples = np.array(samples)
            samples = np.squeeze(samples, axis=0)
            save_images(samples, image_manifold_size(samples.shape[0]),
                        './{}/train_{:02f}.png'.format(config.sample_dir, time.time()))

    def save_encoder(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir_encoder)

        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

    def save_decoder(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir_decoder)

        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

    def load_decoder(self, checkpoint_dir):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='btp_decoder')
        weight_initiallizer = tf.train.Saver(var_list)
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir_decoder)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
          chkp.print_tensors_in_checkpoint_file(os.path.join(checkpoint_dir, ckpt_name), tensor_name='', all_tensors=False,all_tensor_names=True)
          weight_initiallizer.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
          counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
          print(" [*] Success to read {}".format(ckpt_name))
          return True, counter
        else:
          print(" [*] Failed to find a checkpoint")
          return False, 0

    def load_encoder(self, checkpoint_dir):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='btp_encoder')
        weight_initiallizer = tf.train.Saver(var_list)
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir_encoder)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
          chkp.print_tensors_in_checkpoint_file(os.path.join(checkpoint_dir, ckpt_name), tensor_name='', all_tensors=False,all_tensor_names=True)
          weight_initiallizer.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
          counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
          print(" [*] Success to read {}".format(ckpt_name))
          return True, counter
        else:
          print(" [*] Failed to find a checkpoint")
          return False, 0

    def get_samples(self, sample_dir, option=1):
        if option == 1:
            image_frame_dim = int(math.ceil(self.batch_size ** .5))
            z_batch_samples = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
            samples = self.sess.run(
                [self.gz],
                feed_dict={
                    self.z_placeholder: z_batch_samples,
                },
            )
            samples = np.array(samples)
            if samples.shape[0] == 1:
                samples = np.squeeze(samples, axis=0)
            save_images(samples, [image_frame_dim,image_frame_dim],
                        './{}/inference_{}.png'.format(sample_dir, time.time()))
        elif option == 2:
                z_batch = np.random.normal(loc=0.0, scale=1.0,
                                                   size=(self.batch_size, self.z_dim)).astype(np.float32)
                image_samples = self.sess.run(fetches=[self.gz_sampler],
                                              feed_dict={self.z_placeholder: z_batch})
                samples = np.squeeze(np.array(image_samples), axis=0)
                for i in xrange(self.batch_size):
                    scipy.misc.imsave('./{}/inference_{}.png'.format(sample_dir, time.time()), inverse_transform(samples[i,:,:,:]))

    @property
    def model_dir_decoder(self):
        return "{}_{}_{}".format(
            self.batch_size,
            self.output_height, self.output_width)

    @property
    def model_dir_encoder(self):
        return "{}_{}_{}".format(
            36,
            self.output_height, self.output_width)