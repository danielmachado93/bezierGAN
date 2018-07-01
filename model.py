import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.python.tools import inspect_checkpoint as chkp

from utils import *
from random import shuffle

class BTGAN(object):
    def __init__(self, sess,output_height=128, output_width=128,
                            bezier_num = 16, btp_batch_size = 1,
                             batch_size=36, sample_num = 64,
                             z_dim=100, gf_dim=64, df_dim=64,
                             c_dim=1,
                             checkpoint_dir=None):

        self.sess = sess
        self.output_width = output_width
        self.output_height = output_height
        self.batch_size = batch_size
        self.btp_batch_size = btp_batch_size
        self.sample_num = sample_num
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir


        # Define Bezier Manifold
        # bezier_manifold_dim * bezier_manifold_dim = 4N (total bezier points - 4 points per curve)
        self.bezier_manifold_dim = 2 * bezier_num
        self.N = bezier_num*bezier_num

        self.model_setup()

    def model_setup(self):
        # Setup placeholders
        self.z_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim], name='z_placeholder')
        self.x_placeholder = tf.placeholder(tf.float32,
                                            shape=[None, self.output_width, self.output_height, self.c_dim],
                                            name='x_placeholder')
        self.b_placeholder = tf.placeholder(tf.float32,
                                            shape=[None, 8],
                                            name='b_placeholder')

        # Evaluate Discriminator and Generator and Sampler Generator
        self.btp_decoder(bezier_points=self.b_placeholder,reuse_variables=False,is_training=False)
        self.dx, self.dx_logits = self.btgan_discriminator(input_images=self.x_placeholder, reuse_variables=False)
        self.gz,_ = self.btgan_generator(z=self.z_placeholder, is_train=True, reuse_variables=False)

        self.dgz, self.dgz_logits = self.btgan_discriminator(input_images=self.gz, reuse_variables=True)
        self.gz_sampler, self.bp = self.btgan_generator(z=self.z_placeholder, is_train=False, reuse_variables=True)

        # Define Losses
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dx_logits,
            labels=tf.ones_like(self.dx_logits),
            name='d_loss_real'))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dgz_logits,
            labels=tf.zeros_like(self.dgz_logits),
            name='d_loss_fake'))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.dgz_logits,
            labels=tf.ones_like(self.dgz_logits),
            name='d_loss_fake'))

        self.d_loss = self.d_loss_fake + self.d_loss_real

        # Trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'btgan_d_' in var.name]
        self.g_vars = [var for var in t_vars if 'btgan_g_' in var.name]

        # Saver
        self.saver = tf.train.Saver()

    def btgan_generator(self, z, is_train, reuse_variables):
        with tf.variable_scope('btgan_g_', reuse=reuse_variables) as scope:
            # 1º - DCGAN for generate bezier manifold points
            # =================================================================================
            # Fully Connect Layer [batch_size, z_dim] -> [batch_size, BM/16, BM/16, gf * 8] :: BM = Bezier Manifold dim
            g_w1 = tf.get_variable(name='g_w1', shape=[self.z_dim, (self.bezier_manifold_dim // 16) * (
                        self.bezier_manifold_dim // 16) * 8 * self.gf_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b1 = tf.get_variable('g_b1',
                                   shape=[(self.bezier_manifold_dim / 16) * (self.bezier_manifold_dim / 16) * 8 * self.gf_dim],
                                   initializer=tf.constant_initializer(0.0))
            g1 = tf.add(tf.matmul(z, g_w1), g_b1, name='g_fc')

            g1 = tf.reshape(g1,
                            shape=[-1, self.bezier_manifold_dim // 16, self.bezier_manifold_dim // 16, 8 * self.gf_dim],
                            name='g1_reshape')
            g1 = tf.contrib.layers.batch_norm(inputs=g1, decay=0.90, epsilon=1e-5, scope='g1_bn', is_training=is_train,
                                              scale=True, updates_collections=None)
            g1 = tf.nn.relu(g1, name='g1_ReLu')


            # First Deconv Layer [batch_size, BM/16, BM/16, gf * 8] -> [batch_size, BM/8, BM/8, gf * 4]
            g_w2 = tf.get_variable('g_w2', shape=[5, 5, self.gf_dim * 4, self.gf_dim * 8],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b2 = tf.get_variable('g_b2', shape=[4 * self.gf_dim],
                                   initializer=tf.constant_initializer(0.0))
            g2 = tf.nn.conv2d_transpose(g1, filter=g_w2,
                                        output_shape=[self.batch_size, self.bezier_manifold_dim // 8, self.bezier_manifold_dim // 8,
                                                      self.gf_dim * 4],
                                        strides=[1, 2, 2, 1])
            g2 = tf.nn.bias_add(g2, g_b2)
            g2 = tf.contrib.layers.batch_norm(inputs=g2, decay=0.90, epsilon=1e-5, scope='g2_bn', is_training=is_train,
                                              scale=True, updates_collections=None)
            g2 = tf.nn.relu(g2, name='g2_ReLu')

            # Second Deconv Layer [batch_size, BM/8, BM/8, gf * 4] -> [batch_size, BM/4, BM/4, gf * 2]
            g_w3 = tf.get_variable('g_w3', shape=[5, 5, self.gf_dim * 2, self.gf_dim * 4],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b3 = tf.get_variable('g_b3', shape=[2 * self.gf_dim],
                                   initializer=tf.constant_initializer(0.0))
            g3 = tf.nn.conv2d_transpose(g2, filter=g_w3,
                                        output_shape=[self.batch_size, self.bezier_manifold_dim // 4, self.bezier_manifold_dim // 4,
                                                      self.gf_dim * 2],
                                        strides=[1, 2, 2, 1])
            g3 = tf.nn.bias_add(g3, g_b3)
            g3 = tf.contrib.layers.batch_norm(inputs=g3, decay=0.90, epsilon=1e-5, scope='g3_bn', is_training=is_train,
                                              scale=True, updates_collections=None)
            g3 = tf.nn.relu(g3, name='g3_ReLu')

            # Third Deconv Layer [batch_size, BM/4, BM/4, gf * 2] -> [batch_size, BM/2, BM/2, gf]
            g_w4 = tf.get_variable('g_w4', shape=[5, 5, self.gf_dim, self.gf_dim * 2],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b4 = tf.get_variable('g_b4', shape=[self.gf_dim],
                                   initializer=tf.constant_initializer(0.0))
            g4 = tf.nn.conv2d_transpose(g3, filter=g_w4,
                                        output_shape=[self.batch_size, self.bezier_manifold_dim // 2, self.bezier_manifold_dim // 2,
                                                      self.gf_dim],
                                        strides=[1, 2, 2, 1])
            g4 = tf.nn.bias_add(g4, g_b4)
            g4 = tf.contrib.layers.batch_norm(inputs=g4, decay=0.90, epsilon=1e-5, scope='g4_bn', is_training=is_train,
                                              scale=True, updates_collections=None)
            g4 = tf.nn.relu(g4, name='g4_ReLu')

            # Fourth Deconv Layer [batch_size, BM/2, BM/2, gf] -> [batch_size, BM, BM, 2]
            g_w5 = tf.get_variable('g_w5', shape=[5, 5, 2, self.gf_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b5 = tf.get_variable('g_b5', shape=[2],
                                   initializer=tf.constant_initializer(0.0))
            g5 = tf.nn.conv2d_transpose(g4, filter=g_w5,
                                        output_shape=[self.batch_size, self.bezier_manifold_dim, self.bezier_manifold_dim,
                                                      2], # 2 coordinates points
                                        strides=[1, 2, 2, 1])
            g5 = tf.nn.bias_add(g5, g_b5)
            g5 = tf.tanh(g5, 'g_out')  #[-1,1] coordinate - space

        # 2º - Bezier Transformer Layer - transform activated bezier manifold to pixel space manifold
        # ===========================================================================================

        #  Transform from [-1,1] --> [0,1]
        BP = (g5 + 1.0)/2.0

        # Bezier Points Space
        BP_0 = tf.reshape(BP, shape=[self.batch_size,self.N, 8],name='bp_reshape') #---->[B*N,2*4]
        PS = None
        PS_pb = None # Pixel Space Per Batch
        # Map to Pixel Space Manifold
        BP = tf.unstack(BP_0,axis=0,name="unstack1")
        BP = tf.unstack(BP, axis=0,name="unstack2")
        for i in range(self.batch_size):
            for j in range(self.N):
                bezier_points = tf.reshape(BP[i][j],shape=[1,8])
                curve = self.btp_decoder(bezier_points=bezier_points,reuse_variables=True,is_training=False)
                if(j==0):
                    PS_pb = curve
                elif(j!=0):
                    PS_pb = tf.concat(values=[PS_pb,curve],axis=0)
            PS_pb = tf.reduce_sum(PS_pb,axis=0)
            PS_pb = tf.expand_dims(PS_pb,axis=0)
            if(i==0):
                PS = PS_pb
            else:
                PS = tf.concat(values=[PS,PS_pb],axis=0) # concat N times
        return PS, BP_0# --> [B,W,H,1] e [0.0,1.0]

    def btgan_discriminator(self, input_images, reuse_variables=None):
        with tf.variable_scope('btgan_d_', reuse=reuse_variables) as scope:
            # Same discriminator of DCGAN ARTICLE
            # Use Batch-norm. Avoid apply batch norm to the discriminator first layer
            # Use lReLU act in discriminator for all layers
            # =================================================================================
            # First convolutional layer - input [batch_size, W, H, c_dim] -> output [batch_size, W/2, H/2, df_dim]
            d_w1 = tf.get_variable('d_w1', [5, 5, self.c_dim, self.df_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b1 = tf.get_variable('d_b1',[self.df_dim], initializer=tf.constant_initializer(0.0))
            d1 = tf.nn.conv2d(input=input_images, filter=d_w1, strides = [1,2,2,1], padding='SAME')
            d1 = tf.add(d1,d_b1)
            d1 = tf.nn.leaky_relu(features=d1,alpha=0.2,name='d1_LReLU')

            # Second convolutional layer - input [batch_size, W/2, H/2, df_dim] -> [batch_size, W/4, H/4, df_dim*2]
            d_w2 = tf.get_variable('d_w2', [5, 5, self.df_dim, self.df_dim*2],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b2 = tf.get_variable('d_b2', [self.df_dim*2], initializer=tf.constant_initializer(0.0))
            d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 2, 2, 1], padding='SAME')
            d2 = tf.nn.bias_add(d2,d_b2)
            d2 = tf.contrib.layers.batch_norm(inputs=d2, decay=0.90, epsilon=1e-5,scope='d2_bn',is_training=True,
                                              scale=True, updates_collections=None)
            d2 = tf.nn.leaky_relu(features=d2, alpha=0.2, name='d2_LReLU')

            # Third convolutional layer - input [batch_size, W/4, H/4, df_dim*2] -> [batch_size, W/8, H/8, df_dim*4]
            d_w3 = tf.get_variable('d_w3', [5, 5, self.df_dim*2, self.df_dim*4],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b3 = tf.get_variable('d_b3', [self.df_dim*4], initializer=tf.constant_initializer(0.0))
            d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 2, 2, 1], padding='SAME')
            d3 = tf.nn.bias_add(d3, d_b3)
            d3 = tf.contrib.layers.batch_norm(inputs=d3, decay=0.90, epsilon=1e-5, scope='d3_bn',is_training=True,
                                              scale=True, updates_collections=None)
            d3 = tf.nn.leaky_relu(features=d3, alpha=0.2, name='d3_LReLU')

            # Four convolutional layer - [batch_size, W/8, H/8, df_dim*4] -> [batch_size, W/16, H/16, df_dim*8]
            d_w4 = tf.get_variable('d_w4', [5, 5, self.df_dim * 4, self.df_dim * 8],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b4 = tf.get_variable('d_b4', [self.df_dim * 8], initializer=tf.constant_initializer(0.0))
            d4 = tf.nn.conv2d(input=d3, filter=d_w4, strides=[1, 2, 2, 1], padding='SAME')
            d4 = tf.nn.bias_add(d4 , d_b4)
            d4 = tf.contrib.layers.batch_norm(inputs=d4, decay=0.90, epsilon=1e-5, scope='d4_bn',is_training=True,
                                              scale=True, updates_collections=None)
            d4 = tf.nn.leaky_relu(features=d4, alpha=0.2, name='d4_LReLU')

            d_flat = tf.layers.flatten(d4, name='d_flat')

            # Fully Connected Layer
            d_w5 = tf.get_variable('d_w5', [(self.output_height/16)*(self.output_width/16)*self.df_dim*8, 1],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b5 = tf.get_variable('d_b5', [1],
                                   initializer=tf.constant_initializer(0.0))
            d5 = tf.add(tf.matmul(d_flat, d_w5), d_b5)

            return tf.nn.sigmoid(d5, name='d_out_sigmoid'), d5

    def btp_decoder(self, bezier_points, reuse_variables=None, is_training = None):
        with tf.variable_scope('btp_decoder', reuse=reuse_variables) as scope:
            w16, w8, w4, w2, w1 = self.output_width // 16, self.output_width // 8, self.output_width // 4, self.output_width // 2, self.output_width
            h16, h8, h4, h2, h1 = self.output_height // 16, self.output_height // 8, self.output_height // 4, self.output_height // 2, self.output_height
            self.nfd_dim = 64
            stddev = 0.02

            # First Block
            fc_w = tf.get_variable('fc_w', shape=[8, w16*h16*self.nfd_dim*8], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            fc_b = tf.get_variable('fc_b', shape=[w16*h16*self.nfd_dim*8], initializer=tf.constant_initializer(0.0))
            fc = tf.nn.bias_add(tf.matmul(bezier_points,fc_w),fc_b,name='fc')
            fc = tf.reshape(tensor=fc,shape=[self.btp_batch_size,w16,h16,8*self.nfd_dim],name='fc_reshape')

            # Up-sampling 1
            us1_w = tf.get_variable(name='us1_w',shape=[2,2,8*self.nfd_dim,8*self.nfd_dim], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
            us1_b = tf.get_variable('us1_b', shape=[8 * self.nfd_dim],
                                   initializer=tf.constant_initializer(0.0))
            us1 = tf.nn.conv2d_transpose(value=fc, filter=us1_w, output_shape=[self.btp_batch_size,w8,h8,self.nfd_dim*8],
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
            us2 = tf.nn.conv2d_transpose(value=b2_conv2, filter=us2_w, output_shape=[self.btp_batch_size, w4, h4, self.nfd_dim * 4],
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
                                         output_shape=[self.btp_batch_size, w2, h2, self.nfd_dim * 2],
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
                                         output_shape=[self.btp_batch_size, w1, h1, self.nfd_dim],
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

    def train(self, config, data_paths):
        # set optimizers
        d_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, epsilon=0.1)\
                    .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1, epsilon=0.1)\
                    .minimize(self.g_loss, var_list=self.g_vars)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

        # Load Model if could load
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Load BezierToPixel Decoder
        could_load, checkpoint_counter = self.load_decoder(config.ckpt_decoder_dir,self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load BTP decoder SUCCESS")
        else:
            print(" [!] Load BTP decoder failed...")

        # Create Sample batch
        data = data_paths
        sample_files = data[0:self.sample_num]
        sample = [
            get_image(sample_file,
                      input_height=config.input_height,
                      input_width=config.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=config.crop,
                      grayscale=config.grayscale) for sample_file in sample_files]
        print(np.array(sample).shape)
        if (config.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)
        sample_inputs = (sample_inputs + 1.0) * 0.5
        x_batch_samples = sample_inputs
        z_batch_samples = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)


        # Training
        for epoch in xrange(config.epoch):
            # Shuffle Data
            shuffle(data)

            # Calculate number of iterations
            batch_idxs = min(len(data), config.train_size) // config.batch_size

            # For each iteration
            for idx in xrange(0, batch_idxs):
                # create z_batch and x_batch
                batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [
                    get_image(batch_file,
                              input_height=config.input_height,
                              input_width=config.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              crop=config.crop,
                              grayscale=config.grayscale) for batch_file in batch_files]
                if config.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_images = (batch_images + 1.0) * 0.5
                z_batch = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
                x_batch = batch_images

                # Update D net
                errD, _ = self.sess.run(fetches=[self.d_loss, d_optim],
                                        feed_dict={self.x_placeholder: x_batch, self.z_placeholder: z_batch})
                # Update 2 times G net
                errG, _ = self.sess.run(fetches=[self.g_loss, g_optim],
                                        feed_dict={self.z_placeholder: z_batch})
                errG, _ = self.sess.run(fetches=[self.g_loss, g_optim],
                                        feed_dict={self.z_placeholder: z_batch})

                # Print some info
                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, config.epoch, idx, batch_idxs,
                         time.time() - start_time, errD, errG))

                # For each 50 iterations save some samples
                if np.mod(counter, 20) == 1:
                    samples, d_loss, g_loss, bezier_points_samples = self.sess.run(
                        [self.gz_sampler, self.d_loss, self.g_loss, self.bp],
                        feed_dict={
                            self.z_placeholder: z_batch_samples,
                            self.x_placeholder: x_batch_samples,
                        },
                    )
                    print(np.array(samples).shape)
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    #bezier_points_samples = np.squeeze(np.array(bezier_points_samples), axis=0)
                    np.savetxt('./{}/train_{:02d}_{:04d}.txt'.format(config.sample_dir, epoch, idx), bezier_points_samples[0], delimiter=",")

                # Save Training
                if np.mod(counter, 50) == 2:
                    self.save(config.checkpoint_dir, counter)
                    print("========SAVED==========")

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

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

    def load_decoder(self, decoder_ckpt_dir, checkpoint_dir):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='btp_decoder')
        weight_initiallizer = tf.train.Saver(var_list)
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, decoder_ckpt_dir)

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
        elif option == 3:
            z_batch = np.random.normal(loc=0.0, scale=1.0,
                                       size=(self.batch_size, self.z_dim)).astype(np.float32)
            image_samples = self.sess.run(fetches=[self.bp],
                                          feed_dict={self.z_placeholder: z_batch})
            samples = np.squeeze(np.array(image_samples), axis=0)
            np.savetxt("data.txt",samples[0],delimiter=",")


