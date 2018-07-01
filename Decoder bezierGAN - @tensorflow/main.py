import os
import scipy.misc
import numpy as np
from glob import glob
from model import bezier_to_pixel_transformer_net
from utils import pp, to_json, show_all_variables
import tensorflow as tf

flags = tf.app.flags
# Training parameters
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")

flags.DEFINE_integer("output_height", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 128, "The size of the output images to produce. If None, same value as output_height [None]")

flags.DEFINE_string("btp_pixel_space_dataset", "./data/bezier_dataset_1", "The name of dataset in *data* folder [celebA, mnist, lsun]")
flags.DEFINE_string("btp_pixel_space_dataset_test", "./data/bezier_dataset_test_1", "The name of dataset in *data* folder [celebA, mnist, lsun]")
flags.DEFINE_string("btp_image_file_format", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("btp_checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("btp_checkpoint_dir_encoder", "./checkpoint/36_128_128", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("mode", "decoder", "mode -> encoder/decoder")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

def main():

    with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
        btp_net = bezier_to_pixel_transformer_net(
            sess,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size)

        if FLAGS.train:
            if (FLAGS.mode == "encoder"):
                btp_net.train_encoder(FLAGS)
            elif (FLAGS.mode == "decoder"):
                btp_net.train_decoder(FLAGS)
        else: # INFERENCE
            if(FLAGS.mode=="encoder"):
                if not btp_net.load_encoder(FLAGS.btp_checkpoint_dir)[0]:
                    raise Exception("[!] Train a model first, then run test mode")
                # Render samples to "samples" folder
                # render manifold of samples with dim = n*n = number_of_samples
                btp_net.test_encoder(FLAGS)
            elif(FLAGS.mode == "decoder"):
                if not btp_net.load_decoder(FLAGS.btp_checkpoint_dir)[0]:
                    raise Exception("[!] Train a model first, then run test mode")
                # Render samples to "samples" folder
                # render manifold of samples with dim = n*n = number_of_samples
                btp_net.test_decoder(FLAGS)

        print("====DONE=====")
main()