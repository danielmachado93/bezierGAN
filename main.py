import os
from glob import glob
import tensorflow as tf
from model import BTGAN

flags = tf.app.flags
# Train Parameters◘
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", 9000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
# Samples Parameters
flags.DEFINE_boolean("grayscale", True, "is gray - by definition")
flags.DEFINE_string("dataset", "dataset", "The name of dataset in *data* folder [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("ckpt_decoder_dir", "16_128_128_decoder", "Directory of BTP decoder ckpt")
flags.DEFINE_integer("sample_num", 1, "number of image samples [samples]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_integer("input_height", 128, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 128, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
# BT-GAN Parameters
flags.DEFINE_integer("output_height", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 128, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("bezier_manifold_dim", 8, "bezier manifold dim || min n = 8 (N = 64 curves ), N=2^n structure")

FLAGS = flags.FLAGS

def main():
    if FLAGS.grayscale:
        c_dim = 1
    else:
        c_dim = 3
    # Define Data for training
    data = glob(os.path.join("./data", FLAGS.dataset, FLAGS.input_fname_pattern))
    
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        bt_gan = BTGAN(
            sess,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            bezier_num= FLAGS.bezier_manifold_dim,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            c_dim=c_dim,
            checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.train:
            bt_gan.train(FLAGS, data)
        else: # INFERENCE
            if not bt_gan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            # Render samples to "samples" folder
            # Option 1 render manifold of samples with dim = n*n = number_of_samples
            # Option 2 render imagens one by one
            sess.run(tf.global_variables_initializer())
            bt_gan.get_samples(sample_dir=FLAGS.sample_dir, option=3)

        print("====DONE=====")

main()

