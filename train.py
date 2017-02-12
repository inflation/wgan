import os
import time
from pprint import pprint

import tensorflow as tf
import numpy as np

from dataset import ImageDataset
from models import WDCGAN
import utils

flags = tf.app.flags
flags.DEFINE_integer("iter", 64, "Iteration to train [64]")
flags.DEFINE_integer("d_iter", 5, "Iteration of discriminator to train [5]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_boolean("adam", False, "True for using Adam Optimizer, False for RMSProp [False]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("learning_rate_G", 5e-5, "Learning rate of G for RMSProp [0.00005]")
flags.DEFINE_float("learning_rate_D", 5e-5, "Learning rate of D for RMSProp [0.00005]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 128, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 128, "The size of the output images to produce [128]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("channels", 3, "Number of image channels. [3]")
flags.DEFINE_string("dataset", "data", "The name of the dataset [data]")
flags.DEFINE_string("dataset_path", None, "The path to the dataset [None]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save the logs [logs]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS


def main(_):
    pprint(flags.FLAGS.__flags)

    if FLAGS.dataset_path == None:
        raise AttributeError("Dataset path cannot be none!")

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    start_time = time.time()

    with tf.Session(config=run_config) as sess:
        model = WDCGAN(dataset_name=FLAGS.dataset,
                       output_height=FLAGS.output_height,
                       output_width=FLAGS.output_width,
                       channels=FLAGS.channels,
                       batch_size=FLAGS.batch_size)

        dataset = ImageDataset(data_path=FLAGS.dataset_path,
                               batch_size=FLAGS.batch_size,
                               output_height=FLAGS.input_height,
                               output_width=FLAGS.input_width,
                               channels=FLAGS.channels)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        model.build_model()

        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        if utils.load(saver, sess, model, FLAGS.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if FLAGS.adam:
            d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)\
                .minimize(model.d_loss, var_list=model.d_vars)
            g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)\
                .minimize(model.g_loss, var_list=model.g_vars)
        else:
            d_optim = tf.train.RMSPropOptimizer(FLAGS.learning_rate_D)\
                .minimize(model.d_loss, var_list=model.d_vars)
            g_optim = tf.train.RMSPropOptimizer(FLAGS.learning_rate_G)\
                .minimize(model.g_loss, var_list=model.g_vars)

        def next_batch():
            batch_image = sess.run(dataset.get_batch())
            batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size, model.z_dim]).astype(np.float32)
            threads = tf.train.start_queue_runners(coord=coord)

            return batch_image, batch_z

        sample_z = np.random.uniform(-1, 1, size=(model.sample_num, model.z_dim))
        sample_inputs = sess.run(dataset.get_image())

        sess.run(tf.global_variables_initializer())

        for i in range(FLAGS.iter):
            if i < 25 or i % 500 == 0:
                d_iters = 100
            else:
                d_iters = 5

            for j in range(d_iters):
                batch = next_batch()

                _, summary_str = sess.run([d_optim, model.d_sum],
                                           feed_dict={model.inputs: batch[0],
                                                      model.z: batch[1]})
                summary_writer.add_summary(summary_str, i)

            batch = next_batch()
            _, summary_str = sess.run([g_optim, model.g_sum],
                                      feed_dict={model.inputs: batch[0],
                                                 model.z: batch[1]})
            summary_writer.add_summary(summary_str, i)

            error_d_fake = model.d_loss_fake.eval({model.z: batch[1]})
            error_d_real = model.d_loss_real.eval({model.inputs: batch[0]})
            error_g = model.g_loss.eval({model.z: batch[1]})

            print("Iteration: [%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (i, time.time() - start_time, error_d_fake - error_d_real, error_g))

            if i % 100 == 1:
                try:
                    samples, d_loss, g_loss = sess.run(
                        [model.S, model.d_loss, model.g_loss],
                        feed_dict={
                            model.z: sample_z,
                            model.inputs: sample_inputs,
                        },
                    )
                    # TODO: Save images
                    # utils.save_images(samples, [8, 8],
                    #             './{}/train_{:02d}_{:04d}.png'.format(
                    #                 config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                except:
                    print("one pic error!...")

            if i % 500 == 2:
                utils.save(saver, sess, model, FLAGS.checkpoint_dir, i)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()