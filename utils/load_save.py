import os

import tensorflow as tf


def save(saver, sess, model, checkpoint_dir, step):
    checkpoint_dir = os.path.join(checkpoint_dir, model.model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,
               os.path.join(checkpoint_dir, model.name),
               global_step=step)


def load(saver, sess, model, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False
