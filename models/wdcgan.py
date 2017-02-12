import tensorflow as tf
from layers import leaky_relu, conv2d, conv2d_trans, linear, BatchNorm


class WDCGAN():
    def __init__(self, dataset_name,
                 output_height=128, output_width=128, channels=3,
                 batch_size=128, sample_num=64,
                 z_dim=100,
                 generater_filter_dim=64, discriminator_filter_dim=64):

        self.name = "WDCGAN"
        self.dataset_name = dataset_name

        self.is_grayscale = (channels == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.channels = channels
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = generater_filter_dim
        self.df_dim = discriminator_filter_dim

        self.d_bn1 = BatchNorm(name='d_bn1')
        self.d_bn2 = BatchNorm(name='d_bn2')
        self.d_bn3 = BatchNorm(name='d_bn3')

        self.g_bn0 = BatchNorm(name='g_bn0')
        self.g_bn1 = BatchNorm(name='g_bn1')
        self.g_bn2 = BatchNorm(name='g_bn2')
        self.g_bn3 = BatchNorm(name='g_bn3')

        self.y = None
        self.inputs = None
        self.sample_inputs = None

        self.z = None
        self.z_sum = None

        self.G = None
        self.G_sum = None
        self.D = None
        self.D_sum = None

        self.S = None
        self.D_fake = None
        self.D_fake_sum = None

        self.g_loss = None
        self.g_loss_sum = None

        self.d_loss_real = None
        self.d_loss_real_sum = None
        self.d_loss_fake = None
        self.d_loss_fake_sum = None
        self.d_loss = None
        self.d_loss_sum = None

        self.g_vars = None
        self.d_vars = None

        self.g_sum = None
        self.d_sum = None

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            out_h, out_w = self.output_height, self.output_width
            out_h2, out_h4, out_h8, out_h16 = int(out_h / 2), int(out_h / 4), int(out_h / 8), int(out_h / 16)
            out_w2, out_w4, out_w8, out_w16 = int(out_w / 2), int(out_w / 4), int(out_w / 8), int(out_w / 16)

            # project `z` and reshape
            z_fake, h0_w, h0_b = linear(
                z, self.gf_dim * 8 * out_h16 * out_w16, 'g_h0_lin', with_w=True)

            h0 = tf.reshape(
                z_fake, [-1, out_h16, out_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1, h1_w, h1_b = conv2d_trans(
                h0, [self.batch_size, out_h8, out_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2, h2_w, h2_b = conv2d_trans(
                h1, [self.batch_size, out_h4, out_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, h3_w, h3_b = conv2d_trans(
                h2, [self.batch_size, out_h2, out_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, h4_w, h4_b = conv2d_trans(
                h3, [self.batch_size, out_h, out_w, self.channels], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            out_h, out_w = self.output_height, self.output_width
            out_h2, out_h4, out_h8, out_h16 = int(out_h / 2), int(out_h / 4), int(out_h / 8), int(out_h / 16)
            out_w2, out_w4, out_w8, out_w16 = int(out_w / 2), int(out_w / 4), int(out_w / 8), int(out_w / 16)

            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim * 8 * out_h16 * out_w16, 'g_h0_lin'),
                            [-1, out_h16, out_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = conv2d_trans(h0, [self.batch_size, out_h8, out_w8, self.gf_dim * 4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = conv2d_trans(h1, [self.batch_size, out_h4, out_w4, self.gf_dim * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = conv2d_trans(h2, [self.batch_size, out_h2, out_w2, self.gf_dim * 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = conv2d_trans(h3, [self.batch_size, out_h, out_w, self.channels], name='g_h4')

            return tf.nn.tanh(h4)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = leaky_relu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = leaky_relu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = leaky_relu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = leaky_relu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return h4

    def build_model(self):
        image_dims = [self.output_height, self.output_width, self.channels]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)
        self.D = self.discriminator(inputs)
        self.D_fake = self.discriminator(self.G, reuse=True)
        self.S = self.sampler(self.z)

        self.D_sum = tf.summary.histogram("d", self.D)
        self.D_fake_sum = tf.summary.histogram("d_fake", self.D_fake)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(self.D)
        self.d_loss_fake = tf.reduce_mean(self.D_fake)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = tf.reduce_mean(self.D_fake - self.D)
        self.g_loss = tf.reduce_mean(-self.D_fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.g_sum = tf.summary.merge([self.z_sum, self.D_fake_sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.D_sum, self.d_loss_real_sum, self.d_loss_sum])

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

if __name__ == '__main__':
    print('hi')
