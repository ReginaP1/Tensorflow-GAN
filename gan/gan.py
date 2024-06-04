import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (Input, Conv2D, Flatten, Dense, Dropout, BatchNormalization, UpSampling2D,
                                     Reshape, Activation, LeakyReLU)
# import horovod.tensorflow.keras as hvd
# from horovod import spark
# import os
# import sys

batch_size = 32
data_path = '/home/regina/Python/dataset'
latent_dim = 250
data = keras.utils.image_dataset_from_directory(data_path, batch_size=batch_size, image_size=(512, 512),
                                                label_mode=None, shuffle=True)


def build_generator():
    inputs = Input(shape=(latent_dim,))
    x = Dense(4*4*256, activation='relu', kernel_regularizer='l1_l2')(inputs)
    x = Reshape((4, 4, 256))(x)

    for i in range(5):
        x = UpSampling2D()(x)
        x = Conv2D(512, kernel_size=3, padding='same', groups=128)(x)
        x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2D(512, kernel_size=3, padding='same', groups=128)(x)
    x = UpSampling2D()(x)
    x = Conv2D(3, kernel_size=3, padding='same', groups=128)(x)
    outputs = Activation(keras.activations.tanh)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def build_discriminator():
    inputs = Input(shape=(512, 512, 3))
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', groups=8)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', groups=16)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.1)(x)

    for o in range(1, 4):
        x = Conv2D(128*o, kernel_size=3, strides=2, padding='same', groups=32*o)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dropout(0.1)(x)

    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer='l1_l2')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


generator = build_generator()
generator.summary()
discriminator = build_discriminator()
discriminator.summary()

loss_fn = keras.losses.BinaryCrossentropy()

generator_optimizer = keras.optimizers.Adamax(1.5e-4, 0.5)
discriminator_optimizer = keras.optimizers.Adamax(1.5e-4, 0.5)

# generator_optimizer = hvd.DistributedOptimizer(generator_optimizer)
# discriminator_optimizer = hvd.DistributedOptimizer(discriminator_optimizer)


def train_step(batch_real_images, epoch1):  #callbacks
    noise = tf.random.normal(shape=(batch_size, latent_dim))
    generated_images = generator(noise, training=True)
    combined_images = tf.concat([generated_images, batch_real_images], axis=0)
    labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
    # add random noise to the labels
    labels += 0.05 * tf.random.uniform(tf.shape(labels))

    with tf.GradientTape() as tape:
        discriminator.trainable = True
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    misleading_labels = tf.zeros((batch_size, 1))

    with tf.GradientTape() as tape:
        discriminator.trainable = False
        predictions = discriminator(generator(noise, training=True))
        g_loss = loss_fn(misleading_labels, predictions)
        g_loss += 10 * tf.math.reduce_sum(tf.abs(generated_images))
    grads = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    # for callback in callbacks:
    #     callback.on_epoch_end()

    return 'Epoch:', epoch1, 'Discriminator loss:', d_loss, 'Generator loss:', g_loss


def train_hvd():
    # import tempfile
    # import os
    # import shutil
    # import atexit
    # from horovod.tensorflow.keras import callbacks
    #
    # hvd.init()
    #
    # gpu = tf.config.experimental.list_physical_devises("GPU")
    # tf.config.experimental.set_memory_growth(gpu, True)
    # if gpu:
    #     tf.config.experimental.set_visible_devices(gpu[hvd.local_rank()], "GPU")
    #
    # callbacks = [callbacks.BroadcastGlobalVariablesCallback(0),]
    # 
    # ckpt_dir = tempfile.mkdtemp()
    # ckpt_file = os.path.join(ckpt_dir, 'checkpoint.h5')
    # atexit.register(lambda: shutil.rmtree(ckpt_dir))
    #
    # if hvd.rank() == 0:
    #     callbacks.append(keras.callbacks.ModelCheckpoint(ckpt_file, monitor='loss', mode='min', save_best_only=True))

    for epoch in range(250):
        real_images = next(iter(data))
        # train_step(real_images, epoch, callbacks)
        train_step(real_images, epoch)

        # if hvd.rank() == 0:
        #     with open(ckpt_file, 'rb') as f:
        #         return f.read()
        #

# lr_single_node = 0.1
# num_proc = 3
#
# best_model_bytes = spark.run(train_hvd(), args=(lr_single_node,),
#                              num_proc=num_proc, env=os.environ.copy(),
#                              stdout=sys.stdout, stderr=sys.stderr,
#                              verbose=2, prefix_output_with_timestamp=True)[0]
#


def generate_and_save_images(model):
    img_num = 20
    noise = tf.random.normal(shape=(img_num, latent_dim))
    generated_image = model(noise, training=True)
    for o in range(img_num):
        img = keras.preprocessing.image.array_to_img(generated_image[o])
        img.save('/home/regina/Python/dataset/generated_images/image{:02d}.png'.format(o))


generate_and_save_images(generator)

generator.save('/home/regina/Python/generator.h5')
discriminator.save('/regina/home/Python/discriminator.h5')
