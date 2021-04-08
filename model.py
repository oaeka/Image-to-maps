from keras import layers
from keras import initializers
from keras import Model
from keras import optimizers
from numpy import load
from numpy.random import randint
from numpy import ones
from numpy import zeros
from matplotlib import pyplot
from keras.models import load_model


def define_discriminator(image_shape):
    init = initializers.RandomNormal(stddev=0.02)

    in_src_image = layers.Input(shape=image_shape)
    in_target_image = layers.Input(shape=image_shape)

    merged = layers.Concatenate()([in_src_image, in_target_image])

    d = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = layers.LeakyReLU(alpha=0.2)(d)

    d = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)

    d = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)

    d = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)

    d = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)

    d = layers.Conv2D(1, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    patch_out = layers.Activation('sigmoid')(d)

    model = Model([in_src_image, in_target_image], patch_out)
    opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = initializers.RandomNormal(stddev=0.02)
    g = layers.Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)

    if batchnorm:
        g = layers.BatchNormalization()(g, training=True)
    g = layers.LeakyReLU(alpha=0.2)(g)
    return g


def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = initializers.RandomNormal(stddev=0.02)
    g = layers.Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = layers.BatchNormalization()(g, training=True)

    if dropout:
        g = layers.Dropout(0.5)(g, training=True)
    g = layers.Concatenate()([g, skip_in])
    g = layers.Activation('relu')(g)

    return g


def define_generator(image_shape=(256, 256, 3)):
    init = initializers.RandomNormal(stddev=0.02)

    in_image = layers.Input(shape=image_shape)

    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)

    b = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = layers.Activation('relu')(b)

    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    g = layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = layers.Activation('tanh')(g)

    model = Model(in_image, out_image)
    return model


def define_gan(g_model, d_model, image_shape):
    for layer in d_model.layers:
        if not isinstance(layers, layers.BatchNormalization):
            layer.trainable = False

    in_src = layers.Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    model = Model(in_src, [dis_out, gen_out])

    opt = optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


def load_real_samples(filename):
    data = load(filename)

    X1, X2 = data['arr_0'], data['arr_1']

    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset

    ix = randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]

    y = ones((n_samples, patch_shape, patch_shape, 1))
    print(y.shape)
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize_performance(step, g_model, dataset, n_samples=3):
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)

    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])

    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])

    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])

    filename1 = 'plot_%06d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()

    filename2 = 'model_%06d.h5' % (step + 1)
    g_model.save(filename2)
    print("> Saved : %s and %s " % (filename1, filename2))


def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        print("> %d, d1[%.3f] d2[.%3f] g[%.3f]" % (i + 1, d_loss1, d_loss2, g_loss))

        # if (i + 1) % (bat_per_epo * 10) == 0:
        summarize_performance(i, g_model, dataset)
        break


dataset = load_real_samples('maps_256.npz')
print("Loaded ", dataset[0].shape, dataset[1].shape)

image_shape = dataset[0].shape[1:]

d_model = define_discriminator(image_shape)
g_model = load_model('/content/drive/My Drive/Model/model_054800.h5')

gan_model = define_gan(g_model, d_model, image_shape)
# gan_model.summary()
train(d_model, g_model, gan_model, dataset)
