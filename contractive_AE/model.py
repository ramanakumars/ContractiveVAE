from .globals import *

# A function to compute the value of latent space
def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch,dim))
    return mu + K.exp(sigma/2)*eps


class ContractiveLossMetric(tf.keras.metrics.Metric):
    def __init__(self, name='contractive', **kwargs):
        super(ContractiveLossMetric, self).__init__(name=name, **kwargs)
        self.contractive_loss = self.add_weight(name='loss', initializer='zeros')

    def update_state(self, y_true, y_pred, value=0, sample_weights=None):
        self.contractive_loss.assign_add(value)

    def result(self):
        return self.contractive_loss

    def reset_state(self):
        self.contractive_loss.assign(0)

c_loss_metric = ContractiveLossMetric()

class ContractiveVAE():

    def __init__(self, latent_dim=128, conv_filt=256, hidden=[]):
        self.latent_dim = latent_dim
        self.conv_filt  = conv_filt
        self.hidden     = hidden 

        hidden_name = ''
        for layeri in hidden:
            hidden_name += '_%d'%layeri

        self.name = 'cvae_%dls_conv%d%s'%(latent_dim, conv_filt, hidden_name)

        print(self.name)

    def create_model(self, sigma0=0., lam=1.e-3, beta=1.e-3):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        latent_dim = self.latent_dim

        ''' ENCODER '''
        input_shape = (128, 128, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')
        
        # convolution part
        enc_c1 = Conv2D(conv_filt, (3,3), padding='same', activation='relu')(encoder_input)
        enc_p1 = AveragePooling2D(pool_size=(2,2), padding='same')(enc_c1)
        enc_b1 = BatchNormalization()(enc_p1)
        
        enc_c2 = Conv2D(conv_filt, (3,3), padding='same', activation='relu')(enc_p1)
        enc_p2 = AveragePooling2D(pool_size=(2,2), padding='same')(enc_c2)
        enc_b2 = BatchNormalization()(enc_p2)
        
        enc_c3 = Conv2D(conv_filt, (3,3), padding='same', activation='relu')(enc_p2)
        enc_p3 = AveragePooling2D(pool_size=(2,2), padding='same')(enc_c3)
        enc_b3 = BatchNormalization()(enc_p3)

        input_conv = enc_p3
        
        # sampling and bottleneck 
        enc_inps = []
        enc_flat = Flatten(name='flat_inp')(enc_b3)
        enc_inps.append(enc_flat)

        for layeri in hidden:
            enc_inps.append(Dense(layeri, name='dense_%d'%layeri, activation='sigmoid')(enc_inps[-1]))
        mu = Dense(latent_dim, name='mu', activation=None)(enc_inps[-1])
        sigma = Dense(latent_dim, name='sig',  activation=None)(enc_inps[-1])

        latent_space = Lambda(compute_latent, output_shape=(latent_dim,), name='latent')([mu, sigma])
        
        # Build the encoder
        self.encoder = Model(encoder_input, [mu, sigma, latent_space], name='encoder')

        self.mu = mu; self.sigma = sigma; self.z = latent_space
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        last_inp = decoder_input = Input(shape=(latent_dim,), name='dec_inp')
        if len(hidden) > 0:
            for layeri in hidden[::-1]:
                last_inp = Dense(layeri, name='dense_dec_%d'%layeri, activation='sigmoid')(last_inp)
        dec2 = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(last_inp)
        dec3 = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(dec2)

        #dec_bn = BatchNormalization()(dec3)

        dec_u1 = UpSampling2D((2,2))(dec3)
        dec_c1 = Conv2DTranspose(filters=conv_filt, kernel_size=3,padding='same',activation='relu')(dec_u1)
        dec_b1 = BatchNormalization()(dec_c1)
        
        dec_u2 = UpSampling2D((2,2))(dec_c1)
        dec_c2 = Conv2DTranspose(filters=conv_filt, kernel_size=3,padding='same',activation='relu')(dec_u2)
        dec_b2 = BatchNormalization()(dec_c2)

        dec_u3 = UpSampling2D((2,2))(dec_c2)  
        decoder_output = Conv2DTranspose(filters=3, kernel_size=5, padding='same', activation='relu')(dec_u3)
        
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

        self.output = self.decoder(self.encoder(encoder_input)[2])
        
        self.encoder.summary()

        self.decoder.summary()
        
        # Build the VAE
        self.vae = Model(encoder_input, self.output, name='VAE')

        self.vae.encoder = self.encoder
        self.vae.decoder = self.decoder

        #self.vae.enc_conv_flat = Model(encoder_input, enc_inps[0], name='enc_conv_flat')
        #self.vae.enc_hidden    = Model(Input(tensor=enc_inps[0]), [mu, sigma, z], name='enc_hidden')

        # set the training parameters
        self.vae.sig0     = sigma0*K.ones_like(sigma)
        self.vae.c_lambda = lam
        self.vae.kl_beta  = beta
        
        self.vae.summary()
        
    def add_loss_funcs(self):
        self.name += "lam_%1e_sig%1f_beta_%1e"

        recon_loss = K.mean(K.square(self.input - self.output), axis=(1,2,3))*128*128

        mui, sigmai, z = self.encoder(self.input)
        mup, sigp, zp = self.encoder(self.output)

        # KL divergence loss
        # kl = 1 + sigmai + 4 (-K.square(mui-mup) - K.exp(sigmai))/(2*K.exp(4))
        kl = 1 - sigmai + self.vae.sig0 + (K.square(mui-mup) + K.exp(sigmai))/K.exp(self.vae.sig0)
        kl = K.sum(kl, axis=-1)
        kl *= 0.5*self.vae.kl_beta

        hidden = self.hidden
        for i, layeri in enumerate(hidden):
            # x1 = first dense layer
            x1 = self.vae.get_layer('encoder').get_layer('dense_%d'%layeri).output
            # W1 = weights of first dense layer
            W1 = self.vae.get_layer('encoder').get_layer('dense_%d'%layeri).weights[0]
            W1 = tf.expand_dims(K.transpose(W1), axis=0)

            # calculate dPhi(W1*x)/d(W1*x) = dz1
            x1_1  = tf.expand_dims(x1, axis=-1)
            x1_2 = tf.expand_dims(x1, axis=1)
            # print(enc2.shape, enc_expanded.shape, enc_expanded.shape, W1.shape)
            dz1 = K.batch_dot(x1_1, (1-x1_2))

            if i!=0:
                W1 = K.batch_dot(W1, dx1)
            # calculate dx1/dx = dx1
            dx1 = K.batch_dot(dz1, W1, axes=1)
        # print(tf.expand_dims(W1m, axis=0).shape, denc2.shape)

        # h = output layer
        h = self.vae.get_layer('encoder').get_layer('mu').output
        # W2 = output layer weights
        W2 = self.vae.get_layer('encoder').get_layer('mu').weights[0]
        W2 = tf.expand_dims(K.transpose(W2), axis=0)

        # calculate W = W2*dx1
        W = K.batch_dot(W2, dx1)

        # contractive loss = norm( dPhi(W2*x1)/d(W2*x1) * W )
        contractive_loss = self.vae.c_lambda*K.sum(K.ones_like(h)*K.sum(W**2, axis=2), axis=1)

        c_loss     = tf.nn.compute_average_loss(contractive_loss)
        r_loss     = tf.nn.compute_average_loss(recon_loss)
        kl_loss    = tf.nn.compute_average_loss(kl)    

        # sum of all three losses
        loss = r_loss + kl_loss + c_loss

        self.vae.add_loss(loss)
        self.vae.add_metric(r_loss, aggregation='mean', name='mse')
        self.vae.add_metric(kl_loss, aggregation='mean', name='kl')
        self.vae.add_metric(c_loss, aggregation='mean', name='contr')

    def compile(self, learning_rate=0.0001):
        opt = Adam(learning_rate=learning_rate)
        self.vae.compile(optimizer=opt)


    def train(self, data, epochs=300, batch_size=10):
        savesfolder = self.savesfolder =  'models-%s/'%self.name

        self.nepochs = epochs

        if not os.path.exists(savesfolder):
            os.mkdir(savesfolder)

        self.history = self.vae.fit(data, epochs=epochs, validation_split=0.1, 
                          validation_freq=5, batch_size=batch_size, shuffle=True)

    def save(self):
        savesfolder = self.savesfolder =  'models-%s/'%self.name
        self.encoder.save_weights(savesfolder+"encoderw.h5")
        self.decoder.save_weights(savesfolder+"decoderw.h5")
        self.vae.save_weights(savesfolder+"VAEw.h5")
    
    def load(self):
        savesfolder = self.savesfolder = 'models-%s/'%self.name
        self.encoder.load_weights(savesfolder+"encoderw.h5")
        self.decoder.load_weights(savesfolder+"decoderw.h5")
        self.vae.load_weights(savesfolder+"VAEw.h5")

'''
class VAE(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)


    @tf.function
    def call(self, inputs, training=False):
        return self.decoder(self.encoder(inputs)[2])

    @tf.function
    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            flat_inp = self.enc_conv_flat(data)
            mu       = self.enc_hidden(flat_inp)

            # forward pass
            out = self(data, training=True) 

            # get the MSE loss
            comp_loss = self.compiled_loss(out, data, regularization_losses=self.losses)

            # get the jacobian for the contractive loss
            jacobian = tape.jacobian(mu, flat_inp)

            c_loss   = tf.nn.compute_average_loss(self.c_lambda*K.sum(K.square(jacobian), axis=(-1, -2)))

            loss = comp_loss + c_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients      = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.metrics.update_state(out, data)
        c_loss_metric.update_state(None, None, c_loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
'''
