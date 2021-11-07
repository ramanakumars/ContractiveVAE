from .globals import *

# A function to compute the value of latent space
def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch,dim))
    return mu + K.exp(sigma/2)*eps


class ContractiveVAE():

    def __init__(self, latent_dim=128, conv_filt=256, hidden=[]):
        self.latent_dim = latent_dim
        self.conv_filt  = self.conv_fit
        self.hidden     = hidden 

        hidden_name = ''
        for layeri in hidden:
            hidden_name.append('_%d'%layeri)

        self.name = 'cvae_%dls_conv%d%s'%(latent_dim, conv_filt, hidden_name)

        print(self.name)

    def create_model(self):
        hidden     = self.hidden
        conv_filt  = self.conv_fit
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
        last_inp = Flatten()(enc_b3)

        for layeri in hidden:
            last_inp = Dense(layeri, name='dense_%d'%layeri, activation='sigmoid')(last_inp)
        mu = Dense(latent_dim, name='mu', activation=None)(last_inp)
        sigma = Dense(latent_dim, name='sig',  activation=None)(last_inp)

        latent_space = Lambda(compute_latent, output_shape=(latent_dim,), name='latent')([mu, sigma])
        
        # Build the encoder
        encoder = Model(encoder_input, [mu, sigma, latent_space], name='encoder')

        self.mu = mu; self.sigma = sigma; self.z = latent_space
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        last_inp = Input(shape=(latent_dim,), name='dec_inp')
        if len(hidden) > 0:
            for layeri in hidden[::-1]:
                dec1 = Dense(layeri, name='dense_dec_%d'%layeri, activation='sigmoid')(last_inp)
        dec2 = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(last_inp)
        dec3 = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(dec2)

        #dec_bn = BatchNormalization()(dec3)

        dec_u1 = UpSampling2D((2,2))(dec_bn)
        dec_c1 = Conv2DTranspose(filters=conv_filt, kernel_size=3,padding='same',activation='relu')(dec_u1)
        dec_b1 = BatchNormalization()(dec_c1)
        
        dec_u2 = UpSampling2D((2,2))(dec_c1)
        dec_c2 = Conv2DTranspose(filters=conv_filt, kernel_size=3,padding='same',activation='relu')(dec_u2)
        dec_b2 = BatchNormalization()(dec_c2)

        dec_u3 = UpSampling2D((2,2))(dec_c2)  
        decoder_output = Conv2DTranspose(filters=3, kernel_size=5, padding='same', activation='relu')(dec_u3)
        
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

        self.output = decoder(encoder(encoder_input)[2])
        
        self.encoder.summary()

        self.decoder.summary()
        
        # Build the VAE
        self.vae = Model(encoder_input, output, name='VAE')
        
        vae.summary()
        
    def add_loss_funcs(self, sigma0=0., lam=1.e-3, beta=1.e-3):
        lam  = 1.e-1
        beta = 0.1

        self.name += "lam_%1e_sig%1f_beta_%1e"

        recon_loss = K.mean(K.square(inp - out), axis=(1,2,3))*128*128

        mui, sigmai, z = self.encoder(inp)
        mup, sigp, zp = self.encoder(out)

        sig0 = sigma0*K.ones_like(sigmai)

        # KL divergence loss
        # kl = 1 + sigmai + 4 (-K.square(mui-mup) - K.exp(sigmai))/(2*K.exp(4))
        kl = 1 - sigmai + sig0 + (K.square(mui-mup) + K.exp(sigmai))/K.exp(sig0)
        kl = K.sum(kl, axis=-1)
        kl *= 0.5*beta

        for i, layeri in enumerate(hidden):
            # x1 = first dense layer
            x1 = vae.get_layer('encoder').get_layer('dense_%d'%layeri).output
            # W1 = weights of first dense layer
            W1 = vae.get_layer('encoder').get_layer('dense_%d'%layeri).weights[0]
            W1 = K.transpose(W1)


            # calculate dPhi(W1*x)/d(W1*x) = dz1
            x1_1  = tf.expand_dims(x1, axis=-1)
            x1_2 = tf.expand_dims(x1, axis=1)
            # print(enc2.shape, enc_expanded.shape, enc_expanded.shape, W1.shape)
            dz1 = K.batch_dot(x1_1, (1-x1_2))

            if i!=0:
                W1 = K.batch_dot(tf.expand_dims(W1, axis=0), dx1)

            # calculate dx1/dx = dx1
            dx1 = K.dot(dz2, W1)
        # print(tf.expand_dims(W1m, axis=0).shape, denc2.shape)

        # h = output layer
        h = self.vae.get_layer('encoder').get_layer('mu').output
        # W2 = output layer weights
        W2 = self.vae.get_layer('encoder').get_layer('mu').weights[0]
        W2 = K.transpose(W2)

        # calculate W = W2*dx1
        W = K.batch_dot(tf.expand_dims(W2, axis=0), dx1)

        # contractive loss = norm( dPhi(W2*x1)/d(W2*x1) * W )
        contractive_loss = lam*K.sum(K.ones_like(h)*K.sum(W**2, axis=2), axis=1)
        c_loss     = tf.nn.compute_average_loss(contractive_loss)

        r_loss = tf.nn.compute_average_loss(recon_loss)
        kl_loss = tf.nn.compute_average_loss(kl)    

        # sum of all three losses
        loss = r_loss + c_loss + kl_loss

        self.vae.add_loss(loss)
        self.vae.add_metric(total_loss, aggregation='mean', name='mse')
        self.vae.add_metric(c_loss, aggregation='mean', name='contractive')
        self.vae.add_metric(kl_loss, aggregation='mean', name='kl')

    def compile(self, learning_rate=0.0001):
        opt = Adam(learning_rate=learning_rate)
        self.vae.compile(optimizer=opt)


    def train(self, data, epochs=300, batch_size=10):
        savesfolder = 'models-%s/'%self.name

        if not os.path.exists(savesfolder):
            os.mkdir(savesfolder)

        self.history = vae.fit(data, epochs=epochs, validation_split=0.1, 
                          validation_freq=5, batch_size=batch_size, shuffle=True)

    def save(self):
        savesfolder = 'models-%s/'%self.name
	self.encoder.save_weights(savesfolder+"encoderw.h5")
	self.decoder.save_weights(savesfolder+"decoderw.h5")
	self.vae.save_weights(savesfolder+"VAEw.h5")
    
    def load(self):
        savesfolder = 'models-%s/'%self.name
	self.encoder.load_weights(savesfolder+"encoderw.h5")
	self.decoder.load_weights(savesfolder+"decoderw.h5")
	self.vae.load_weights(savesfolder+"VAEw.h5")
