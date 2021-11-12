from .globals import *

# A function to compute the value of latent space
def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch,dim))
    return mu + K.exp(sigma/2)*eps

class VariationalAE():
    def __init__(self, latent_dim=128, conv_filt=256, hidden=[], nconv=3, batch_norm=False, batch_norm2=False):
        self.latent_dim  = latent_dim
        self.conv_filt   = conv_filt
        self.hidden      = hidden 
        self.nconv       = nconv
        self.batch_norm  = batch_norm
        self.batch_norm2 = batch_norm2


    def create_model(self, sigma0=0., beta=1.e-3, dense_act='sigmoid'):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        latent_dim = self.latent_dim
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2
        self.dense_act = dense_act

        ''' ENCODER '''
        input_shape = (128, 128, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')
        
        enc_c = []
        enc_p = []
        enc_b = []

        # convolution part
        for i in range(self.nconv):
            if i==0:
                enc_c.append(Conv2D(conv_filt, (3,3), padding='same', activation='relu')(encoder_input))
            else:
                if batch_norm:
                    enc_c.append(Conv2D(conv_filt, (3,3), padding='same', activation='relu')(enc_b[-1]))
                else:
                    enc_c.append(Conv2D(conv_filt, (3,3), padding='same', activation='relu')(enc_p[-1]))
            enc_p.append(MaxPool2D(pool_size=(2,2), padding='same')(enc_c[-1]))
            if batch_norm:
                enc_b.append(BatchNormalization()(enc_p[-1]))

        input_conv = enc_p[-1]
        
        # sampling and bottleneck 
        enc_inps = []
        if batch_norm:
            enc_flat = Flatten(name='flat_inp')(enc_b[-1])
        elif batch_norm2:
            enc_flat1 = Flatten(name='flat_inp')(enc_p[-1])
            enc_flat  = BatchNormalization()(enc_flat1)
        else:
            enc_flat = Flatten(name='flat_inp')(enc_p[-1])

        enc_inps.append(enc_flat)

        for layeri in hidden:
            enc_inps.append(Dense(layeri, name='dense_%d'%layeri, activation=self.dense_act)(enc_inps[-1]))
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
        dec_inps = []
        decoder_input = Input(shape=(latent_dim,), name='dec_inp')
        dec_inps.append(decoder_input)
        if len(hidden) > 0:
            for layeri in hidden[::-1]:
                dec_inps.append(Dense(layeri, name='dense_dec_%d'%layeri, activation=self.dense_act)(dec_inps[-1]))
        dec2 = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(dec_inps[-1])
        if batch_norm2:
            dec2_1 = BatchNormalization()(dec2)
            dec3 = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(dec2_1)
        else:
            dec3 = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(dec2)

        
        dec_u = []
        dec_c = []
        dec_b = []

        for i in range(self.nconv-1):
            if i==0:
                dec_u.append(UpSampling2D((2,2))(dec3))
            else:
                if batch_norm:
                    dec_u.append(UpSampling2D((2,2))(dec_b[-1]))
                else:
                    dec_u.append(UpSampling2D((2,2))(dec_c[-1]))
            dec_c.append(Conv2DTranspose(filters=conv_filt, kernel_size=3,padding='same',activation='relu')(dec_u[-1]))
            if batch_norm:
                dec_b.append(BatchNormalization()(dec_c[-1]))


        '''
        dec_u1 = UpSampling2D((2,2))(dec3)
        dec_c1 = Conv2DTranspose(filters=conv_filt, kernel_size=3,padding='same',activation='relu')(dec_u1)
        dec_b1 = BatchNormalization()(dec_b1)
        
        dec_u2 = UpSampling2D((2,2))(dec_c1)
        dec_c2 = Conv2DTranspose(filters=conv_filt, kernel_size=3,padding='same',activation='relu')(dec_u2)
        dec_b2 = BatchNormalization()(dec_b2)
        
        dec_u3 = UpSampling2D((2,2))(dec_c2)
        dec_c3 = Conv2DTranspose(filters=conv_filt, kernel_size=3,padding='same',activation='relu')(dec_u3)
        dec_b3 = BatchNormalization()(dec_b3)
        '''
        
        if batch_norm:
            dec_u.append(UpSampling2D((2,2))(dec_b[-1]))
        else:
            dec_u.append(UpSampling2D((2,2))(dec_c[-1]))
        decoder_output = Conv2DTranspose(filters=3, kernel_size=5, padding='same', activation='relu')(dec_u[-1])
        
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

        self.output = self.decoder(self.encoder(encoder_input)[2])
        
        self.encoder.summary()

        self.decoder.summary()
        
        # Build the VAE
        self.ae = Model(encoder_input, self.output, name='VAE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        #self.ae.enc_conv_flat = Model(encoder_input, enc_inps[0], name='enc_conv_flat')
        #self.ae.enc_hidden    = Model(Input(tensor=enc_inps[0]), [mu, sigma, z], name='enc_hidden')

        # set the training parameters
        self.ae.sig0     = sigma0#*K.ones_like(sigma)
        self.ae.kl_beta  = beta
        
        self.ae.summary()
        
    def add_loss_funcs(self):
        recon_loss = K.mean(K.sum(K.square(self.input - self.output), axis=(1,2)), axis=(1))#*128*128

        mui, sigmai, z = self.encoder(self.input)
        mup, sigp, zp = self.encoder(self.output)

        sig0 = self.ae.sig0*K.ones_like(sigmai)

        # KL divergence loss
        # kl = 1 + sigmai + 4 (-K.square(mui-mup) - K.exp(sigmai))/(2*K.exp(4))
        kl = - 1 - sigmai + sig0 + (K.square(mui-mup) + K.exp(sigmai))/K.exp(sig0)
        kl = K.mean(kl, axis=-1)
        kl *= 0.5*self.ae.kl_beta

        r_loss     = tf.nn.compute_average_loss(recon_loss)
        kl_loss    = tf.nn.compute_average_loss(kl)    

        # sum of all three losses
        loss = r_loss + kl_loss# + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')

    def compile(self, learning_rate=0.0001, optimizer='Adam'):
        if optimizer=='Adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer=='Adagrad':
            opt = Adagrad(learning_rate=learning_rate)
        self.ae.compile(optimizer=opt)
        self.create_name()

    def train(self, data, epochs=300, batch_size=10):

        savesfolder = self.get_savefolder()
        print(f"Training {self.name}")

        self.nepochs = epochs

        if not os.path.exists(savesfolder):
            os.mkdir(savesfolder)

        self.history = self.ae.fit(data, epochs=epochs, validation_split=0.1, 
                          validation_freq=5, batch_size=batch_size, shuffle=True)

    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'vae_%dls_conv%d%s_%s'%(self.latent_dim, self.conv_filt, hidden_name, self.dense_act)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"
        
        self.name += "_sig%1.0f_beta_%1.0e"%(self.ae.sig0, self.ae.kl_beta)
        
        print(self.name)

    def save(self):
        savesfolder = self.get_savefolder()
        self.ae.encoder.save_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.save_weights(savesfolder+"decoderw.h5")
        self.ae.save_weights(savesfolder+"VAEw.h5")
    
    def load(self):
        savesfolder = self.get_savefolder()
        self.ae.encoder.load_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.load_weights(savesfolder+"decoderw.h5")
        #self.ae.load_weights(savesfolder+"VAEw.h5")
    def get_savefolder(self):
        self.savesfolder =  f'{MODEL_SAVE_FOLDER}models-{self.name}/'
        return self.savesfolder

class ContractiveAE(VariationalAE):
    def create_model(self):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        latent_dim = self.latent_dim
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2

        ''' ENCODER '''
        input_shape = (128, 128, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')
        
        enc_c = []
        enc_p = []
        enc_b = []

        # convolution part
        for i in range(self.nconv):
            if i==0:
                enc_c.append(Conv2D(conv_filt, (3,3), padding='same', activation='relu')(encoder_input))
            else:
                if batch_norm:
                    enc_c.append(Conv2D(conv_filt, (3,3), padding='same', activation='relu')(enc_b[-1]))
                else:
                    enc_c.append(Conv2D(conv_filt, (3,3), padding='same', activation='relu')(enc_p[-1]))
            enc_p.append(AveragePooling2D(pool_size=(2,2), padding='same')(enc_c[-1]))
            if batch_norm:
                enc_b.append(BatchNormalization()(enc_p[-1]))

        input_conv = enc_p[-1]
        
        # sampling and bottleneck 
        enc_inps = []
        if batch_norm:
            enc_flat = Flatten(name='flat_inp')(enc_b[-1])
        elif batch_norm2:
            enc_flat1 = Flatten(name='flat_inp')(enc_p[-1])
            enc_flat  = BatchNormalization()(enc_flat1)
        else:
            enc_flat = Flatten(name='flat_inp')(enc_p[-1])

        enc_inps.append(enc_flat)

        for layeri in hidden:
            enc_inps.append(Dense(layeri, name='dense_%d'%layeri, activation='sigmoid')(enc_inps[-1]))
        latent_space = Dense(latent_dim, name='latent')(enc_inps[-1])
        
        # Build the encoder
        self.encoder = Model(encoder_input, latent_space, name='encoder')

        self.z = latent_space
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=(latent_dim,), name='dec_inp')
        dec_inps.append(decoder_input)
        if len(hidden) > 0:
            for layeri in hidden[::-1]:
                dec_inps.append(Dense(layeri, name='dense_dec_%d'%layeri, activation='sigmoid')(dec_inps[-1]))
        dec2 = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(dec_inps[-1])
        if batch_norm2:
            dec2_1 = BatchNormalization()(dec2)
            dec3 = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(dec2_1)
        else:
            dec3 = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(dec2)

        
        dec_u = []
        dec_c = []
        dec_b = []

        for i in range(self.nconv-1):
            if i==0:
                dec_u.append(UpSampling2D((2,2))(dec3))
            else:
                if batch_norm:
                    dec_u.append(UpSampling2D((2,2))(dec_b[-1]))
                else:
                    dec_u.append(UpSampling2D((2,2))(dec_c[-1]))
            dec_c.append(Conv2DTranspose(filters=conv_filt, kernel_size=3,padding='same',activation='relu')(dec_u[-1]))
            if batch_norm:
                dec_b.append(BatchNormalization()(dec_c[-1]))

        
        if batch_norm:
            dec_u.append(UpSampling2D((2,2))(dec_b[-1]))
        else:
            dec_u.append(UpSampling2D((2,2))(dec_c[-1]))
        decoder_output = Conv2DTranspose(filters=3, kernel_size=5, padding='same', activation='relu')(dec_u[-1])
        
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

        self.output = self.decoder(self.encoder(encoder_input))
        
        self.encoder.summary()

        self.decoder.summary()
        
        # Build the VAE
        self.ae = Model(encoder_input, self.output, name='AE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        self.ae.summary()
        
    def add_loss_funcs(self, contr_loss=False, lam=1.e-3):
        self.ae.c_lambda = lam
        recon_loss = K.mean(K.sum(K.square(self.input - self.output), axis=(1,2)), axis=(1))#*128*128

        r_loss     = tf.nn.compute_average_loss(recon_loss)

        # sum of all three losses
        loss = r_loss

        if contr_loss:
            hidden = self.hidden
            for i, layeri in enumerate(hidden):
                # x1 = dense layer
                x1 = self.ae.get_layer('encoder').get_layer('dense_%d'%layeri).output
                # sigmoid activation 
                # calculate dPhi(W1*x)/d(W1*x) = dz1
                x1_1  = tf.expand_dims(x1, axis=-1)
                x1_2 = tf.expand_dims(x1, axis=1)
                dz1 = K.batch_dot(x1_1, (1-x1_2))
                
                # relu activation 
                '''
                # calculate dPhi(W1*x)/d(W1*x) = dz1
                x1_2 = tf.expand_dims((K.sign(x1) + 1)/2, axis=1)
                x1_1 = tf.expand_dims(K.ones_like(x1), axis=-1)
                dz1  = K.batch_dot(x1_1, x1_2)
                '''

                # tanh activation 
                '''
                # calculate dPhi(W1*x)/d(W1*x) = dz1
                x1_2 = tf.expand_dims(x1, axis=1)
                x1_1 = tf.expand_dims(x1, axis=-1)
                dz1  = 1 - K.batch_dot(x1_1, x1_2)
                '''

                # W1 = weights of first dense layer
                W1 = self.ae.get_layer('encoder').get_layer('dense_%d'%layeri).weights[0]
                W1 = tf.expand_dims(K.transpose(W1), axis=0)

                # apply chain rule to get the gradient incl. the previous layer
                if i!=0:
                    W1 = K.batch_dot(W1, dx1)

                # calculate dx1/dx = dx1
                dx1 = K.batch_dot(dz1, W1, axes=1)
            # print(tf.expand_dims(W1m, axis=0).shape, denc2.shape)

            # h = output layer
            h = self.ae.get_layer('encoder').get_layer('latent').output
            # W2 = output layer weights
            W2 = self.ae.get_layer('encoder').get_layer('latent').weights[0]
            W2 = tf.expand_dims(K.transpose(W2), axis=0)

            # calculate W = W2*dx1
            W = K.batch_dot(W2, dx1)

            # contractive loss = norm( dPhi(W2*x1)/d(W2*x1) * W )
            contractive_loss = self.ae.c_lambda*K.mean(K.ones_like(h)*K.sum(W**2, axis=2), axis=1)

            c_loss     = tf.nn.compute_average_loss(contractive_loss)

            loss += c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')

        self.contr_loss = contr_loss

    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'cae_%dls_conv%d%s'%(self.latent_dim, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"
        
        self.name += "_lam_%1.0e"%(self.ae.c_lambda)
        
        if self.contr_loss:
            self.name += "_contr"

        print(self.name)

class ContractiveVAE(VariationalAE):
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'cvae_%dls_conv%d%s'%(self.latent_dim, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"
        
        self.name += "_lam_%1.0e_sig%1.0f_beta_%1.0e_contr"%(self.ae.c_lambda, self.ae.sig0, self.ae.kl_beta)

    def add_loss_funcs(self, lam=1.e-3):
        self.ae.c_lambda = lam

        recon_loss = K.mean(K.sum(K.square(self.input - self.output), axis=(1,2)), axis=(1))#*128*128

        mui, sigmai, z = self.encoder(self.input)
        mup, sigp, zp = self.encoder(self.output)

        sig0 = self.ae.sig0*K.ones_like(sigmai)

        # KL divergence loss
        # kl = 1 + sigmai + 4 (-K.square(mui-mup) - K.exp(sigmai))/(2*K.exp(4))
        kl = - 1 - sigmai + sig0 + (K.square(mui-mup) + K.exp(sigmai))/K.exp(sig0)
        kl = K.mean(kl, axis=-1)
        kl *= 0.5*self.ae.kl_beta

        r_loss     = tf.nn.compute_average_loss(recon_loss)
        kl_loss    = tf.nn.compute_average_loss(kl)    

        # sum of all three losses
        loss = r_loss + kl_loss# + c_loss

        hidden = self.hidden
        for i, layeri in enumerate(hidden):
            # x1 = dense layer
            x1 = self.ae.get_layer('encoder').get_layer('dense_%d'%layeri).output
            # sigmoid activation 
            # calculate dPhi(W1*x)/d(W1*x) = dz1
            x1_1  = tf.expand_dims(x1, axis=-1)
            x1_2 = tf.expand_dims(x1, axis=1)
            dz1 = K.batch_dot(x1_1, (1-x1_2))
            
            # relu activation 
            '''
            # calculate dPhi(W1*x)/d(W1*x) = dz1
            x1_2 = tf.expand_dims((K.sign(x1) + 1)/2, axis=1)
            x1_1 = tf.expand_dims(K.ones_like(x1), axis=-1)
            dz1  = K.batch_dot(x1_1, x1_2)
            '''

            # tanh activation 
            '''
            # calculate dPhi(W1*x)/d(W1*x) = dz1
            x1_2 = tf.expand_dims(x1, axis=1)
            x1_1 = tf.expand_dims(x1, axis=-1)
            dz1  = 1 - K.batch_dot(x1_1, x1_2)
            '''

            # W1 = weights of first dense layer
            W1 = self.ae.get_layer('encoder').get_layer('dense_%d'%layeri).weights[0]
            W1 = tf.expand_dims(K.transpose(W1), axis=0)

            # apply chain rule to get the gradient incl. the previous layer
            if i!=0:
                W1 = K.batch_dot(W1, dx1)

            # calculate dx1/dx = dx1
            dx1 = K.batch_dot(dz1, W1, axes=1)
        # print(tf.expand_dims(W1m, axis=0).shape, denc2.shape)

        # h = output layer
        h = self.ae.get_layer('encoder').get_layer('mu').output
        # W2 = output layer weights
        W2 = self.ae.get_layer('encoder').get_layer('mu').weights[0]
        W2 = tf.expand_dims(K.transpose(W2), axis=0)

        # calculate W = W2*dx1
        W = K.batch_dot(W2, dx1)

        # contractive loss = norm( dPhi(W2*x1)/d(W2*x1) * W )
        contractive_loss = self.ae.c_lambda*K.mean(K.ones_like(h)*K.sum(W**2, axis=2), axis=1)

        c_loss     = tf.nn.compute_average_loss(contractive_loss)

        loss += c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')
        self.ae.add_metric(c_loss, aggregation='mean', name='contr')
