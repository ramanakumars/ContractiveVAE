from .globals import *
from .model import VariationalAE

class ColorAE(VariationalAE):
    def create_model(self, conv_act='tanh', pool=False):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2
        self.conv_act  = conv_act
        self.pool   = pool

        ''' ENCODER '''
        input_shape = (128, 128, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)

        # convolution part
        enc_layers = encoder_layers(reshape_layer1, conv_filt, conv_act, hidden)
        input_conv = enc_layers[-1]
        latent_space = Flatten(name='latent')(enc_layers[-1])
        
        # Build the encoder
        self.encoder = Model(encoder_input, latent_space, name='encoder')

        self.encoder.summary()

        self.z = latent_space
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(latent_space)[1:], name='dec_inp')
        dec_inps.append(decoder_input)
        
        if batch_norm2:
            dec2_1 = BatchNormalization()(decoder_input)
            dec3 = Reshape(conv_shape[1:])(dec2_1)
        else:
            dec3 = Reshape(conv_shape[1:])(decoder_input)

        
        dec_layers = decoder_layers(dec3, conv_filt, conv_act, hidden)
        
        decoder_output = Reshape(target_shape=input_shape, name='3d_reshape')(dec_layers[-1])
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input))
        
        # Build the VAE
        self.ae = Model(encoder_input, self.output, name='AE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        self.ae.summary()
        
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'colorae_%dconv%d%s'%(self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)
    
    def add_loss_funcs(self):
        recon_loss = K.mean(K.sum(K.square(self.input - self.output), axis=(1,2)), axis=(1))#*128*128

        z = self.encoder(self.input)
        zp = self.encoder(self.output)

        z_mse_loss = K.sum(K.square(z - zp), axis=1)

        r_loss   = tf.nn.compute_average_loss(recon_loss)
        z_loss   = tf.nn.compute_average_loss(z_mse_loss)

        # sum of all three losses
        loss = r_loss + z_loss# + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(z_loss, aggregation='mean', name='z')

class ColorVAE(VariationalAE):
    def create_model(self, sigma0=0., beta=1.e-3, conv_act='tanh', pool=False):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        latent_dim = self.latent_dim
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2
        self.conv_act  = conv_act
        self.pool   = pool

        ''' ENCODER '''
        input_shape = (128, 128, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        '''
        for i in range(self.nconv):
            conv_layer = Conv3D(conv_filt, (2**(3-i), 2**(3-i),1), padding='valid', strides=(2,2,1), 
                                activation=conv_act, name=f'enc_conv_{i}')
            if i==0:
                enc_layers.append(conv_layer(reshape_layer1))
            else:
                if batch_norm:
                    enc_layers.append(conv_layer(enc_layers[-1]))
                else:
                    enc_layers.append(conv_layer(enc_layers[-1]))
            #if (i%2==1):
            #    enc_layers.append(MaxPool3D(pool_size=(2,2,1), padding='same', name=f'enc_pool_{i}')(enc_layers[-1]))
            if batch_norm:
                enc_layers.append(BatchNormalization(name=f'enc_batch_norm_{i}')(enc_layers[-1]))

        if pool:
            enc_layers.append(AveragePooling3D(pool_size=(2,2,1), padding='valid', name=f'enc_pool')(enc_layers[-1]))

        enc_c1 = Conv3D(4, (4, 4, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_0')(reshape_layer1)
        enc_b1 = BatchNormalization(name='enc_batch_norm_0')(enc_c1)

        enc_c2 = Conv3D(32, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_1')(enc_b1)
        enc_b2 = BatchNormalization(name='enc_batch_norm_1')(enc_c2)

        enc_c3 = Conv3D(conv_filt, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_2')(enc_b2)
        enc_b3 = BatchNormalization(name='enc_batch_norm_2')(enc_c3)
        

        enc_c4 = Conv3D(conv_filt, (2, 2, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_3')(enc_b3)
        enc_b4 = BatchNormalization(name='enc_batch_norm_3')(enc_c4)

        enc_c5 = Conv3D(conv_filt, (2, 2, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_4')(enc_b4)
        enc_b5 = BatchNormalization(name='enc_batch_norm_4')(enc_c5)
        
        #enc_p  = AveragePooling3D(pool_size=(2,2,1), name='enc_pool')(enc_b4)

        enc_layers = [enc_b5]

        for i in range(self.nconv):
            if i < self.nconv-1:
                acti = conv_act
            else:
                acti = None
            enc_layers.append(Conv3D(hidden[i], (1,1,1), padding='valid', strides=(1,1,1), 
                                     activation=acti, name=f'enc_conv_dense_{i}')(enc_layers[-1]))
            enc_layers.append(BatchNormalization(name=f'enc_bn_dense_{i}')(enc_layers[-1]))

        input_conv = enc_layers[-1]
        '''

        enc_layers = encoder_layers(reshape_layer1, conv_filt, conv_act, hidden)

        mu = Flatten(name='mu')(enc_layers[-1])

        latent_space = Lambda(compute_latent2, output_shape=K.int_shape(mu)[1:], name='latent')(mu)
        
        # Build the encoder
        self.encoder = Model(encoder_input, [mu, latent_space], name='encoder')

        self.encoder.summary()

        self.mu = mu; self.z = latent_space
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(enc_layers[-1])
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(latent_space)[1:], name='dec_inp')
        dec_inps.append(decoder_input)
        
        if batch_norm2:
            dec2_1 = BatchNormalization()(decoder_input)
            dec3 = Reshape(conv_shape[1:])(dec2_1)
        else:
            dec3 = Reshape(conv_shape[1:])(decoder_input)


        '''
        dec_layers = [dec3]

        for i in range(self.nconv):
            dec_layers.append(Conv3D(hidden[::-1][i], (1,1,1), padding='valid', strides=(1,1,1), 
                                     activation=conv_act, name=f'dec_conv_dense_{i}')(dec_layers[-1]))
            dec_layers.append(BatchNormalization(name=f'dec_bn_dense_{i}')(dec_layers[-1]))

        dec_c1 = Conv3DTranspose(conv_filt, (2, 2, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='dec_conv_1')(dec_layers[-1])
        dec_b1 = BatchNormalization(name='dec_batch_norm_0')(dec_c1)

        dec_c2 = Conv3DTranspose(conv_filt, (2, 2, 1), padding='valid', strides=(1,1,1), 
                            activation=conv_act, name='dec_conv_2')(dec_b1)
        dec_b2 = BatchNormalization(name='dec_batch_norm_1')(dec_c2)

        dec_c3 = Conv3DTranspose(conv_filt, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='dec_conv_3')(dec_b2)
        dec_b3 = BatchNormalization(name='dec_batch_norm_2')(dec_c3)

        dec_c4 = Conv3DTranspose(32, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='dec_conv_4')(dec_b3)
        dec_b4 = BatchNormalization(name='dec_batch_norm_3')(dec_c4)

        dec_c5 = Conv3DTranspose(4, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='dec_conv_5')(dec_b4)
        dec_b5 = BatchNormalization(name='dec_batch_norm_4')(dec_c5)

        dec_c6 = Conv3DTranspose(1, (4, 4, 1), padding='valid', strides=(2,2,1), 
                            activation='relu', name='dec_conv_6')(dec_b5)
        '''

        dec_layers = decoder_layers(dec3, conv_filt, conv_act, hidden)
        #dec_b6 = BatchNormalization(name='dec_batch_norm_5')(dec_c6)

        #dec_c7 = Conv3D(1, (2, 2, 1), padding='valid', strides=(2,2,1), 
        #                    activation=conv_act, name='dec_conv_7')(dec_b6)

        decoder_output = Reshape(target_shape=input_shape, name='3d_reshape')(dec_layers[-1])
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input)[1])
        
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
        
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'colorvae2_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)
    
    def add_loss_funcs(self):
        recon_loss = K.mean(K.sum(K.square(self.input - self.output), axis=(1,2)), axis=(1))#*128*128

        mui, z  = self.encoder(self.input)
        mup, zp = self.encoder(self.output)

        sig0 = self.ae.sig0*K.ones_like(mui)
        
        kl = - 1 + (K.square(mui-mup) + K.exp(sig0))/K.exp(sig0)
        kl = K.mean(kl, axis=-1)
        kl *= 0.5*self.ae.kl_beta


        r_loss   = tf.nn.compute_average_loss(recon_loss)
        kl_loss  = tf.nn.compute_average_loss(kl)    

        # sum of all three losses
        loss = r_loss + kl_loss# + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')

class ConvVAE(VariationalAE):
    def create_model(self, sigma0=0., beta=1.e-3, conv_act='tanh', pool=False):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        latent_dim = self.latent_dim
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2
        self.conv_act  = conv_act
        self.pool   = pool

        ''' ENCODER '''
        input_shape = (128, 128, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        #reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        enc_layers = encoder_layers2D(self.input, conv_filt, conv_act, hidden)
        input_conv = enc_layers[-1]
        
        mu = Flatten(name='mu')(enc_layers[-1])
        sigma = Dense(K.int_shape(mu)[1], name='sig', activation=None)(Flatten()(enc_layers[-1]))

        latent_space = Lambda(compute_latent, output_shape=K.int_shape(mu)[1:], name='latent')([mu, sigma])
        
        # Build the encoder
        self.encoder = Model(encoder_input, [mu, sigma, latent_space], name='encoder')

        self.encoder.summary()

        self.mu = mu; self.sigma = sigma; self.z = latent_space
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(latent_space)[1:], name='dec_inp')
        dec_inps.append(decoder_input)
        
        if batch_norm2:
            dec2_1 = BatchNormalization()(decoder_input)
            dec3 = Reshape(conv_shape[1:])(dec2_1)
        else:
            dec3 = Reshape(conv_shape[1:])(decoder_input)

        #upsamp_layer = UpSampling3D((2,2,1), name='dec_upsamp')(dec3)
        
        dec_layers = decoder_layers2D(dec3, conv_filt, conv_act, hidden)
        decoder_output = dec_layers[-1]
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input)[2])
        
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
        
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'convvae_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)

class ConvAE(VariationalAE):
    def create_model(self, conv_act='tanh', pool=False):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        latent_dim = self.latent_dim
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2
        self.conv_act  = conv_act
        self.pool   = pool

        ''' ENCODER '''
        input_shape = (128, 128, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        #reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        enc_layers = encoder_layers2D(self.input, conv_filt, conv_act, hidden)
        input_conv = enc_layers[-1]
        
        latent_space = Flatten(name='latent')(enc_layers[-1])
        
        # Build the encoder
        self.encoder = Model(encoder_input, latent_space, name='encoder')

        self.encoder.summary()

        self.mu = mu; self.sigma = sigma; self.z = latent_space
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(latent_space)[1:], name='dec_inp')
        dec_inps.append(decoder_input)
        
        if batch_norm2:
            dec2_1 = BatchNormalization()(decoder_input)
            dec3 = Reshape(conv_shape[1:])(dec2_1)
        else:
            dec3 = Reshape(conv_shape[1:])(decoder_input)

        #upsamp_layer = UpSampling3D((2,2,1), name='dec_upsamp')(dec3)
        
        dec_layers = decoder_layers2D(dec3, conv_filt, conv_act, hidden)
        decoder_output = dec_layers[-1]
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input)[2])
        
        # Build the VAE
        self.ae = Model(encoder_input, self.output, name='VAE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder
        
        # set the training parameters
        self.ae.summary()
        
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'convae_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)
    
    def add_loss_funcs(self):
        recon_loss = K.mean(K.sum(K.square(self.input - self.output), axis=(1,2)), axis=(1))#*128*128

        z = self.encoder(self.input)
        zp = self.encoder(self.output)

        z_mse_loss = K.sum(K.square(z - zp), axis=1)

        r_loss   = tf.nn.compute_average_loss(recon_loss)
        z_loss   = tf.nn.compute_average_loss(z_mse_loss)

        # sum of all three losses
        loss = r_loss + z_loss# + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(z_loss, aggregation='mean', name='z')
