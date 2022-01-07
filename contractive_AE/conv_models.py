from .globals import *
from .model import VariationalAE

class GammaLayer(Layer):
    def __init__(self, latent_dim, n_centroid, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim#/npixels)
        self.n_centroid = n_centroid
        
    def build(self, input_shape):
        theta_init  = np.ones((1, 1, self.n_centroid))/self.n_centroid
        u_init      = np.zeros((1, self.latent_dim, self.n_centroid))
        lambda_init = np.ones((1, self.latent_dim, self.n_centroid))
        
        # define the variables used by this layer
        self.theta_p  = tf.Variable(theta_init, trainable=True, shape=(1, 1, self.n_centroid,), name="pi", dtype=tf.float32)
        self.u_p      = tf.Variable(u_init, trainable=True, shape=(1, self.latent_dim, self.n_centroid), name="u", dtype=tf.float32)
        self.lambda_p = tf.Variable(lambda_init, trainable=True, shape=(1, self.latent_dim, self.n_centroid),name="lambda", dtype=tf.float32)

        super().build(input_shape)

    def get_tensors(self, batch_size):
        u_tensor3 = self.u_p*K.ones((batch_size, self.npixels, self.latent_dim, n_centroid))
        lambda_tensor3 = tf.repeat(self.lambda_p, batch_size, axis=0)
        theta_tensor3 = self.theta_p*K.ones((batch_size, self.npixels, self.latent_dim,n_centroid))

        return theta_tensor3, u_tensor3, lambda_tensor3

    def call(self, x, training=None):
        n_centroid = self.n_centroid; latent_dim = self.latent_dim
        mu, sig, z = x

        batch_size = tf.shape(z)[0]

        # reshape the latent values
        Z = tf.transpose(K.repeat(z, n_centroid),perm=[0,2,3,1])
        z_mean_t = tf.transpose(K.repeat(mu,n_centroid),perm=[0,2,3,1])
        z_log_var_t = tf.transpose(K.repeat(sig,n_centroid),[0,2,3,1])

        # build the tensors for calculating gamma
        #theta_tensor3 = theta_p*tf.ones((batch_size, self.latent_dim, n_centroid))
        theta_tensor3, u_tensor3, lambda_tensor3 = self.get_tensors(batch_size)
        #print(self.u_tensor3.get_shape())
        #print(self.lambda_tensor3.get_shape())
        #print(self.theta_tensor3.get_shape())
        #p_c_z = K.exp(K.sum((K.log(self.theta_p[None,:])-0.5*K.log(2*np.pi*self.lambda_p[:,:])-\
        #                     K.square(Z-self.u_p[:,:])/(2*self.lambda_p[:,:])),axis=1))+1e-10
        a = K.log(theta_tensor3)
        b = 0.5*K.log(2*np.pi*lambda_tensor3)
        c = K.square(Z-u_tensor3)/(2*lambda_tensor3)

        print(a.get_shape(), b.get_shape(), c.get_shape())
        p_c_z = K.exp(K.sum(a - b - c ,axis=1) )+1e-10

        self.gamma = p_c_z/K.sum(p_c_z,axis=-1,keepdims=True)
        self.gamma_t = K.repeat(self.gamma, self.latent_dim)

        #print(self.gamma.get_shape())
        #print(self.gamma_t.get_shape())

        return self.gamma#self.gamma_t

class GammaLayerConv(Layer):
    def __init__(self, latent_dim, n_centroid, npixels, **kwargs):
        super().__init__(**kwargs)
        self.npixels    = npixels
        self.latent_dim = int(latent_dim/npixels)
        self.n_centroid = n_centroid
        
    def build(self, input_shape):
        theta_init  = np.ones((1, self.npixels, 1, self.n_centroid))/self.n_centroid
        u_init      = np.zeros((1, self.npixels, self.latent_dim, self.n_centroid))
        lambda_init = np.random.random((1, self.npixels, self.latent_dim, self.n_centroid))
        
        # define the variables used by this layer
        self.theta_p  = tf.Variable(theta_init, trainable=True, shape=(1, self.npixels, 1, self.n_centroid,), name="pi", dtype=tf.float32)
        self.u_p      = tf.Variable(u_init, trainable=True, shape=(1, self.npixels, self.latent_dim, self.n_centroid), name="u", dtype=tf.float32)
        self.lambda_p = tf.Variable(lambda_init, trainable=True, shape=(1,self.npixels, self.latent_dim, self.n_centroid),name="lambda", dtype=tf.float32)

        super().build(input_shape)

    def get_tensors(self, batch_size):
        u_tensor3 = self.u_p*K.ones((batch_size, self.npixels, self.latent_dim, n_centroid))
        lambda_tensor3 = tf.repeat(self.lambda_p, batch_size, axis=0)
        theta_tensor3 =  self.theta_p*K.ones((batch_size, self.npixels, self.latent_dim,n_centroid))
        '''
                        tf.repeat(\
                                tf.repeat(\
                                    tf.repeat(self.theta_p, self.latent_dim, axis=2), \
                                self.npixels, axis=1), \
                            batch_size, axis=0)
        '''

        return theta_tensor3, u_tensor3, lambda_tensor3

    def get_z_vals(self, x, only_z=False):
        n_centroid = self.n_centroid; latent_dim = self.latent_dim; npixels = self.npixels
        mu, sig, z = x
        

        # reshape the latent values
        Z = tf.transpose(K.repeat(z, n_centroid),perm=[0,2,1])
        Z = Reshape((self.npixels, self.latent_dim, n_centroid))(Z)

        if not only_z:
            z_mean_t = tf.transpose(K.repeat(mu,n_centroid),perm=[0,2,1])
            z_log_var_t = tf.transpose(K.repeat(sig,n_centroid),[0,2,1])
            
            Zmu  = Reshape((self.npixels, self.latent_dim, n_centroid))(z_mean_t)
            Zsig = Reshape((self.npixels, self.latent_dim, n_centroid))(z_log_var_t)

            return Z, Zmu, Zsig
        else:
            return Z

    def call(self, x, training=None):
        n_centroid = self.n_centroid; latent_dim = self.latent_dim
        mu, sig, z = x

        Z = self.get_z_vals(x, only_z=True)
        
        batch_size = tf.shape(z)[0]

        # build the tensors for calculating gamma
        #theta_tensor3 = theta_p*tf.ones((batch_size, self.latent_dim, n_centroid))
        theta_tensor3, u_tensor3, lambda_tensor3 = self.get_tensors(batch_size)
        #print(self.u_tensor3.get_shape())
        #print(self.lambda_tensor3.get_shape())
        #print(self.theta_tensor3.get_shape())
        #p_c_z = K.exp(K.sum((K.log(self.theta_p[None,:])-0.5*K.log(2*np.pi*self.lambda_p[:,:])-\
        #                     K.square(Z-self.u_p[:,:])/(2*self.lambda_p[:,:])),axis=1))+1e-10
        a = K.log(theta_tensor3)
        b = 0.5*K.log(2*np.pi*lambda_tensor3)
        c = K.square(Z-u_tensor3)/(2*lambda_tensor3)

        p_c_z = K.exp(K.sum(a - b - c ,axis=(2)) )+1e-10

        self.gamma = p_c_z/K.sum(p_c_z,axis=-1,keepdims=True)
        #gamma_t = tf.repeat(tf.repeat(self.gamma, self.latent_dim, axis=0), self.npixels, axis=0)
        #gamma_t = tf.repeat(
        #        tf.expand_dims(
        #            tf.repeat(tf.expand_dims(self.gamma, axis=1), self.latent_dim, axis=1),
        #        axis=1),
        #    self.npixels, axis=1)
        #gamma_t = tf.repeat(tf.expand_dims(self.gamma, axis=2), self.latent_dim, axis=2)

        #print(self.gamma.get_shape())
        #print(self.gamma_t.get_shape())

        return self.gamma#self.gamma_t

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
        #sigma = Dense(K.int_shape(mu)[1], name='sig', activation=None)(Flatten()(enc_layers[-1]))
        sigma = tf.ones_like(mu, name='sigma')*sigma0

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

        #upsamp_layer = UpSampling3D((2,2,1), name='dec_upsamp')(dec3)
        
        dec_layers = decoder_layers2D(dec3, conv_filt, conv_act, hidden)
        decoder_output = dec_layers[-1]
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input))
        
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
        recon_loss = K.mean(K.sum(K.abs(self.input - self.output), axis=(1,2)), axis=(1))#*128*128

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

class ConvVAE_DEC(VariationalAE):
    def create_model(self, sigma0=0., beta=1.e-3, conv_act='tanh', pool=False):
        global theta_p, u_p, lambda_p, n_centroid, latent_dim
        n_centroid = self.n_centroid
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

        #reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        enc_layers = encoder_layers2D(self.input, conv_filt, conv_act, hidden)
        input_conv = enc_layers[-1]
        
        mu = Flatten(name='mu')(enc_layers[-1])
        self.latent_dim = latent_dim = K.int_shape(mu)[1]
        self.npixels = K.int_shape(input_conv)[1]*K.int_shape(input_conv)[2]

        self.reduced_latent_dim = tf.cast(self.latent_dim/self.npixels, dtype=tf.int32)

        #sigma = Dense(K.int_shape(mu)[1], name='sig', activation=None)(Flatten()(enc_layers[-1]))
        sigma = K.ones_like(mu, name='sigma')*sigma0

        latent_space = Lambda(compute_latent, output_shape=K.int_shape(mu)[1:], name='latent')([mu, sigma])
        
        # Build the encoder
        self.encoder = Model(encoder_input, [mu, sigma, latent_space], name='encoder')

        self.encoder.summary()

        self.mu = mu; self.sigma = sigma; self.z = latent_space

        self.gamma = GammaLayerConv(latent_dim, n_centroid, self.npixels, name='gamma')([mu, sigma, latent_space])
        
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
        self.ae = Model(encoder_input, [self.output, self.gamma], name='VAE')


        self.cluster = Model(encoder_input, self.gamma, name='DC')

        print(self.cluster.summary())

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        #self.ae.enc_conv_flat = Model(encoder_input, enc_inps[0], name='enc_conv_flat')
        #self.ae.enc_hidden    = Model(Input(tensor=enc_inps[0]), [mu, sigma, z], name='enc_hidden')

        # set the training parameters
        self.ae.sig0     = sigma0#*K.ones_like(sigma)
        self.ae.kl_beta  = beta

        
        self.ae.summary()
        
    def add_loss_funcs(self):
        global theta_p, u_p, lambda_p

        recon_loss = K.mean(K.sum(K.square(self.input - self.output), axis=(1,2)), axis=(1))#*128*128

        mui,sigi, z  = self.encoder(self.input)
        mup, sigp, zp = self.encoder(self.output)

        batch_size = tf.shape(z)[0]

        gamma_layer = self.ae.get_layer('gamma')
        Z, ZMean, ZLogVar = gamma_layer.get_z_vals([mui, sigi, z])

        theta_tensor3, u_tensor3, lambda_tensor3 = gamma_layer.get_tensors(batch_size)

        gamma = self.cluster(self.input)
        #gamma_t = K.repeat(K.repeat(gamma, self.reduced_latent_dim), self.npixels)
        #gamma_t = K.repeat_elements((gamma, self.latent_dim, axis=0), self.npixels, axis=0)
        gamma_t = tf.repeat(tf.expand_dims(self.gamma, axis=2), tf.shape(theta_tensor3)[2], axis=2)


        sig0 = self.ae.sig0*K.ones_like(mui)
        
        kl = - 1 + (K.square(mui-mup) + K.exp(sig0))/K.exp(sig0)
        kl = K.mean(kl, axis=-1)
        kl *= 0.5*self.ae.kl_beta

        r_loss   = tf.nn.compute_average_loss(recon_loss)
        kl_loss  = tf.nn.compute_average_loss(kl)    

        print(f"gamma_t: {gamma_t.get_shape()}, gamma: {gamma.get_shape()}  tensor: {theta_tensor3.get_shape()}")

        a = K.sum(K.mean(0.5*gamma_t*(tf.cast(self.reduced_latent_dim, tf.float32)*K.log(np.pi*2)+K.log(lambda_tensor3)+
                        K.exp(ZLogVar)/lambda_tensor3+
                        K.square(ZMean-u_tensor3)/lambda_tensor3),axis=(1)), axis=(1,2))
        b =  0.5*K.sum(sigi+1,axis=-1)
        c = K.sum(K.log(K.mean(theta_tensor3, axis=(2))*gamma),axis=(1,2))
        d = K.sum(K.log(gamma)*gamma,axis=(1,2))

        #print(gamma_t.get_shape(), lambda_tensor3.get_shape(), ZLogVar.get_shape(), lambda_tensor3.get_shape(),
        #      ZMean.get_shape(), u_tensor3.get_shape())
        print(sigi.get_shape())
        clust_loss = tf.nn.compute_average_loss(a-c+d)
        # sum of all three losses
        loss = r_loss + kl_loss + clust_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')
        self.ae.add_metric(clust_loss, aggregation='mean', name='clust')
        self.ae.add_metric(K.min(lambda_tensor3), aggregation='mean', name='lambda_min')
        self.ae.add_metric(K.max(lambda_tensor3), aggregation='mean', name='lambda_max')
        self.ae.add_metric(K.min(theta_tensor3), aggregation='mean', name='theta_min')
        self.ae.add_metric(K.max(theta_tensor3), aggregation='mean', name='theta_max')
        self.ae.add_metric(K.min(gamma), aggregation='mean', name='gamma_min')
        self.ae.add_metric(K.max(gamma), aggregation='mean', name='gamma_max')
            
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'convvae_cluster_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)

class VAE_DEC(VariationalAE):
    def create_model(self, sigma0=0., beta=1.e-3, conv_act='tanh', pool=False):
        global theta_p, u_p, lambda_p, n_centroid, latent_dim
        n_centroid = self.n_centroid
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2
        self.conv_act  = conv_act
        self.pool   = pool

        ''' ENCODER '''
        input_shape = (28, 28, 1)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        #reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        enc_layers.append(Conv2D(32, (3,3), strides=(2,2), activation=conv_act, padding='same')(self.input))
        enc_layers.append(BatchNormalization()(enc_layers[-1]))
        enc_layers.append(Conv2D(64, (3,3), strides=(2,2), activation=conv_act, padding='same')(enc_layers[-1]))
        enc_layers.append(BatchNormalization()(enc_layers[-1]))

        input_conv = enc_layers[-1]

        flat = Flatten(name='flatten')(enc_layers[-1])
        
        dense_layers = []
        dense_layers.append(Dense(1024, activation=conv_act)(flat))
        dense_layers.append(Dense(256, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(64, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(16, activation=conv_act)(dense_layers[-1]))

        mu = Dense(2, name='mu', activation=None)(dense_layers[-1])
        latent_dim = mu.get_shape()[1]
        self.latent_dim = mu.get_shape()[1]
        #sigma = Dense(K.int_shape(mu)[1], name='sig', activation=None)(Flatten()(enc_layers[-1]))
        sigma = Dense(2, name='sigma', activation=None)(dense_layers[-1])

        latent_space = Lambda(compute_latent, output_shape=K.int_shape(mu)[1:], name='latent')([mu, sigma])
        
        # Build the encoder
        self.encoder = Model(encoder_input, [mu, sigma, latent_space], name='encoder')

        self.encoder.summary()

        self.mu = mu; self.sigma = sigma; self.z = latent_space
        
        self.gamma = GammaLayer(latent_dim, n_centroid, name='gamma')([mu, sigma, latent_space])
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(latent_space)[1:], name='dec_inp')
        dec_inps.append(decoder_input)

        dense_layers = []
        dense_layers.append(Dense(16, activation=conv_act)(decoder_input))
        dense_layers.append(Dense(64, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(256, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(1024, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(K.int_shape(flat)[1], activation='relu')(dense_layers[-1]))
        
        if batch_norm2:
            dec2_1 = BatchNormalization()(dense_layers[-1])
            dec3 = Reshape(conv_shape[1:])(dec2_1)
        else:
            dec3 = Reshape(conv_shape[1:])(dense_layers[-1])

        #upsamp_layer = UpSampling3D((2,2,1), name='dec_upsamp')(dec3)
        
        dec_layers = []
        dec_layers.append(Conv2DTranspose(64, (3, 3), strides=(2,2), padding='same',
                            activation=conv_act, name='dec_conv_1')(dec3))
        dec_layers.append(BatchNormalization()(dec_layers[-1]))
        dec_layers.append(Conv2DTranspose(1, (3, 3), strides=(2,2), padding='same',
                            activation=conv_act, name='dec_conv_2')(dec_layers[-1]))
        dec_layers.append(BatchNormalization()(dec_layers[-1]))
        decoder_output = dec_layers[-1]
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input)[2])
        
        # Build the VAE
        self.ae = Model(encoder_input, [self.output, self.gamma], name='VAE')


        self.cluster = Model(encoder_input, self.gamma, name='DC')

        print(self.cluster.summary())

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        #self.ae.enc_conv_flat = Model(encoder_input, enc_inps[0], name='enc_conv_flat')
        #self.ae.enc_hidden    = Model(Input(tensor=enc_inps[0]), [mu, sigma, z], name='enc_hidden')

        # set the training parameters
        self.ae.sig0     = sigma0#*K.ones_like(sigma)
        self.ae.kl_beta  = beta

        
        self.ae.summary()
        
    def add_loss_funcs(self):
        global theta_p, u_p, lambda_p

        recon_loss = K.sum(K.square(self.input - self.output), axis=(1,2))#*128*128

        mui,sigi, z  = self.encoder(self.input)
        mup, sigp, zp = self.encoder(self.output)

        gamma = self.cluster(self.input)
        Z = tf.transpose(K.repeat(z, self.n_centroid),perm=[0,2,1])
        ZMean = tf.transpose(K.repeat(mui, self.n_centroid),perm=[0,2,1])
        ZLogVar = tf.transpose(K.repeat(sigi, self.n_centroid),[0,2,1])
        gamma_t = K.repeat(gamma, self.latent_dim)

        batch_size = tf.shape(z)[0]

        gamma_layer = self.ae.get_layer('gamma')

        theta_tensor3, u_tensor3, lambda_tensor3 = gamma_layer.get_tensors(batch_size)

        sig0 = self.ae.sig0*K.ones_like(mui)
        
        kl = - 1 + (K.square(mui-mup) + K.exp(sig0))/K.exp(sig0)
        kl = K.mean(kl, axis=-1)
        kl *= 0.5*self.ae.kl_beta

        r_loss   = tf.nn.compute_average_loss(recon_loss)
        kl_loss  = tf.nn.compute_average_loss(kl)    

        print(f"gamma_t: {gamma_t.get_shape()}, gamma: {gamma.get_shape()}")


        a = K.sum(0.5*gamma_t*(self.latent_dim*K.log(np.pi*2)+K.log(lambda_tensor3)+
                        K.exp(ZLogVar)/lambda_tensor3+
                        K.square(ZMean-u_tensor3)/lambda_tensor3),axis=(1,2))
        b =  0.5*K.sum(sigi+1,axis=-1)
        c = K.sum(K.log(K.mean(theta_tensor3, axis=1)*gamma),axis=(1,2))
        d = K.sum(K.log(gamma)*gamma,axis=-1)

        #print(gamma_t.get_shape(), lambda_tensor3.get_shape(), ZLogVar.get_shape(), lambda_tensor3.get_shape(),
        #      ZMean.get_shape(), u_tensor3.get_shape())
        clust_loss = tf.nn.compute_average_loss(a-c+d)
        # sum of all three losses
        loss = r_loss + kl_loss + clust_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')
        self.ae.add_metric(clust_loss, aggregation='mean', name='clust')
            
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'mnistvae_cluster_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)
