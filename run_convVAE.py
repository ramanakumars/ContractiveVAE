#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from contractive_AE import ConvVAE
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

latent_dim    = 128
conv_filt     = 512
nconv         = 2
hidden        = [128, 16]
beta          = 0.5
learning_rate = 1.e-3
sigma0        = 0.
batch_norm    = True
batch_norm2   = False
pool          = False
conv_act      = 'tanh'

with nc.Dataset('../junodata/segments_20211229.nc', 'r') as dataset:
    data = dataset.variables['imgs'][:]

#data_augment = data[:,::-1,::-1,:]
#data_new = np.vstack((data, data_augment))
#print(data_new.shape)

vae = ConvVAE(latent_dim, conv_filt, hidden, nconv, batch_norm, batch_norm2)
vae.create_model(sigma0=sigma0, beta=beta, conv_act=conv_act, pool=pool)
vae.add_loss_funcs()
vae.compile(learning_rate=learning_rate, optimizer='Adam', decay=0.)
#vae.create_lr_scheduler(learning_rate, 0.95, 30)
#vae.load()

vae.name += "_sig0"
vae.train(data, epochs=1500, batch_size=32)

vae.save()

''' PLOT DIAGNOSTICS '''
savesfolder = vae.savesfolder

fig, ax = plt.subplots(dpi=150)
ax.plot(range(1, vae.nepochs+1, 1), vae.history.history['loss'], 'k-', label='training')
ax.plot(range(1, vae.nepochs+1, 1), vae.history.history['kl'], 'k-.', label='KL')
ax.plot(range(5, vae.nepochs+1, 5), vae.history.history['val_loss'], 'r-', label='validation')
ax.plot(range(5, vae.nepochs+1, 5), vae.history.history['val_kl'], 'r-.', label='val_KL')
ax.set_yscale('log')
ax.legend(loc='upper right', ncol=2)
ax.set_xlabel(r'Epoch')
ax.set_ylabel(r'Loss')
fig.savefig(savesfolder+"loss.png")

ind_all = np.asarray(range(len(data)))
np.random.shuffle(ind_all)
dshuffle   = data[ind_all,:,:,:]

mui, sigmai, z  = vae.encoder.predict(dshuffle, batch_size=32)
recon         = vae.ae.predict(dshuffle, batch_size=32)
mup, sigmap, zp = vae.encoder.predict(recon, batch_size=32)

plt.rc('figure', facecolor='white')

recon_loss = np.mean(np.mean((recon - dshuffle)**2.,axis=(1, 2))*128*128, axis=-1)

nz = 10
dz = int(len(z)//nz)

plt.figure(dpi=150)
plt.hist(recon_loss, bins=50)
# plt.yscale('log')
plt.xlabel(r'Loss')
plt.savefig(savesfolder+"loss_hist.png")

plt.figure(dpi=150)
for i in range(nz):
    plt.hist(z[(i*dz):(i+1)*dz,:].flatten(), bins=50, histtype='step')
# plt.yscale('log')
plt.xlabel(r'$z$')
plt.savefig(savesfolder+"z.png")

plt.figure(dpi=150)
for i in range(nz):
    plt.hist(mui[(i*dz):(i+1)*dz,:].flatten(), bins=50, histtype='step')
# plt.yscale('log')
plt.xlabel(r'$\mu$')
plt.savefig(savesfolder+"mu.png")

plt.figure(dpi=150)
for i in range(nz):
    plt.hist(sigmai[(i*dz):(i+1)*dz,:].flatten(), bins=50, histtype='step')
# plt.yscale('log')
plt.xlabel(r'$\sigma$')
plt.savefig(savesfolder+"sig.png")

data_recon = vae.ae.predict(data[:5,:,:,:])
fig, axs = plt.subplots(2, 5)
for i in range(5):
    axs[0,i].imshow(data[i,:,:])
    axs[1,i].imshow(data_recon[i,:,:])
    axs[0,i].axis('off')
    axs[1,i].axis('off')

plt.tight_layout()
plt.savefig(savesfolder+"recon.png", bbox_inches='tight')
