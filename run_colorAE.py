#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from contractive_AE import ColorAE
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

latent_dim    = 128
conv_filt     = 2048
nconv         = 3
hidden        = [512, 128, 16]
beta          = 1.
learning_rate = 1.e-5
sigma0        = -4
batch_norm    = True
batch_norm2   = False
pool          = False
conv_act      = 'tanh'

with nc.Dataset('../junodata/segments.nc', 'r') as dataset:
    data = dataset.variables['imgs'][:]

vae = ColorAE(latent_dim, conv_filt, hidden, nconv, batch_norm, batch_norm2)
vae.create_model(conv_act=conv_act, pool=pool)
vae.add_loss_funcs()
vae.compile(learning_rate=learning_rate, optimizer='Adam')

vae.train(data, epochs=300, batch_size=32)

vae.save()

''' PLOT DIAGNOSTICS '''
savesfolder = vae.savesfolder

fig, ax = plt.subplots(dpi=150)
ax.plot(range(1, vae.nepochs+1, 1), vae.history.history['loss'], 'k-', label='training')
ax.plot(range(1, vae.nepochs+1, 1), vae.history.history['z'], 'k-.', label='z')
ax.plot(range(5, vae.nepochs+1, 5), vae.history.history['val_loss'], 'r-', label='validation')
ax.plot(range(5, vae.nepochs+1, 5), vae.history.history['val_z'], 'r-.', label='val_z')
ax.set_yscale('log')
ax.legend(loc='upper right', ncol=2)
ax.set_xlabel(r'Epoch')
ax.set_ylabel(r'Loss')
fig.savefig(savesfolder+"loss.png")

ind_all = np.asarray(range(len(data)))
np.random.shuffle(ind_all)
dshuffle   = data[ind_all,:,:,:]

z  = vae.encoder.predict(dshuffle)
recon         = vae.ae.predict(dshuffle)
zp = vae.encoder.predict(recon)

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
