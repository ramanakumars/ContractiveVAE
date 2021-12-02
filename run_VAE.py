from contractive_AE import VariationalAE
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

latent_dim    = 128
conv_filt     = 256
nconv         = 6
hidden        = [4096, 512, 256]
beta          = 5.
learning_rate = 1.e-5
sigma0        = -2
batch_norm    = False
batch_norm2   = True

with nc.Dataset('../junodata/segments.nc', 'r') as dataset:
    data = dataset.variables['imgs'][:]

vae = VariationalAE(latent_dim, conv_filt, hidden, nconv, batch_norm, batch_norm2)
vae.create_model(sigma0=sigma0, beta=beta, dense_act='tanh')
vae.add_loss_funcs()
vae.compile(learning_rate=learning_rate)

vae.train(data, epochs=300, batch_size=32)

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

mui, sigmai, z  = vae.encoder.predict(dshuffle)
recon         = vae.ae.predict(dshuffle)
mup, sigmap, zp = vae.encoder.predict(recon)

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



