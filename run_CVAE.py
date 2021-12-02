from contractive_AE import ContractiveVAE
import netCDF4 as nc
import matplotlib.pyplot as plt

latent_dim    = 128
conv_filt     = 256
nconv         = 3
hidden        = [512]
lam           = 1.
beta          = 5.
learning_rate = 1.e-5
sigma0        = -4
batch_norm    = False
batch_norm2   = True
contr_loss    = True

with nc.Dataset('../junodata/segments.nc', 'r') as dataset:
    data = dataset.variables['imgs'][:]

cvae = ContractiveVAE(latent_dim, conv_filt, hidden, nconv, batch_norm, batch_norm2)
cvae.create_model(sigma0=sigma0, lam=lam, beta=beta)
cvae.add_loss_funcs(contr_loss=contr_loss)
cvae.compile(learning_rate=learning_rate)

cvae.train(data, epochs=300, batch_size=16)

cvae.save()

''' PLOT DIAGNOSTICS '''
savesfolder = cvae.savesfolder

fig, ax = plt.subplots(dpi=150)
ax.plot(range(1, cvae.nepochs+1, 1), cvae.history.history['loss'], 'k-', label='training')
ax.plot(range(1, cvae.nepochs+1, 1), cvae.history.history['contr'], 'k--', label='contractive')
ax.plot(range(1, cvae.nepochs+1, 1), cvae.history.history['kl'], 'k-.', label='KL')
ax.plot(range(5, cvae.nepochs+1, 5), cvae.history.history['val_loss'], 'r-', label='validation')
ax.plot(range(5, cvae.nepochs+1, 5), cvae.history.history['val_contr'], 'r--', label='val_contractive')
ax.plot(range(5, cvae.nepochs+1, 5), cvae.history.history['val_kl'], 'r-.', label='val_KL')
ax.set_yscale('log')
ax.legend(loc='upper right', ncol=2)
ax.set_xlabel(r'Epoch')
ax.set_ylabel(r'Loss')
fig.savefig(savesfolder+"loss.png")

ind_all = np.asarray(range(len(data)))
np.random.shuffle(ind_all)
dshuffle   = data[ind_all,:,:,:]

mui, sigmai, z  = cvae.encoder.predict(dshuffle)
recon         = vae.predict(dshuffle)
mup, sigmap, zp = cvae.encoder.predict(recon)

plt.rc('figure', facecolor='white')

recon_loss = np.mean(np.mean((recon - dshuffle)**2.,axis=(1, 2))*128*128, axis=-1)

nz = 10
dz = int(len(z)//nz)

plt.figure(dpi=150)
plt.hist(recon_loss, bins=50)
# plt.yscale('log')
plt.xlabel(r'Loss')
plt.savefig(savesfolder+"loss.png")

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

plt.show()

