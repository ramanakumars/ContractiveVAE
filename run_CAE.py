from contractive_AE import ContractiveAE
import netCDF4 as nc
import matplotlib.pyplot as plt

latent_dim    = 128
conv_filt     = 128
nconv         = 4
hidden        = [4096, 512]
lam           = 1.
learning_rate = 1.e-5
batch_norm    = False
batch_norm2   = True
contr_loss    = False

with nc.Dataset('../junodata/segments.nc', 'r') as dataset:
    data = dataset.variables['imgs'][:]

cae = ContractiveAE(latent_dim, conv_filt, hidden, nconv, batch_norm, batch_norm2)
cae.create_model(lam=lam)
cae.add_loss_funcs(contr_loss=contr_loss)
cae.compile(learning_rate=learning_rate)

cae.train(data, epochs=300, batch_size=16)

cae.save()

''' PLOT DIAGNOSTICS '''
savesfolder = cae.savesfolder

fig, ax = plt.subplots(dpi=150)
ax.plot(range(1, cae.nepochs+1, 1), cae.history.history['loss'], 'k-', label='training')
ax.plot(range(1, cae.nepochs+1, 1), cae.history.history['contr'], 'k--', label='contractive')
ax.plot(range(5, cae.nepochs+1, 5), cae.history.history['val_loss'], 'r-', label='validation')
ax.plot(range(5, cae.nepochs+1, 5), cae.history.history['val_contr'], 'r--', label='val_contractive')
ax.set_yscale('log')
ax.legend(loc='upper right', ncol=2)
ax.set_xlabel(r'Epoch')
ax.set_ylabel(r'Loss')
fig.savefig(savesfolder+"loss.png")

ind_all = np.asarray(range(len(data)))
np.random.shuffle(ind_all)
dshuffle   = data[ind_all,:,:,:]

z     = cae.encoder.predict(dshuffle)
recon = cae.ae.predict(dshuffle)
zp    = cae.encoder.predict(recon)

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

plt.show()
