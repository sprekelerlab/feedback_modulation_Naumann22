"""
Demo script for the network with spatially diffuse modulation.
- train a single network model
- test the trained model on new contexts
- save or load the trained model
- plot the resulting model behaviour and performance

Recommended use of this script:
1. Run once with LOAD_MODEL=False and SAVE_MODEL=True for 5000-10000 batches (will take several hours)
2. Run again with LOAD_MODEL=True and SAVE_MODEL=False to visualise results and play with the trained model

Note: it may take several hundred batches until the loss starts decreasing noticeably.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import data_maker as dm
import helpers as hlp
from model_base import SpatialNet, Runner

# change these flags to train, load and/or save models
LOAD_MODEL = False
SAVE_MODEL = True  # recommended to save model when it is trained de novo

##########################
# Step 1: setup & params #
##########################

# model parameter dict; some parameters are not explicitly specified, such that defaults will be used (see model_base)
params = {
          's_dim': 2,          # number of sources
          'n_hid': 100,        # number of hidden units in LSTM
          'Nz': 100,           # number of neurons in the neural population
          'Nm': 4,             # number of feedback signals (default: 4)
          'mod_width': 0.2,    # width of diffusive modulation (default: 0.2)
          'tau': 100,          # timescale of modulation (default 100)
          'n_sample': 1000,    # number of samples (time steps) per context
}

# train/test parameters
n_batch = 10000  # should be at least 3000, better 5000-10000, one batch should take around 1-2 seconds
batch_size = 32
n_train = n_batch*batch_size
n_test = 20
lr = 0.001

# generate sources
inputs = dm.gen_chords(2, base_freq=100)  # generates two sources of 2s duration

# generate train & test contexts (mixings)
data_train, data_test = dm.gen_train_test_data(n_train, n_test, batch_size=batch_size, mat_size=(params['s_dim'],
                                                                                                 params['s_dim']))

############################
# Step 2: Train/test model #
############################

# create model
model = SpatialNet(params['s_dim'], params['n_hid'], params['s_dim'] ** 2, tau=params['tau'], Nz=params['Nz'],
                   Nm=params['Nm'], mod_width=params['mod_width'])

# set up loss function and optimiser
loss_function = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
writer = None  # can be replace by tensorflow writer
runner = Runner(model, loss_function, optimizer, writer, batch_size=batch_size)

# train/load model
if not LOAD_MODEL:
    store_train = runner.train(inputs, data_train, nt=params['n_sample'])
    # plot loss over time
    fig, ax = plt.subplots()
    ax.semilogy(store_train['loss'])
    ax.set(xlabel='# batch', ylabel='loss')
    plt.show()
elif LOAD_MODEL:
    load_file_name = f"trained_spatial_net_chords_{n_batch}batch.pt"  # you must have a model by that name saved
    model = torch.load(os.path.join('trained_models', load_file_name))
    model.eval()
    runner.model = model  # link loaded model to runner for testing

# save model
if SAVE_MODEL:
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    save_path = os.path.join('trained_models', f"trained_spatial_net_chords_{n_batch}batch.pt")
    torch.save(model, save_path)

# test
store_test = runner.test(inputs, data_test)


########################
# Step 3: Plot results #
########################

# set up figure
fig = plt.figure(figsize=(6, 5), dpi=150)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], wspace=0.4, hspace=0.4, top=0.95, right=0.9)
ax_a = fig.add_subplot(gs[0, 0])
sub_gs_b = gs[0, 1].subgridspec(3, 1)
ax_b = [fig.add_subplot(sub_gs_b[ii]) for ii in range(3)]
sub_gs_c = gs[1, :].subgridspec(2, 1, height_ratios=[2, 1])
ax_c = [fig.add_subplot(sub_gs_c[ii]) for ii in range(2)]

# colours
sigcol = ['#2C6E85', '#5FB89A']
darkred = '#A63A50'
col_m = hlp.colours(params['Nm'] + 2, 'BuPu')[2:]

# a: plot spatial extent of modulation
for i_m in range(params['Nm']):
    ax_a.plot(model.Wm[:, i_m], np.arange(params['Nz']), c=col_m[i_m])

# b: plot sources, outputs and error
for ii in range(2):
    ax_b[ii].plot(store_test['s'][:, ii], c=sigcol[ii], alpha=0.5)
    ax_b[ii].plot(store_test['y'][:, ii], '--', c=sigcol[ii])
ax_b[-1].plot(np.linalg.norm(store_test['s'] - store_test['y'], axis=1), c=darkred)

# c: plot modulation and error in context
tstart, tend = 1000, 9000
pcm = ax_c[0].imshow(store_test['w'][tstart:tend, ::-1].T, cmap='RdBu_r', vmin=-7, vmax=7, aspect='auto')
ax_c[1].plot(np.linalg.norm(store_test['s'] - store_test['y'], axis=1)[tstart:tend], c=darkred)

# axis labels, cleaning up figure
ax_a.set(ylabel='# neuron', xlabel='FB modulation', xticks=[0, 1], yticks=[0, 50, 100])

b_ylabels = ['s1/y1', 's2/y2', '||s-y||']
ctx_show = 2  # which context to show (shows this and the one before)
xlims = [params['n_sample'] * ctx_show - 500, params['n_sample'] * ctx_show + 500]
for jj in range(3):
    ax_b[jj].set(xlim=xlims, xticks=[xlims[0], xlims[0]+500, xlims[1]], xticklabels=[], ylabel=b_ylabels[jj])
ax_b[-1].set(xticklabels=[-500, 0, 500], xlabel='time (a.u.)')

cax = ax_c[0].inset_axes([1.02, 0., 0.02, 1], transform=ax_c[0].transAxes)
cbar = fig.colorbar(pcm, ax=ax_c[0], cax=cax, ticks=[-5, 0, 5])
cbar.set_label('mod', labelpad=-20, y=1.15, rotation=0)
ax_c[0].set(ylabel='# neuron', xlim=[0, 8000], xticks=[], yticks=[0, 50, 100], yticklabels=[100, 50, 0])
ax_c[1].set(xlim=[0, 8000], xlabel='# context', ylabel=r'||s-y||', xticks=np.arange(0, 8001, 1000),
            xticklabels=np.arange(9))

plt.show()
