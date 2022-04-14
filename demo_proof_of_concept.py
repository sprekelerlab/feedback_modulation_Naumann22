"""
Demo script for proof of concept.
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
import analyses as an
from model_base import ModulationNet, Runner
from helpers import plot_violin

# change these flags to train, load and/or save models
LOAD_MODEL = False
SAVE_MODEL = True  # recommended to save model when it is trained de novo

##########################
# Step 1: setup & params #
##########################

# model parameter dict
params = {
          's_dim': 2,          # number of sources
          'n_hid': 100,        # number of hidden units in LSTM
          'tau': 1,            # timescale of modulation, increase to make modulation slower (e.g. tau=100)
          'mix_noise': 0.001,  # noise added to sensory stimuli
          'net_input': 'xy',   # input to LSTM, options: x, y or xy (both, default)
          'n_sample': 1000,    # number of samples (time steps) per context
}

# train/test parameters
n_batch = 5  # should be at least 3000, better 5000-10000, one batch should take around 1-2 seconds
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
model = ModulationNet(params['s_dim'], params['n_hid'], params['s_dim'] ** 2, tau=params['tau'],
                      input_to_net=params['net_input'], mix_noise=params['mix_noise'])

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
    load_file_name = f"trained_modulation_net_{n_batch}batch.pt"  # you must have a model by that name saved
    model = torch.load(os.path.join('trained_models', load_file_name))
    model.eval()
    runner.model = model  # link loaded model to runner for testing

# save model
if SAVE_MODEL:
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    save_path = os.path.join('trained_models', f"trained_modulation_net_{n_batch}batch.pt")
    torch.save(model, save_path)

# test
store_test = runner.test(inputs, data_test)


########################
# Step 3: Plot results #
########################

# set up figure
fig = plt.figure(figsize=(6, 5), dpi=150)
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], wspace=0.4, hspace=0.4, top=0.95, right=0.95)
sub_gs_a = gs[:, 0].subgridspec(7, 1)
ax_a = [fig.add_subplot(sub_gs_a[ii]) for ii in range(7)]
sub_gs_b = gs[0, 1].subgridspec(2, 1, height_ratios=[2, 1])
ax_b = [fig.add_subplot(sub_gs_b[ii]) for ii in range(2)]
ax_c = fig.add_subplot(gs[1, 1])

# colours
sigcol = ['#2C6E85', '#5FB89A']
darkred = '#A63A50'
meansigcol = '#2f9498'
c_weights = ['#E76161', '#FB816F', '#FCAB84', '#F1C998']

# a: plot sources, sensory stimuli, outputs and error
for ii in range(2):
    ax_a[0 + ii].plot(store_test['s'][:, ii], c=sigcol[ii], alpha=0.5)
    ax_a[2 + ii].plot(store_test['x'][:, ii], c='gray')
    ax_a[-3 + ii].plot(store_test['s'][:, ii], c=sigcol[ii], alpha=0.5)
    ax_a[-3 + ii].plot(store_test['y'][:, ii], '--', c=sigcol[ii])
ax_a[-1].plot(np.linalg.norm(store_test['s'] - store_test['y'], axis=1), c=darkred)

# b: plot readout weights and error of weights
ns, ne = 0, 6*params['n_sample']
n_mix = n_test
for ii in range(np.prod(model.w_dim)):
    ax_b[0].plot(store_test['wt'][:, ii], ':', c='k', linewidth=1)
    ax_b[0].plot(store_test['w'][:, ii], alpha=0.95, c=c_weights[ii], linewidth=1)
ax_b[1].plot(np.linalg.norm(store_test['wt'] - store_test['w'], axis=1), c=darkred)

# c: signal clarity
perf_stim, _ = an.get_signal_clarity(store_test['s'], store_test['x'], params, n_test=n_test)
perf_out, _ = an.get_signal_clarity(store_test['s'], store_test['y'], params, n_test=n_test)
plot_violin(ax_c, 0, perf_stim, color='gray')
plot_violin(ax_c, 1, perf_out, color=meansigcol)

# axis labels, cleaning up figure
a_ylabels = [r's$_1$', r's$_2$', r'x$_1$', r'x$_2$', r'y$_1$', r'y$_2$', '||s-y||']
ctx_show = 2  # which context to show (shows this and the one before)
xlims = [params['n_sample'] * ctx_show - 500, params['n_sample'] * ctx_show + 500]
for jj in range(7):
    ax_a[jj].set(xlim=xlims, xticks=[xlims[0], xlims[0]+500, xlims[1]], xticklabels=[], ylabel=a_ylabels[jj])
ax_a[-1].set(xticklabels=[-500, 0, 500], xlabel='time (a.u.)')
ax_b[0].set(xticks=np.arange(ns, ne, params['n_sample']), xticklabels=[], ylabel='readout $W$', xlim=[ns, ne],
            ylim=[-4, 4], yticks=[-3, 0, 3])
ax_b[1].set(xticks=np.arange(ns, ne, params['n_sample']), xticklabels=np.arange(6),
            ylabel=r'||$W$–$W^*$||', xlabel='# contexts', xlim=[ns, ne])
ax_c.set(xlim=[-1, 2], ylim=[0, 1], yticks=[0, 0.5, 1], xticks=[0, 1], xticklabels=['stim', 'output'],
         ylabel='signal clarity')
plt.show()
