"""
Demo script for the analyses showing invariant subspaces. Reproduces core findings of Fig. 4 and Fig. 5
- train a single network model
- test the trained model on new contexts
- save or load the trained model
- analyse and manipulate the trained model, revealing invariant subspaces
- plot results
- note that the results may depend on the random seed (e.g. some context changes are large and some are small),
  in the 3D plots you may want to play with different perspectives

Recommended use of this script:
1. Run once with LOAD_MODEL=False and SAVE_MODEL=True for 5000-10000 batches (will take several hours)
2. Run again with LOAD_MODEL=True and SAVE_MODEL=False to analyse model and visualise results

Note: it may take several hundred batches until the loss starts decreasing noticeably.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.decomposition import PCA
import data_maker as dm
import analyses as an
import helpers as hlp
from model_base import SpatialNet, Runner
from helpers import plot_violin

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

# generate sources – here we train the network on simple sines for visualisation purposes
inputs = dm.gen_sines(2, base_freq=100)  # generates two sources of 2s duration

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
    load_file_name = f"trained_spatial_net_sines_{n_batch}batch.pt"  # you must have a model by that name saved
    model = torch.load(os.path.join('trained_models', load_file_name))
    model.eval()
    runner.model = model  # link loaded model to runner for testing

# save model
if SAVE_MODEL:
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    save_path = os.path.join('trained_models', f"trained_spatial_net_sines_{n_batch}batch.pt")
    torch.save(model, save_path)

# test
store_test = runner.test(inputs, data_test)


################################
# Step 3: Analyse model & plot #
################################

# set up figure(s)
fig1 = plt.figure(figsize=(4, 2.5), dpi=150)
gs = fig1.add_gridspec(1, 2, wspace=0.6, hspace=0.4, top=0.95, right=0.95, left=0.15, bottom=0.17)
ax1_a = fig1.add_subplot(gs[0])
ax1_b = fig1.add_subplot(gs[1])

fig2 = plt.figure(figsize=(6, 5.5), dpi=150)
gs = fig2.add_gridspec(3, 3, height_ratios=[1.5, 1.5, 1], wspace=0.6, hspace=0.7, top=0.95, right=0.9)
sub_a = gs[0, :].subgridspec(3, 1)
ax2_a = [fig2.add_subplot(sub_a[ii]) for ii in range(3)]
ax2_b = [fig2.add_subplot(gs[1, ii], projection='3d') for ii in range(3)]
ax2_c = [fig2.add_subplot(gs[2, ii]) for ii in range(3)]

# colours
sigcol = ['#2C6E85', '#5FB89A']
darkred = '#A63A50'
darkblue = '#3A5BA7'
meansigcol = '#2f9498'
# col_m = hlp.colours(params['Nm'] + 2, 'BuPu')[2:]
col_m = hlp.colours(25, 'BuPu_r')[5:]
col_exp = [meansigcol, 'tomato', 'C2']


# Fig 1: Single neuron vs population invariance
# ---------------------------------------------

# fig1 a&b: signal clarity and population-level decoding of signals for different populations

cols = ['gray', darkblue, meansigcol]
var_list = ['x', 'z', 'y']
for ii, pop in enumerate(var_list):
    perf, corr = an.get_signal_clarity(store_test['s'], store_test[pop], params, n_test=n_test)
    mean_perf = np.mean(np.abs(np.abs(corr[:, 0, :]) - np.abs(corr[:, 1, :])), axis=0)
    plot_violin(ax1_a, ii, mean_perf, showmeans=True, color=cols[ii])

    r2_scores = an.linear_source_decoding(store_test[pop], store_test['s'], n_test//2, n_sample=params['n_sample'])
    plot_violin(ax1_b, ii, r2_scores, showmeans=True, color=cols[ii])

# axis labels, cleaning up figure
ax1_a.set(ylim=[0, 1], xticks=np.arange(len(var_list)), xticklabels=var_list, xlabel='population',
          yticks=[0, 0.5, 1], ylabel='signal clarity')
ax1_b.set(ylim=[0, 1], xticks=np.arange(len(var_list)), xticklabels=var_list, yticks=[0, 0.5, 1],
          xlabel='population', ylabel='signal decoding')


# Fig 2: Feedback maintains an invariant subspace
# -----------------------------------------------

# fit PCA on test data
pca = PCA(n_components=3, whiten=True)
pca.fit(store_test['z'])

# generate 2 more contexts
# run experiment: simulate model for one context, change context but freeze feedback, unfreeze feedback
_, data_test2 = dm.gen_train_test_data(1, 2, batch_size=1, mat_size=(params['s_dim'], params['s_dim']))
store = runner.test(inputs, data_test2, freeze_fb_after=1)

# a: plot timecourse of readout and modulation
ts = 500
ax2_a[0].plot(store['s'][ts:, 0], c=sigcol[0], alpha=0.5)
ax2_a[0].plot(store['y'][ts:, 0], c=sigcol[0], ls='--')
ax2_a[1].plot(store['s'][ts:, 1], c=sigcol[1], alpha=0.5)
ax2_a[1].plot(store['y'][ts:, 1], c=sigcol[1], ls='--')
for ii in range(10):
    ax2_a[2].plot(store['m'][ts:, ii * 10], c=col_m[ii], lw=1)  # plot modulation to a subset of the neurons

# b&c: plot population activity for 3 phases of experiment, in 3d space (readout + 1st PC) and 2 space (readout)
pcs = pca.transform(store['z'])  # project data into pc space
out = store['y']
subs = np.array([store['y'][:, 0], store['y'][:, 1], pcs[:, 0]]).T

for ii in range(3):
    ts, te = (ii+1)*500+200, (ii+2)*500  # time range to plot, skips first 200 time steps of an experiment phase

    # plot subspaces in 3D and projection into readout to the bottom of 3D plot
    ax2_b[ii].plot3D(subs[ts:te, 0], subs[ts:te, 1], subs[ts:te, 2], color=col_exp[ii])
    ax2_b[ii].plot3D(subs[ts:te, 0], subs[ts:te, 1], np.ones(te-ts) * (-3), color=col_exp[ii], alpha=0.5, ls=':', lw=1)

    # plot 2d plane of first experiment stage into 3d plot
    if ii == 0:
        t1, t2, t3 = 650, 783, 849  # choose 3 random points from the first experiment phase
        xx, yy, zz = hlp.get_plane(subs[:, 0], subs[:, 1], subs[:, 2], t1=t1, t2=t2, t3=t3)  # get plane for time series
    ax2_b[ii].plot_surface(xx, yy, zz, alpha=0.2, color=meansigcol)  # plot plane

    # plot activity in 2D readout space
    ax2_c[ii].plot(store['s'][ts:te, 0], store['s'][ts:te, 1], color=meansigcol, alpha=0.5)
    ax2_c[ii].plot(out[ts:te, 0], out[ts:te, 1], color=col_exp[ii], ls=':')

# compute angle between subspaces and plot into figure
subs_cnt, subs_frz, subs_unf = subs[500:1000], subs[1000:1500], subs[1500:2000]
nv_cnt, nv_frz, nv_unf = an.get_normal_vector(subs_cnt), an.get_normal_vector(subs_frz), an.get_normal_vector(subs_unf)
angle_context = an.compute_angle(nv_cnt, nv_frz)
angle_fb = an.compute_angle(nv_cnt, nv_unf)
ax2_b[0].set_title('Original subspace', size=10)
ax2_b[1].set_title(f"$\Delta$ angle = {angle_context:2.1f} deg", size=10)
ax2_b[2].set_title(f"$\Delta$ angle = {angle_fb:2.1f} deg", size=10)

# determine context distance and whether new and old context are on the same or different sides of context space
# result is printed
atrue = np.array([np.linalg.inv(store['wt'][kk].flatten().reshape((2, 2))) for kk in [0, params['n_sample']]])
ctx_dist = np.linalg.norm(atrue[0] - atrue[1])
ctx_side = (atrue[0, 0, 0] > atrue[0, 1, 0]) == (atrue[1, 0, 0] > atrue[1, 1, 0])
ctx_side_str = 'same' if ctx_side else 'different'
print(f"Context dist: {ctx_dist:1.2f}; {ctx_side_str} side")

# axis labels and cleaning up
ax2_a[0].set(xlim=[0, 1500], xticks=[], ylim=[-1.5, 1.5], ylabel=f's$_1$/y$_1$')
ax2_a[1].set(xlim=[0, 1500], xticks=[], ylim=[-1.5, 1.5], ylabel=f's$_2$/y$_2$')
ax2_a[2].set(xlim=[0, 1500], ylim=[-6, 6], xlabel='time (a.u.)', ylabel='mod')
# add rectangle to indicate freezing of feedback
rect = plt.Rectangle((500, 0), width=500, height=3.2 + fig2.subplotpars.wspace, transform=ax2_a[2].get_xaxis_transform(),
                     clip_on=False, edgecolor="None", facecolor="tomato", linewidth=3, alpha=0.15)
ax2_a[2].add_patch(rect)
for ii in range(3):
    ax2_b[ii].set(xlim=[-2, 2], ylim=[-2, 2], zlim=[-3, 3], xlabel=r'y$_1$', ylabel=r'y$_2$', zlabel=f'PC$_1$')
    ax2_c[ii].set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], xlabel=r'y$_1$', ylabel=r'y$_2$')

plt.show()
