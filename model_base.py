"""
Classes to create and run models.
- model classes: UnmixingModel (parent), ModulationNet, SpatialNet, DaleRateNet (, DirectDemixNet)
- running models: Runner class contains train and test method
- revive_model function can be used to continue running a previous created model from its last state
"""

# import packages (requires PyTorch installation)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import torch.utils.tensorboard as tb
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # needed to fix bug (on Mac OS)


class UnmixingModel(nn.Module):
    """Parent class for unmixing models. Contains forward pass method.
       How readout weights are computed is specific to the model architecture.
       In this basic example the RNN output ("network_output") directly determines the readout weights (w_new)."""
    def __init__(self, signal_dim, hidden_dim, output_dim, input_to_net='xy', mix_noise=0.001, **kwargs):
        super(UnmixingModel, self).__init__()

        # core model parameters
        self.signal_dim = signal_dim  # dimension of signal
        self.hidden_dim = hidden_dim  # number of hidden states in LSTM
        self.x_dim = signal_dim  # x and y have same dimension as input signals
        self.y_dim = signal_dim
        self.w_dim = (self.x_dim, self.y_dim)
        self.output_dim = output_dim  # number of outputs, here x_dim x y_dim
        if len(input_to_net) > 1:  # i.e. input_to_net = xy, then concatenate x (signals) and y (FF net output)
            self.input_dim = self.x_dim+self.y_dim
        else:
            self.input_dim = eval('self.'+input_to_net+'_dim')
        self.input_to_net = input_to_net
        self.w0 = torch.zeros(1)  # not relevant for the base class
        # self.learn_baseline = 0
        self.mix_noise = mix_noise

        # self.Ni = 0

        # model architecture
        self.rec_net = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, w_old, hidden):
        """
        Runs the forward pass of the unmixing model.
        Note:    w_old, w_out and hidden are lists for compatibility with more complex models.
        Input:   signal mixtures x, readout weights and hidden states from previous time step
        Output:  output signals y, new readout weights and hidden states
        """

        # compute output signal y from previous readout and mixture
        y = torch.einsum("bsij, bsj -> bsi", w_old[0], x)

        # determine model input (x, y or both)
        if 'xy' in self.input_to_net:
            network_input = torch.cat((x, y), 2)
        elif self.input_to_net == 'x':
            network_input = x
        else:  # network input is only the input signal
            network_input = y

        # run model (LSTM & readout layer)
        lstm_out, hidden_out = self.rec_net(network_input, hidden)
        network_output = self.hidden2out(lstm_out)

        # get new readout weights (particular to model architecture)
        w_out = self.get_new_w(w_old[0], network_output.reshape((-1, 1)+self.w_dim))

        return (w_out,), y, hidden_out, network_output

    def get_new_w(self, w_old, net_out):
        """ w_new = LSTM(input) """

        return net_out


class ModulationNet(UnmixingModel):
    """Instantaneous or integrated modulation, multiplicative or additive."""
    def __init__(self, input_dim, hidden_dim, output_dim, input_to_net='xy', w0_var=0.001, add=0, tau=1,
                 mix_noise=0.001, **kwargs):
        super(ModulationNet, self).__init__(input_dim, hidden_dim, output_dim, input_to_net=input_to_net,
                                            mix_noise=mix_noise)

        # modulation type and timescale
        self.add = add
        self.tau = tau

        # init of baseline weight
        if add == 0:
            w0_mean = 1
        else:
            w0_mean = 0
        self.w0 = torch.normal(w0_mean, w0_var, size=self.w_dim)

    def get_new_w(self, w_old, net_out):
        """ tau dW = W0 */+ LSTM(input) - W
            Note that this is the same as tau dM = LSTM(input) - M ; W = W0 */+ M """

        if self.add:
            w_new = w_old + (self.w0 + net_out - w_old) / self.tau
        else:
            w_new = w_old + (self.w0 * net_out - w_old) / self.tau
        return w_new


class SpatialNet(ModulationNet):
    """Network with reduced spatial specificity. Mixture is projected into a higher dimensional 'middle layer' z.
       The input weights to the middle layer are modulated by a small number of modulation units m. The spatial
       specificity of modulation is determined by the modulation kernel Wm."""
    def __init__(self, input_dim, hidden_dim, output_dim, input_to_net='xy', w0_var=0.001, tau=1, mix_noise=0.001,
                 Nz=10, Nm=2, mod_width=1, learn_baseline=False, learn_ro=False, **kwargs):

        super(SpatialNet, self).__init__(input_dim, hidden_dim, Nm, input_to_net=input_to_net, w0_var=w0_var,
                                         mix_noise=mix_noise, tau=tau)

        self.w_dim = (Nz,)  # it's not really w_dim but the dimension of modulation
        self.Nz = Nz  # number of units in middle layer
        self.Nm = Nm  # number of modulation units (= output_dim)

        # baseline and readout weights (x->z and z->y)
        # - if FF weights are optimised, they're created as a torch parameter, this adds them to the optimisation
        self.learn_baseline = learn_baseline
        w0 = torch.normal(0, 0.5, size=(Nz, self.x_dim))  # baseline FF weights
        if learn_baseline:
            self.w0 = torch.nn.Parameter(w0)
        else:
            self.w0 = w0
        self.learn_ro = learn_ro
        Wro = torch.normal(0, 0.5, size=(self.y_dim, Nz))  # readout weights
        if learn_ro:
            self.Wro = torch.nn.Parameter(Wro)
        else:
            self.Wro = Wro

        # set up modulation kernel Wm
        if mod_width == 0:  # box-like kernel
            self.Wm = torch.zeros((Nz, Nm))
            for i_m in range(Nm):
                nmod = Nz//Nm
                self.Wm[nmod*i_m:nmod*(i_m+1), i_m] = 1
        elif mod_width >= 10:  # flat kernel
            self.Wm = torch.ones((Nz, Nm))
        else:  # van Mises kernel with spatial extent "mod_width"
            self.Wm = torch.zeros((Nz, Nm))
            zloc = torch.from_numpy(np.linspace(0, 2*np.pi, Nz, endpoint=False))
            for i_m in range(Nm):
                wm = torch.exp(1/mod_width*torch.cos((zloc-2*np.pi*i_m/Nm)))
                self.Wm[:, i_m] = wm/wm.max()

    def forward(self, x, m_old, hidden):
        """
        Runs the forward pass of the SpatialNet model.
        Note:   m_old, m_out and hidden are lists
        Input:  signal mixtures x, modulation to z and hidden states from previous time step
        Output: output signals y, new readout weights and hidden states
        """

        z = m_old[0] * torch.einsum("ij, bsj -> bsi", self.w0, x)
        y = torch.einsum("ij, bsj -> bsi", self.Wro, z)

        # determine model input (x, y or both)
        if 'xy' in self.input_to_net:
            network_input = torch.cat((x, y), 2)
        elif self.input_to_net == 'x':
            network_input = x
        else:  # network input is only the input signal
            network_input = y

        # run model
        lstm_out, hidden_out = self.rec_net(network_input, hidden)
        network_output = self.hidden2out(lstm_out)
        M_new = self.get_new_m(m_old[0], torch.einsum("ij, bsj -> bsi", self.Wm, network_output))

        return (M_new, z), y, hidden_out, network_output

    def get_new_m(self, M_old, net_out):
        """ tau dM = Wm(*)LSTM(input) - M , where (*) is a convolution with the modulation kernel"""

        return M_old + (net_out - M_old) / self.tau


class DaleRateNet(SpatialNet):
    """ Biologically motivated version of the model.
        Mixed and output signals are represented by populations of rate-based units ('neurons'..) and connected
        with Dalean weights. Model consists of 3 layers: x (sensory stim), z_L (lower-level, not modulated) and z_H
        (higher-level, modulated).
    """

    def __init__(self, signal_dim, hidden_dim, output_dim, input_to_net='xy', tau=1, inh_frac=1, Nz=40, Nm=4, NzL=40,
                 learn_baseline=False, learn_wx=False, learn_ro=False, mod_width=1, mod_target=0, **kwargs):
        super(DaleRateNet, self).__init__(signal_dim, hidden_dim, Nm, input_to_net=input_to_net, tau=tau, Nz=Nz, Nm=Nm,
                                          mod_width=mod_width, learn_ro=learn_ro)

        # neuron numbers and dimensions
        self.NzL = NzL
        self.w_dim = (Nz,)  # dimension of modulation to middle layer

        # compute weight scaling depending on modulation target (mean modulation is 1/2)
        self.mod_target = mod_target
        scale = 20
        if self.mod_target == 0:  # modulation targets exc and inh
            scale_exc = scale
            scale_inh = scale
        elif self.mod_target == 1:  # modulation targets only exc
            scale_exc = scale
            scale_inh = scale/2
        elif self.mod_target == 2:  # modulation targets only inh
            scale_exc = scale/2
            scale_inh = scale

        # which FF weights to optimise during training
        self.learn_baseline = learn_baseline
        self.learn_ro = learn_ro  # (used in parent class SpatialNet)
        self.learn_wx = learn_wx

        # weights from sensory stimuli to lower level population (x -> z_L)
        wx = torch.normal(0, 0.5, size=(NzL, signal_dim))
        if learn_wx:
            self.wx = torch.nn.Parameter(wx)
        else:
            self.wx = wx

        # baseline weight (z_L -> z_H, positive only, absolute taken in forward pass)
        w0 = torch.abs(torch.normal(1, 0.5, size=(Nz, NzL))/Nz*scale_exc)
        if learn_baseline:
            self.w0 = torch.nn.Parameter(w0)  # add to optimisation if learned
        else:
            self.w0 = w0

        # feedforward inhibition (z_L -> i -> z_H)
        N_inh = int(round(inh_frac*NzL))
        w_ix = torch.abs(torch.normal(1, 0.5, size=(N_inh, NzL))/N_inh)
        w_zi = torch.abs(torch.normal(1, 1, size=(Nz, N_inh))/Nz*scale_inh)
        self.w_inh = w_zi @ w_ix  # effective inhibition is precomputed

        # non-linearities
        self.rectify = torch.nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mod_old, hidden):
        """
        Runs the forward pass for this model variant.
        """

        z_L = self.rectify(torch.einsum("ij, bsj -> bsi", self.wx, x))
        inh = torch.einsum("ij, bsj -> bsi", self.w_inh, z_L)

        # modulation can target exc inputs, inh inputs, or both (=gain modulation)
        if self.mod_target == 0:  # target exc & inh
            z_H = mod_old[1] * self.rectify(torch.einsum("ij, bsj -> bsi", torch.abs(self.w0), z_L)-inh)
        elif self.mod_target == 1:  # target only exc
            z_H = self.rectify(mod_old[1]*torch.einsum("ij, bsj -> bsi", torch.abs(self.w0), z_L)-inh)
        elif self.mod_target == 2:  # target only inh
            z_H = self.rectify(torch.einsum("ij, bsj -> bsi", torch.abs(self.w0), z_L)-mod_old[1]*inh)

        y = torch.einsum("ij, bsj -> bsi", self.Wro, z_H)

        # determine model input (x, y or both)
        if 'xy' in self.input_to_net:
            network_input = torch.cat((x, y), 2)
        elif self.input_to_net == 'x':
            network_input = x
        else:  # network input is only the input signal
            network_input = y

        # run model
        lstm_out, hidden_out = self.rec_net(network_input, hidden)
        net_out = self.hidden2out(lstm_out)
        m = self.get_new_m(mod_old[0], net_out)
        p = 1-self.sigmoid(torch.einsum("ij, bsj -> bsi", self.Wm, m))

        return (m, p, z_L, z_H), y, hidden_out, net_out

    # def get_new_m(self, m_old, net_out):  (same as SpatialNet)  # TODO: could be removed, I think?
    #     """ tau dm = u(t) - m """
    #
    #     return m_old + (net_out - m_old) / self.tau


class DirectDemixNet(UnmixingModel):
    """Network directly unmixes signals: Inputs -> RNN -> Output (not part of the manuscript Naumann et al., 2021)"""
    def __init__(self, signal_dim, hidden_dim, output_dim, input_to_net='xy', mix_noise=0.001, **kwargs):
        super(DirectDemixNet, self).__init__(signal_dim, hidden_dim, output_dim, input_to_net=input_to_net,
                                             mix_noise=mix_noise)

        self.input_dim = signal_dim

        self.rec_net = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.hidden2out = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x, w_old, hidden):

        # run model
        lstm_out, hidden_out = self.rec_net(x, hidden)
        y = self.hidden2out(lstm_out)

        return (self.w0,), y, hidden_out, y


class Runner:

    def __init__(self, model, loss_function, optimizer, writer, batch_size=1):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.writer = writer
        self.batch_size = batch_size

    def numpify(self, var, bs=2):
        """Helper. Converts torch variables to numpy by detaching and – if necessary – slicing."""
        if type(var) is np.ndarray:
            return var
        else:
            var_np = var.detach().numpy()
            if bs > 1:
                var_np = var_np[0]
            if hasattr(var_np, "__len__"):
                var_np = var_np.flatten()
        return var_np

    def update_store(self, store, s, y, w, wt):
        """Update store dictionary with current state of variables."""

        store['s'].append(self.numpify(s))
        store['y'].append(self.numpify(y))
        store['wt'].append(self.numpify(wt))
        store['w'].append(self.numpify(w[0]))

        return store

    def reset_store(self, store):
        """Reset all arrays in a store dictionary."""

        for k in store.keys():
            store[k] = []
        return store

    def plot_intermediate_to_writer(self, store, name, k):
        """Plot snapshot of current network behaviour to tensorboard."""

        fig, ax = plt.subplots(2, 1, dpi=200, sharex=True)
        ax[0].plot(store['wt'], 'k--')
        ax[0].plot(store['w'])
        ax[1].plot(store['s'])
        ax[1].plot(store['y'], '--')
        ax[0].set(ylabel='weights', ylim=[-4, 4])
        ax[1].set(xlabel='#samples x #mixings', ylabel='signals', ylim=[-1.2, 1.2])
        self.writer.add_figure(name, fig, k)

    def log_gradients_to_writer(self, weights, rid, k):
        """Log gradients of model to tensorboard writer."""

        self.writer.add_scalar('grad_hh_weights'+rid, torch.sum(torch.abs(weights[0][1].grad)), k)
        self.writer.add_scalar('grad_hh_bias'+rid, torch.sum(torch.abs(weights[0][3].grad)), k)
        self.writer.add_scalar('grad_ih_weights'+rid, torch.sum(torch.abs(weights[0][0].grad)), k)
        self.writer.add_scalar('grad_ih_bias'+rid, torch.sum(torch.abs(weights[0][2].grad)), k)

        # how to access specific weight matrices, if needed:
        #  self.writer.add_histogram('lstm_weights_hi'+wid, self.model.lstm.all_weights[0][1][:nhid, :], k)
        #  self.writer.add_histogram('lstm_weights_hf'+wid, self.model.lstm.all_weights[0][1][nhid:2*nhid, :], k)
        #  self.writer.add_histogram('lstm_weights_hg'+wid, self.model.lstm.all_weights[0][1][2*nhid:3*nhid, :], k)
        #  self.writer.add_histogram('lstm_weights_ho'+wid, self.model.lstm.all_weights[0][1][3*nhid:, :], k)

    def train(self, signals, train_data, nt=1000, max_grad_value=1, plot_every=100, log_every=50, run_id='',
              save_weight_loss=0, lambda_reg=0):
        """Train model with a given set of signals and mixing matrices using TBPTT.
           Return dict of batch-loss (output-signal diff) and mean deviation from target weight in every sample."""

        # get info from inputs
        n_batch = len(train_data)  # number of batches
        lent = signals.shape[0]    # number of samples in each context
        nsig = signals.shape[1]    # number of source signals

        # booleans for comparing effective weights to target (only for UnmixingModel and ModulationNet)
        if save_weight_loss == 0:  # if weight loss should not be saved over learning, only save it in last batch
            save_weight_loss = n_batch-1
        compute_w_loss = len(self.model.w_dim) > 1  # w quadratic, only then can W be true inverse of A

        # initialise hidden states and weights
        hidden_init = (torch.zeros(1, self.batch_size, self.model.hidden_dim),
                       torch.zeros(1, self.batch_size, self.model.hidden_dim))
        hidden_states = [hidden_init]
        if isinstance(self.model, DaleRateNet):  # in this model variant p also needs to be initialised
            m_init = torch.zeros((self.batch_size, 1, self.model.Nm))
            p_init = torch.normal(0.5, 0.01, size=(self.batch_size, 1)+self.model.w_dim)
            w_states = [(m_init, p_init)]
        else:
            w_init = torch.zeros((self.batch_size, 1)+self.model.w_dim)
            w_states = [(w_init,)]

        # dictionaries for storing stuff
        store = {'loss': [], 'loss_w': []}
        store_intermediate = {'s': [], 'y': [], 'w': [], 'wt': [], 'p': [], 'm': [], 'xb': [], 'z': []}

        # loop over mixing matrices, i.e. contexts (model trained in chunks)
        print('Training model...')
        for k, (A, A_inv) in enumerate(train_data):
            w_target = A_inv.reshape((self.batch_size, 1, self.model.y_dim, -1))  # target weights (mixing inverse)

            # start timer, reset loss and gradients
            print(f"batch {k + 1}/{n_batch}")
            start = time.time()
            loss_all = torch.zeros(1)
            net_out_sum = torch.zeros(1)
            loss_w_single = torch.zeros(nt)
            self.optimizer.zero_grad()

            # if there are more signals available than needed, choose randomly (for frequency generalisation)
            # - note that signals need to be sorted to avoid ambiguities between order and mixture
            if signals.shape[1] > self.model.signal_dim:
                sig_idx = np.sort(np.random.choice(np.arange(nsig), self.model.signal_dim, replace=False))
            else:
                sig_idx = np.arange(self.model.signal_dim)

            # randomly phase-shift signals or use random part of signals
            if lent > nt:  # if source signals are longer than sequence length --> select random part
                tstart = np.random.randint(0, lent-nt, size=self.model.signal_dim)
                s_use = np.array([signals[tstart[ii]:tstart[ii] + nt, idx] for ii, idx in enumerate(sig_idx)]).T
            else:  # if source signals are not longer than sequence length --> roll
                tshift = np.random.randint(0, lent, size=self.model.signal_dim)
                s_use = np.array([np.roll(signals[:, idx], tshift[ii]) for ii, idx in enumerate(sig_idx)]).T

            # loop over data points in sequence (fixed mixing/context)
            for j, s_ in enumerate(s_use):

                # run forward model pass
                s = torch.reshape(torch.from_numpy(s_), (1, 1, self.model.signal_dim)).repeat(self.batch_size, 1, 1)
                x = torch.einsum("bij, bsj -> bsi", A, s) \
                    + torch.normal(0, self.model.mix_noise, size=(self.batch_size, 1, self.model.x_dim))
                w, y, hidden, net_out = self.model(x, w_states[-1], hidden_states[-1])

                # compute loss
                loss = self.loss_function(s, y)
                loss_all += loss
                net_out_sum += net_out.abs().sum()
                if compute_w_loss:
                    loss_w_single[j] = torch.mean((w[0]-w_target)**2)

                # append states
                w_states.append(w)
                hidden_states.append(hidden)

                # store intermediate results (for 3 different mixings/contexts)
                if k % plot_every in [0, 1, 2]:
                    self.update_store(store_intermediate, s, y, w, w_target)

            # do backprop, update parameters and detach
            loss_tot = loss_all + lambda_reg*net_out_sum
            loss_tot.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), max_grad_value)
            self.optimizer.step()
            hidden_states = [(hidden_states[-1][0].detach(), hidden_states[-1][1].detach())]
            w_states = [tuple([w_i.detach() for w_i in w_states[-1]])]

            # print info about loss and regulariser
            print(f"\t loss: {loss_all.item():3.3f}, took {time.time() - start:1.1f}s")
            print(f"\t regularisation term: {lambda_reg*net_out_sum.item():3.3f}")

            # store loss
            store['loss'].append(self.numpify(loss_all))
            if k % save_weight_loss == 0:
                store['loss_w'].append(self.numpify(loss_w_single, bs=1))

            # log loss to tensorboard
            if self.writer is not None and k % log_every == 0:
                self.writer.add_scalar('loss'+run_id, loss_all, k)
                self.log_gradients_to_writer(self.model.rec_net.all_weights, run_id, k)

            # plot intermediate results to tensorboard
            if self.writer is not None and k % plot_every == 2:
                self.plot_intermediate_to_writer(store_intermediate, 'unmixing'+run_id, k)
                store_intermediate = self.reset_store(store_intermediate)

        # make sure contents of store are np arrays (for snep, software used to run simulations on the lab cluster)
        for key in store:
            store[key] = np.array(store[key])

        return store

    def test(self, signals, test_data, nt=1000, store_hidden=False, freeze_fb_after=None):
        """Test model performance on the signals and save signals, mixings, outputs, readout weights and targets."""

        # get info from inputs
        n_test = len(test_data)  # number of batches
        lent = signals.shape[0]    # number of samples in each context
        nsig = signals.shape[1]    # number of source signals

        # initialise weights and hidden states
        hidden_init = (torch.zeros(1, 1, self.model.hidden_dim),
                       torch.zeros(1, 1, self.model.hidden_dim))
        hidden_states = [hidden_init]
        if isinstance(self.model, DaleRateNet):
            m_init = torch.zeros((1, 1, self.model.Nm))
            p_init = torch.normal(0.5, 0.1, size=(1, 1)+self.model.w_dim)
            w_states = [(m_init, p_init)]
        else:
            w_init = torch.zeros((1, 1)+self.model.w_dim)
            w_states = [(w_init,)]

        # initialise dictionary for storing variables
        store = {'s': [], 'x': [], 'y': [], 'w': [], 'wt': []}
        if store_hidden:
            store.update({'hidden': []})
        if isinstance(self.model, SpatialNet):
            store.update({'z': [], 'm': []})
        if isinstance(self.model, DaleRateNet):
            store.update({'p': [], 'xb': []})

        # loop over mixing matrices, i.e. contexts
        count = 0
        print('Testing model...')
        for k, (A, A_inv) in enumerate(test_data):
            w_target = A_inv.reshape((1, 1, -1))

            # start timer, reset loss and gradients
            print(f"test mixing {k + 1}/{n_test}")
            start = time.time()
            loss_all = torch.zeros(1)

            # loop over data points in sequence (fixed mixing/context)
            for j in range(nt):  # change mixing every nt sampels
                s_ = signals[int(count % lent), :self.model.signal_dim]  # the signal is continuous across contexts
                count += 1

                # run forward model pass and compute loss
                s = torch.reshape(torch.from_numpy(s_), (1, 1, self.model.signal_dim))
                x = torch.einsum("bij, bsj -> bsi", A, s) \
                    + torch.normal(0, self.model.mix_noise, size=(1, 1, self.model.x_dim))
                w, y, hidden, net_out = self.model(x, w_states[-1], hidden_states[-1])

                # optional manipulation: freeze feedback
                if freeze_fb_after:
                    if k >= freeze_fb_after and j < 500:
                        if isinstance(self.model, SpatialNet) and not isinstance(self.model, DaleRateNet):
                            w = (w_states[-1][0], w[1])  # freeze modulation (i.e. block FB)
                        elif isinstance(self.model, DaleRateNet):
                            w = (w[0], w_states[-1][1], w[2], w[3])

                # compute loss
                loss = self.loss_function(s, y)
                loss_all += loss
                w_states.append(w)
                hidden_states.append(hidden)

                # store results
                store = self.update_store(store, s, y, w, w_target)
                store['x'].append(self.numpify(x))
                if isinstance(self.model, SpatialNet) and not isinstance(self.model, DaleRateNet):
                    store['m'].append(self.numpify(w[0]))
                    store['z'].append(self.numpify(w[1]))
                if isinstance(self.model, DaleRateNet):  # isinstance(self.model, PreInhNet) or
                    store['m'].append(self.numpify(w[0]))
                    store['p'].append(self.numpify(w[1]))
                    store['xb'].append(self.numpify(w[2]))
                    store['z'].append(self.numpify(w[3]))
                if store_hidden:
                    store['hidden'].append(self.numpify(hidden[0]))

            # forget old hidden states and weights (i.e. free memory)
            hidden_states = [(hidden_states[-1][0], hidden_states[-1][1])]
            w_states = [tuple([w_i for w_i in w_states[-1]])]
            print(f"\t loss: {loss_all.item():3.3f}, took {time.time() - start:1.1f}s")

        # make sure contents of store are np arrays (for snep, software used to run simulations on the lab cluster)
        for key in store:
            store[key] = np.array(store[key])

        return store


def revive_model(params, trained_weights, state=None):
    """ Function to 'revive' a pre-trained model from saved parameters. This function relies on certain contents of
        the parameter dictionaries 'params' and 'state'.

        params: dictionary containing model parameters, such as 'model_class', 's_dim', 'n_hid', 'Nz'
                and the boolean variables 'learn_baseline' and 'learn_ro' (see
        trained_weights: ordered dictionary of the named model parameters (from model.named_parameters())
        state:  dictionary containing information of the state of the network at the end of last training,
                depending on the model type and which weights are learned, it contains 'w0' (baseline weights),
                'Wro' (readout weights) and 'w_inh' (effective FF inhibitory weights). See model classes for details.
        """

    # first compute number of outputs necessary from LSTM (depends on model variant)
    n_out = params['Nz'] if params['model_class'] in ['DaleRateNet', 'SpatialNet'] else params['s_dim']**2

    # create model from parameters
    model = eval(params['model_class'])(params['s_dim'], params['n_hid'], n_out, **params)

    # fill model with trained parameters
    for p in trained_weights.keys():
        trained_weights[p] = torch.from_numpy(trained_weights[p])
    model.load_state_dict(trained_weights)

    # if readout weights are provided by a the state dictionary, write stored weights into model, if necessary
    if state is not None:
        if not params['learn_baseline']:
            model.w0 = torch.from_numpy(state['w0'])
        if not params['learn_ro']:
            model.Wro = torch.from_numpy(state['Wro'])
        if isinstance(model, DaleRateNet):
            model.w_inh = torch.from_numpy(state['w_inh'])

    return model


if __name__ in "__main__":

    # Here you can run the models defined above.

    # plt.style.use('pretty')  # Note: this is a custom stylesheet, will fail if stylesheet with this name doesn't exist

    #####################
    # Define parameters #
    #####################

    # training parameters
    BATCH_SIZE = 32  # number of trials (=different contexts) per batch
    N_MIX = 20*BATCH_SIZE  # total number of mixings (=contexts). Note: training the model(s) takes minimum 3000 batches
    N_HID = 100  # number of hidden units in LSTM
    N_TEST = 4  # number of contexts for testing
    LR = 0.001  # learning rate for optimisation
    lambda_reg = 0  # regularisation strength, use for DaleRateNet with value: 10e-6
    mix_type = 'random_norm'  # type of generated mixing matrices
    n_sample = 1000  # number of samples in one context
    writer_on = False  # whether to write updates to tensorboard

    # model paremeters
    MODEL_CLASS = UnmixingModel  # model type; options: UnmixingModel, ModulationNet, SpatialNet, DaleRateNet
    net_input = 'xy'  # input to the LSTM; options: 'x' (only stimuli), 'y' (only network output), 'xy' (both)
    add = 0  # whether to use additive modulation, default is multiplicative (note: not possible for all variants)
    tau = 100  # timescale of modulation
    learn_baseline = False  # whether to optimise baseline weights (W0)
    learn_wx = False  # whether to optimise x -> z_L weights
    learn_ro = False  # whether to optimise readout weights
    mod_target = 0  # target of modulation; options: 1 (exc FF weights), 2 (inh FF weights), 0 (both) (DaleRateNet only)

    # dimensions of populations
    signal_dim = 2  # number of sources, per default same as number of sensory stimuli and outputs
    # populations sizes (only relevant for SpatialNet and/or DaleRateNet)
    Nz = 100  # number of neurons in middle layer
    NzL = 40  # number of neurons in lower level population
    Nm = 4  # number of modulation units

    #################
    # Generate data #
    #################

    # generate sources
    import data_maker as dm
    inputs = dm.gen_chords(2, base_freq=100)  # alternative signals commented out below
    # inputs = dm.gen_sines(dur=2, freqs_rel=[1, 1.4])
    # inputs = dm.gen_ica_style(dur=2, freqs_rel=[1.4, 0.8, 1.2])

    # generate contexts (mixings)
    data_train, data_test = dm.gen_train_test_data(N_MIX, N_TEST, mix_type=mix_type, batch_size=BATCH_SIZE,
                                                   mat_size=(signal_dim, signal_dim))

    ######################
    # Create & run model #
    ######################

    # Define model, loss, optimizer, runner, tensorboard writer
    model = MODEL_CLASS(signal_dim, N_HID, signal_dim**2, tau=tau, input_to_net=net_input, add=add,
                        w0_var=0.001, mix_noise=0.001, learn_baseline=learn_baseline, Nz=Nz, Nm=Nm, NzL=40,
                        mod_width=0.2,
                        learn_ro=learn_ro, learn_wx=learn_wx, inh_frac=0.5, mod_target=mod_target)
    loss_function = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    writer = tb.SummaryWriter('../logs/unmix_integral') if writer_on else None  # may require creating this directory)
    runner = Runner(model, loss_function, optimizer, writer, batch_size=BATCH_SIZE)

    # train
    results_train = runner.train(inputs, data_train, nt=n_sample, plot_every=500, log_every=50, lambda_reg=lambda_reg)

    # test
    results_test = runner.test(inputs, data_test, store_hidden=True)

    # if writer is used, close after training & testing model
    if writer_on:
        writer.close()

    ############
    # Plotting #
    ############

    # plot loss
    fig1, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=150)
    ax1.semilogy(results_train['loss'])
    ax1.set(xlabel='batches', ylabel='loss')
    plt.tight_layout()

    fig, ax2 = plt.subplots(4, 1, figsize=(5, 4), dpi=150)
    ax2[0].plot(results_test['wt'], 'k--')
    ax2[0].plot(results_test['w'])
    ax2[1].plot(results_test['x'], 'gray', lw=1)
    ax2[2].plot(results_test['y'][:, 0], 'C0--')
    ax2[2].plot(results_test['s'][:, 0], 'C0', alpha=0.5)
    ax2[3].plot(results_test['y'][:, 1], 'C2--')
    ax2[3].plot(results_test['s'][:, 1], 'C2', alpha=0.5)

    ax2[0].set(ylabel='weights')
    ax2[1].set(ylabel='stim x')
    ax2[2].set(ylabel=r'$y_1$/$s_1$')
    ax2[3].set(xlabel='time (samples x mixings)', ylabel=r'$y_2$/$s_2$')
    plt.tight_layout()

    # if SpatialNet: plot modulation units and middle layer z activity
    if isinstance(model, SpatialNet) and not isinstance(model, DaleRateNet):

        fig, ax = plt.subplots(2, 1, figsize=(5, 3), dpi=150)
        ax[0].plot(results_test['m'])  # modulation unit activity
        for ii in range(Nz):
            ax[1].plot(results_test['z'][:, ii], lw=1)  # middle layer z activity
        ax[0].set(ylabel='mod units')
        ax[1].set(xlabel='samples x mixings', ylabel='z unit activity')
        plt.tight_layout()

    # if DaleRateNet: plot modulation units, gain modulation, lower and higher level population activity
    if isinstance(model, DaleRateNet):

        fig, ax = plt.subplots(4, 1, figsize=(4, 5), dpi=150, sharex=True)
        ax[0].plot(results_test['m'])  # modulation unit activity
        ax[1].pcolor(results_test['p'].T, lw=1, alpha=0.6, cmap='Greens_r', vmin=0, vmax=1)  # modulation (rel. prob.)
        ax[2].plot(results_test['xb'])  # note: xb is z_L, the lower-level population
        ax[3].plot(results_test['z'])   # and z is z_H, the higher-level population
        ax[0].set(ylabel='mod. units m')
        ax[1].set(ylabel='gain mod. p')
        ax[2].set(ylabel=r'$z_L$')
        ax[3].set(xlabel='time (samples x mixings)', ylabel=r'$z_H$')
        plt.tight_layout()

    plt.show()
