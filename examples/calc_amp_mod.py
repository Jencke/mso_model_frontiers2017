#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Simple Example that runs the MSO model for 400Hz with a 400 Hz
tone amplitude modulated at 2Hz and plots the firing rate difference.
'''

from __future__ import print_function, division

import numpy as np
import brian as br
import matplotlib.pyplot as plt
from brian.units import second
import cochlea as coch
import thorns

from  mso_model.mso_connect import inhibitory_to_mso, excitatory_to_mso, cochlea_to_gbc
from mso_model.mso import make_mso_group
from mso_model.helper import run
import mso_model.audiotools as audio

def run_exp(c_freq, sound_freq , itd, level=50, i=0):
    br.globalprefs.set_global_preferences(useweave=True, openmp=True, usecodegen=True,
                                          usecodegenweave=True )

    br.defaultclock.dt = 20E-6 * second

    #Basic Parameters
    fs_coch = 100e3 # s/second
    duration = 1800E-3 # seconds##
    pad = 120E-3 # second
    n_neuron = 300
    dt_coch = 1/fs_coch
    n_pad = int(pad/dt_coch)
    n_itd = int(itd/dt_coch)
    const_dt = 100e-6

    sound = audio.generate_tone(sound_freq, duration, fs_coch)
    mod = audio.cos_amp_modulator(sound, 2, fs_coch)
    sound *= mod
    sound = coch.set_dbspl(sound, 50)
    sound = sound * audio.cosine_fade_window(sound, 20e-3, fs_coch)
    sound = np.concatenate((np.zeros(n_pad), sound, np.zeros(n_pad)))

    sound = audio.delay_signal(sound, np.abs(itd), fs_coch)

    duration = len(sound) / fs_coch

    # construct ipsi and contra-lateral ANF trains and convert them to
    # neuron groups
    cochlea_train_left = coch.run_zilany2014(sound=sound[:, 0],
                                             fs=fs_coch,
                                             anf_num=(n_neuron, 0, 0),
                                             cf=c_freq,
                                             species='human',
                                             seed=None)

    cochlea_train_right = coch.run_zilany2014(sound=sound[:, 1],
                                             fs=fs_coch,
                                             anf_num=(n_neuron, 0, 0),
                                             cf=c_freq,
                                             species='human',
                                              seed=None)

    anf_group_left = coch.make_brian_group(cochlea_train_left)
    anf_group_right = coch.make_brian_group(cochlea_train_right)

    # Setup a new mso group and new gbc groups
    mso_group_left = make_mso_group(n_neuron)
    mso_group_right = make_mso_group(n_neuron)

    gbc_group_left = cochlea_to_gbc(anf_group_left, n_neuron)
    gbc_group_right = cochlea_to_gbc(anf_group_right, n_neuron)

    # Synaptic connections for the groups
    syn_mso_l_in  = inhibitory_to_mso(mso_group=mso_group_left,
                                      ipsi_group=gbc_group_left['neuron_groups'][0],
                                      contra_group=gbc_group_right['neuron_groups'][0],
                                      strength=1.5e-8,
                                      ipsi_delay=0,
                                      contra_delay=-0.6e-3 + const_dt)
    syn_mso_l_in_ipsi, syn_mso_l_in_contra = syn_mso_l_in

    syn_mso_r_in  = inhibitory_to_mso(mso_group=mso_group_right,
                                      ipsi_group=gbc_group_right['neuron_groups'][0],
                                      contra_group=gbc_group_left['neuron_groups'][0],
                                      strength=1.5e-8,
                                      ipsi_delay=0,
                                      contra_delay=-0.6e-3 + const_dt)
    syn_mso_r_in_ipsi, syn_mso_r_in_contra = syn_mso_r_in

    syn_mso_l_ex_ipsi, syn_mso_l_ex_contra = excitatory_to_mso(mso_group=mso_group_left,
                                                                ipsi_group=anf_group_left,
                                                                contra_group=anf_group_right,
                                                                strength=20e-9,
                                                                contra_delay = const_dt)

    syn_mso_r_ex_ipsi, syn_mso_r_ex_contra = excitatory_to_mso(mso_group=mso_group_right,
                                                                ipsi_group=anf_group_right,
                                                                contra_group=anf_group_left,
                                                                strength=20e-9,
                                                                contra_delay = const_dt)


    sp_mon_left = br.SpikeMonitor(mso_group_left, record=True)
    sp_mon_right = br.SpikeMonitor(mso_group_right, record=True)

    network = ([mso_group_left,
                mso_group_right,
                anf_group_left,
                anf_group_right,
                syn_mso_l_ex_ipsi,
                syn_mso_l_ex_contra,
                syn_mso_l_in_ipsi,
                syn_mso_l_in_contra,
                syn_mso_r_ex_ipsi,
                syn_mso_r_ex_contra,
                syn_mso_r_in_ipsi,
                syn_mso_r_in_contra,
                sp_mon_left,
                sp_mon_right]
               + gbc_group_left['neuron_groups']
               + gbc_group_right['neuron_groups'])

    run(duration , network)

    mso_train_left = thorns.make_trains(sp_mon_left)
    mso_train_left.duration = duration
    mso_train_right = thorns.make_trains(sp_mon_right)
    mso_train_right.duration = duration

    return {'spikes_left':mso_train_left,
            'spikes_right':mso_train_right,
            'anf_train_left':cochlea_train_left,
            'n_itd':n_itd}

if __name__ == '__main__':
    res = run_exp(c_freq=400, sound_freq=400, itd=200e-6)

    trim = lambda x: thorns.trim(x, start=0.125, stop=2.125)
    spikes_left = trim(res['spikes_left'])
    spikes_right = trim(res['spikes_right'])

    psth_l, bins = thorns.psth(spikes_left, 30e-3)
    psth_r, bins = thorns.psth(spikes_right, 30e-3)

    fig, ax = plt.subplots(1, 1)
    ax.plot(bins[:-1], psth_r - psth_l)
    ax.set_xlabel('time / s')
    ax.set_ylabel('$\Delta R$ / sps')
