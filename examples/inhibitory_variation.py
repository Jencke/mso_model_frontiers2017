#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

__author__ = "Jörg Encke"

import numpy as np
import brian as br
from brian.units import second
import matplotlib.pyplot as plt
import cmath
from auditory_brain import *
from auditory_brain.connections import mso as connect
import cochlea as coch
from thorns import waves as wv
import auditory_brain as ab
import pandas
import thorns
import audiotools as audio

def run_exp(c_freq, itd, str_e, str_i):
    br.globalprefs.set_global_preferences(useweave=True, openmp=True, usecodegen=True,
                                          usecodegenweave=True )


    br.defaultclock.dt = 20E-6 * second

    #Basic Parameters
    fs_coch = 100e3 # s/second
    duration = 100E-3 # seconds##
    pad = 20E-3 # second
    n_neuron = 300
    dbspl = 50
    n_neuron = 500
    dt_coch = 1/fs_coch
    n_pad = int(pad/dt_coch)
    n_itd = int(itd/dt_coch)
    const_dt = 100e-6

    # Create the input Sound
    # duration = duration + 2 * pad
    # sound = thorns.waves.ramped_tone(fs_coch, c_freq, duration, ramp=20E-3)
    # sound = coch.set_dbspl(sound, dbspl)
    # sound = np.concatenate((np.zeros(n_pad), sound, np.zeros(n_pad)))
    # sound_c = np.roll(sound, n_itd)
    # sound = np.column_stack([sound, sound_c])
    # sound_t = thorns.waves.t(sound,fs_coch)


    sound = audio.audiotools.generate_tone(c_freq, duration, fs_coch)
    sound = sound * audio.audiotools.cosine_fade_window(sound, 20e-3, fs_coch)
    sound = coch.set_dbspl(sound, dbspl)
    sound = np.concatenate((np.zeros(n_pad), sound, np.zeros(n_pad)))
    sound = audio.audiotools.delay_signal(sound, np.abs(itd), fs_coch)


    if itd < 0:
        sound = sound[:, ::-1]
    duration = len(sound) / fs_coch

    # construct ipsi and contra-lateral ANF trains and convert them to
    # neuron groups
    cochlea_train_left = coch.run_zilany2014(sound=sound[:, 0],
                                             fs=fs_coch,
                                             anf_num=(n_neuron, 0, 0),
                                             cf=c_freq,
                                             species='human',
                                             seed=0)

    cochlea_train_right = coch.run_zilany2014(sound=sound[:, 1],
                                             fs=fs_coch,
                                             anf_num=(n_neuron, 0, 0),
                                             cf=c_freq,
                                             species='human',
                                             seed=0)

    anf_group_left = coch.make_brian_group(cochlea_train_left)
    anf_group_right = coch.make_brian_group(cochlea_train_right)


    # Setup a new mso group and new gbc groups
    mso_group_left = ab.make_mso_new_group(n_neuron)
    mso_group_right = ab.make_mso_new_group(n_neuron)

    gbc_group_left = ab.connections.mso_new.cochlea_to_gbc(anf_group_left, n_neuron)
    gbc_group_right = ab.connections.mso_new.cochlea_to_gbc(anf_group_right, n_neuron)

    # Synaptic connections for the groups
    syn_mso_l_in_ipsi, syn_mso_l_in_contra = ab.connections.mso_new.inhibitory_to_mso(mso_group=mso_group_left,
                                                                   ipsi_group=gbc_group_left['neuron_groups'][0],
                                                                   contra_group=gbc_group_right['neuron_groups'][0],
                                                                   strength=str_i,
                                                                   ipsi_delay=0,
                                                                   contra_delay=-0.6e-3 + const_dt)

    syn_mso_r_in_ipsi, syn_mso_r_in_contra = ab.connections.mso_new.inhibitory_to_mso(mso_group=mso_group_right,
                                                                   ipsi_group=gbc_group_right['neuron_groups'][0],
                                                                   contra_group=gbc_group_left['neuron_groups'][0],
                                                                   strength=str_i,
                                                                   ipsi_delay=0,
                                                                   contra_delay=-0.6e-3 + const_dt)


    syn_mso_l_ex_ipsi, syn_mso_l_ex_contra  = ab.connections.mso_new.excitatory_to_mso(mso_group=mso_group_left,
                                                                                   ipsi_group=anf_group_left,
                                                                                   contra_group=anf_group_right,
                                                                                       strength=str_e,
                                                                                       contra_delay=const_dt)

    syn_mso_r_ex_ipsi, syn_mso_r_ex_contra  = ab.connections.mso_new.excitatory_to_mso(mso_group=mso_group_right,
                                                                                   ipsi_group=anf_group_right,
                                                                                   contra_group=anf_group_left,
                                                                                       strength=str_e,
                                                                                       contra_delay=const_dt)

    # ex_syn_i_mon = br.StateMonitor(mso_group_left, 'ex_syn_i', record=True)
    # ex_syn_c_mon = br.StateMonitor(mso_group_left, 'ex_syn_c', record=True)
    # vu_mon = br.StateMonitor(mso_group_left, 'vu', record=True)
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

    ab.run(duration , network)

    mso_train_left = thorns.make_trains(sp_mon_left)
    mso_train_right = thorns.make_trains(sp_mon_right)

    return {'spikes_left':mso_train_left,
            'spikes_right':mso_train_right}


if __name__ == '__main__':
    params = {'c_freq':[125],
              'itd': np.linspace(-1e-3, 1e-3, 15),
              'str_e':[20e-9],
              'str_i':np.linspace(0e-9, 20e-9, 5)}#, 10e-9, 20e-9]}

    res = thorns.util.map(run_exp, params, backend='ipython', cache='no')

    thorns.util.dumpdb(res, '01_inhibitory_shift')
