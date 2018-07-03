#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2014  Jörg Encke
This file is part of auditory_brain.

auditory_brain is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

auditory_brain is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with auditory_brain.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function, division, absolute_import

__author__ = "Jörg Encke"

import brian as br
from brian.units import siemens, second
from brian import ms
from brian.library.synapses import alpha_synapse, biexp_synapse

import numpy as np
import random

import warnings

import auditory_brain as audit
import cochlea as coch


__all__ = ['binaural_cochlea',
           'cochlea_to_gbc',
           'gbc_to_mso']


def binaural_cochlea(sound, fs, c_freq, anf_num=(100, 0, 0), seed=None):
    """ Run two cochlea models where the sound is thee same apart from a
    predefined itd

    Parameters
    ----------
    sound : numpy.Array
        This contains two columns, the first for the ipsi the second
        for the contralateral ear.
    fs : int
        Sample frequency of the Sound in Hz
    c_freq : float
        center frequency of the auditory nerve fibres in Hz
    anf_num : tuple
        A tuple of lenght 3 giving the number of auditory nerve fibres.
        (HSR, MSR, LSR)
    seed : convertible to 32bit integer
        The seed for the random number generator

    Returns
    -------
    dict
        A dict containing the spikes and the neuron groups

        keys:
        - spikes: A list with the spike trains [ipsi spikes, contra spikes]
        - neuron_groups: A list with he brian neuron groups [ipsi group, contra group]

    """

    # Calculate Spiketrains for ipsi and contralateral sounds
    anf_i = coch.run_zilany2014(sound[:, 0],
                                fs,
                                anf_num=anf_num,
                                cf=(c_freq, c_freq, 1),
                                seed=seed,
                                powerlaw='approximate',
                                species='human',)

    anf_c = coch.run_zilany2014(sound[:, 1],
                                fs,
                                anf_num=anf_num,
                                cf=(c_freq, c_freq, 1),
                                seed=seed,
                                powerlaw='approximate',
                                species='human')

    # Create neuron groups.
    anf_group_c = audit.make_anf_group(anf_c)
    anf_group_i = audit.make_anf_group(anf_i)

    return {"spikes": [anf_i, anf_c],
            "neuron_groups": [anf_group_i, anf_group_c]}


def cochlea_to_gbc(anf_group, n_gbc):

    warnings.warn('please use the function provided by the gbc subpackge')
    gbc_group = audit.make_gbc_group(n_gbc)

    p_connect = 40 / n_gbc
    synapses_i = br.Connection(anf_group, gbc_group, 'ge_syn')
    synapses_i.connect_random(anf_group, gbc_group, p=p_connect,
                              fixed=True, weight=0.0047561806622803005 * 1e-6 * siemens)

    return {'neuron_groups': [gbc_group, synapses_i]}


def inhibitory_to_mso(mso_group,
                      ipsi_group=None,
                      contra_group=None,
                      strength=0,
                      num_i=3,
                      tau_i1=0.14e-3,
                      tau_i2=1.6e-3,
                      ipsi_delay=0,
                      contra_delay=0):
    """Connect inhibitory neurons to MSO neurons.

    This establishes the inhibitory connections to MSO neurons via
    biexponential synapses. num_i neurons from both the ipsi-lateral and
    contra-lateral group are connected to one MSO neuron. The neurons are
    picked randomly.

    Parameters
    ----------
    mso_group : brian.NeuronGroup
        The MSO neuron group
    ipsi_group : brian.NeuronGroup
        The ipsi-lateral neuron group. No group is connected if set to
        None. (default = None)
    contra_group : brian.NeuronGroup
        The contra-lateral neuron group. No group is connected if set to
        None. (default = None)
    strength : float
        The synaptic strength in Simens. (default=None)
    num_i : int
        The number of connections from the ipsi- and contra- lateral
        side to on neuron (default = 3)
    tau_i1 : float
        First time constant of the biexponential in seconds
        (default = 0.14 ms)
    tau_i2 : float
        Second time constant in seconds of the biexpontential
        (default = 1.6 ms)
    ipsi_delay : float
        Delay of all ipsi-lateral inputs in seconds (default = 0)
    contra_delay : float
        Delay of all contra-lateral inputs in seconds (default = 0)

    Returns
    -------
    A list containing the two brian synapse groups

    [ipsi_lateral, contra_lateral]
    """

    # All synaptic inputs are delayed by 1.5ms so that negative delays
    # are possible
    all_delay = 1.5e-3
    ipsi_delay += all_delay
    contra_delay += all_delay

    eqs = biexp_synapse(input='y',
                        tau1=tau_i1 * second,
                        tau2=tau_i2 * second,
                        unit=1,
                        output='gate')

    if ipsi_group:
        ipsi_synapse = br.Synapses(ipsi_group,
                                   mso_group,
                                   [eqs],
                                   pre='y += strength',
                                   freeze=True)
        mso_group.in_syn_i = ipsi_synapse.gate

        # Randomly connect the Neurons
        s_list = range(len(ipsi_group))
        for i in range(len(mso_group)):
            for j in range(num_i):
                c = random.choice(s_list)
                ipsi_synapse[c, i] = True
        # Set the delay for all synapses
        ipsi_synapse.delay[:] += ipsi_delay * second
    else:
        ipsi_synapse = None

    if contra_group:
        contra_synapse = br.Synapses(contra_group,
                                     mso_group,
                                     [eqs],
                                     pre='y += strength',
                                     freeze=True)
        mso_group.in_syn_c = contra_synapse.gate

        # Randomly connect the Neurons
        s_list = range(len(contra_group))
        for i in range(len(mso_group)):
            for j in range(num_i):
                c = random.choice(s_list)
                contra_synapse[c, i] = True
        contra_synapse.delay[:] += contra_delay * second
    else:
        contra_synapse = None

    return [ipsi_synapse, contra_synapse]

def excitatory_to_mso(mso_group,
                      ipsi_group=None,
                      contra_group=None,
                      strength=0,
                      num_e=6,
                      tau_e=.17e-3,
                      ipsi_delay=0,
                      contra_delay=0):
    """Connect excitatory neurons to MSO neurons.

    This establishes the excitatory connections to MSO neurons via
    biexponential synapses. num_i neurons from both the ipsi-lateral and
    contra-lateral group are connected to one MSO neuron. The neurons are
    picked randomly.

    Parameters
    ----------
    mso_group : brian.NeuronGroup
        The MSO neuron group
    ipsi_group : brian.NeuronGroup
        The ipsi-lateral neuron group. No group is connected if set to
        None. (default = None)
    contra_group : brian.NeuronGroup
        The contra-lateral neuron group. No group is connected if set to
        None. (default = None)
    strength : float
        The synaptic strength in Simens. (default=None)
    num_e : int
        The number of connections from the ipsi- and contra- lateral
        side to on neuron (default = 3)
    tau_e : int
        Time constant of the biexponential in seconds. (default = 0.17 ms)
    ipsi_delay : float
        Delay of all ipsi-lateral inputs in seconds (default = 0)
    contra_delay : float
        Delay of all contra-lateral inputs in seconds (default = 0)

    Returns
    -------
    A list containing the two brian synapse groups

    [ipsi_lateral, contra_lateral]
    """

    # All synaptic inputs are delayed by 1.5ms so that negative delays
    # are possible
    all_delay = 1.5e-3
    ipsi_delay += all_delay
    contra_delay += all_delay

    eqs = alpha_synapse(input='y', tau=tau_e * second, unit=1, output='gate')

    # Strength has to pe multiplied with e so that we gain correct
    # amplitudes!
    strength *= np.exp(1)

    # --> MSO (Excitatory)
    if ipsi_group:
        ipsi_synapse = br.Synapses(ipsi_group,
                                   mso_group,
                                   [eqs],
                                   pre='y += strength',
                                   freeze=True)
        mso_group.ex_syn_i = ipsi_synapse.gate

        # Randomly connect the Neurons
        s_list = range(len(ipsi_group))
        for i in range(len(mso_group)):
            for j in range(num_e):
                c = random.choice(s_list)
                ipsi_synapse[c, i] = True
        ipsi_synapse.delay[:] += ipsi_delay * second
    else:
        ipsi_synapse = None

    if contra_group:
        contra_synapse = br.Synapses(contra_group,
                                     mso_group,
                                     [eqs],
                                     pre='y += strength',
                                     freeze=True)
        mso_group.ex_syn_c = contra_synapse.gate

        # Randomly connect the Neurons
        s_list = range(len(contra_group))
        for i in range(len(mso_group)):
            for j in range(num_e):
                c = random.choice(s_list)
                contra_synapse[c, i] = True
        contra_synapse.delay[:] += contra_delay * second
    else:
        contra_synapse = None

    return [ipsi_synapse, contra_synapse]

def connect_random(source_group, target_group,  synapse):
    """randomly connect two neuron groups with a synapse

    Parameters
    ----------
    source_group : brian.NeuronGroup
        The source group.
    target_group : brian.NeuronGroup
        The target group.
    synapse:
        The synapse.

    Returns
    -------
        None
    """
    s_list = range(len(source))
    for i in range(len(target)):
        for j in range(num_e):
            c = random.choice(s_list)
            synapse[c, i] = True


__literature__ = """
[Couchman2010] Couchman, K., Grothe, B., & Felmy, F. (2010). Medial superior
olivary neurons receive surprisingly few excitatory and inhibitory
inputs with balanced strength and short-term dynamics. The Journal of
Neuroscience : The Official Journal of the Society for Neuroscience,
30(50), 17111–21. doi:10.1523/JNEUROSCI.1760-10.2010

"""
