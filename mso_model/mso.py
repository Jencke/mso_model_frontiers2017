# -*- coding: utf-8 -*-
'''
Copyright (C) 2014-2018  Jörg Encke
This file is part of mso_model.

mso_model is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

auditory_brain is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with mso_model.  If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import division

import numpy as np

import brian
from brian import mV, pF, ms, siemens, nA, amp, nS, uohm, second, Mohm



def make_mso_group(num,
                   gbar_na = 3900,
                   gbar_klt = 650,
                   gbar_h = 520,
                   gbar_leak = 13,
                   e_rest = -55.8,
                   voltage_clamp=False):

    ''' Creates a sbc neuron group

    Parameters:
    -----------
    num : int
        Number of neurons in the neuron group.
    celsius : float
        Temperatur in degree celsius, Default = 37
    Output
    ------
    A brian NeuronGroup

    '''
    C = 70*pF # [Couchman2010]

    e_h = -35*mV # [Baumann2013]
    e_k = -90*mV

    e_e = -0 * mV #[Colbourne2005]
    e_i = -70 * mV #[Coulboure2005]


    e_na = 56.2*mV
    e_rest = e_rest*mV # Between -60 and -55

    nf = 1 # proportion of n vs p kinetics
    zss = 0.4 # steady state inactivation of glt


    gbar_na = gbar_na * nS
    gbar_klt = gbar_klt * nS
    gbar_h = gbar_h * nS
    gbar_leak = gbar_leak * nS

    if not voltage_clamp:
        eqs = """
        i_stim : amp
        dvm/dt = - (-i_stim + i_leak + i_na  + i_klt + i_h + i_syn) / C : volt
        vu = vm/mV : 1 # unitless v
        """

    else:
        eqs = """
        i_stim : amp
        vm : volt
        vu = vm/mV : 1 # unitless v
        """

    eqs_na = """
        g_na = gbar_na * m**3 * h : nsiemens
        i_na = g_na * (vm - e_na) : amp

        m_inf = 1 / (1 + exp((vu + 38) / -7)) :1
        h_inf = 1 / (1 + exp((vu + 65) / 6)) :1

        tau_m = (0.48 / (5 * exp((vu + 60) / 18) + 36 * exp((vu + 60) / -25)) ) * ms : ms
        tau_h = (19.23 / (7 * exp((vu + 60) / 11) + 10 * exp((vu + 60) / -25)) + 0.12) * ms : ms

        dm/dt = (m_inf - m) / tau_m :1
        dh/dt = (h_inf - h) / tau_h :1
        """
    eqs += eqs_na

    # [Baumann2013] (Dorsal Cells)
    eqs_h = """
        g_h = gbar_h * a  :nsiemens
        i_h =  g_h * (vm - e_h) : amp

        a_inf = 1 / (1 + exp(0.1 * (vu + 80.4))) :1
        tau_a = (79 + 417 * exp(-(vu + 61.5)**2 / 800)) *ms :ms

        da/dt = (a_inf - a) / tau_a :1
        """
    eqs += eqs_h

    # Potassium low threshold [Khurana2011]
    eqs_klt = """
        g_klt = gbar_klt * w**4 * z : nsiemens
        i_klt =  g_klt * (vm - e_k) : amp

        z_inf = zss + ((1-zss) / (1 + exp((vu + 57) / 5.44))) : 1
        w_inf = 1 / (1 + exp(-(vu + 57.3) / 11.7)) :1


        tau_w = 0.46 * (100. / (6. * exp((vu + 75.) / 12.15) + 24. * exp(-(vu + 75.) / 25) + 0.55)) * ms : ms
        tau_z = 0.24 * ((1000 / (exp((vu + 60) / 20) + exp(-(vu + 60) / 8))) + 50) * ms : ms

        dw/dt =( w_inf - w) / tau_w :1
        dz/dt = (z_inf - z) / tau_z :1
        """
    eqs += eqs_klt

    # leak
    eqs_leak = """
    g_leak = gbar_leak * 1 : nsiemens
    i_leak = g_leak * (vm - e_rest) : amp
    """
    eqs += eqs_leak

    #Synaptic Current
    eqs_syn = """
        ex_syn_i :1
        ex_syn_c :1
        in_syn_i :1
        in_syn_c :1

        i_syn = i_syn_ii + i_syn_ic + i_syn_ei + i_syn_ec : amp

        #inhibitory currents
        i_syn_ii = in_syn_i * siemens * (vm - e_i) : amp
        i_syn_ic = in_syn_c * siemens * (vm - e_i) : amp

        #exitatory currents
        i_syn_ei = ex_syn_i * siemens * (vm - e_e) : amp
        i_syn_ec = ex_syn_c * siemens * (vm - e_e) : amp
        """
    eqs += eqs_syn


    group = brian.NeuronGroup(
        N=num,
        model=eqs,
        threshold=brian.EmpiricalThreshold(threshold=-30*mV, refractory=0.5*ms),
        implicit=True,
    )

    ### Set initial conditions
    group.vm = e_rest
    group.m = group.m_inf
    group.h = group.h_inf
    group.w = group.w_inf
    group.z = group.z_inf
    group.a = group.a_inf

    return group
