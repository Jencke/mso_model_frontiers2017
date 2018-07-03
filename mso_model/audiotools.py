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

import numpy as np

def generate_tone(frequency, duration, fs, start_phase=0, endpoint=False):
    '''Sine tone with a given frequency, duration and sampling rate.

    This function will generate a pure tone with of a given duration
    at a given sampling rate. By default, the first sample will be
    evaluated at 0 and the duration will be the real stimulus duration.

    if endpoint is set to True, the duration will be considerd
    as the maxium point in time so that the function calculates one
    more sample. The stimulus duration is now duration + (1. / fs)

    Parameters:
    -----------
    frequency : scalar
        The tone frequency in Hz.
    duration : scalar
        The tone duration in seconds.
    fs : scalar
        The sampling rate for the tone.
    start_phase : scalar, optional
        The starting phase of the sine tone.
    endpoint : bool, optional
        Whether to generate an additional sample so that
        duration = time at last sample.

    Returns:
    --------
    ndarray : The sine tone

    '''
    len_signal = int(np.round(duration * fs))
    len_signal += 1 if endpoint else 0
    time = np.linspace(0, duration, len_signal, endpoint)
    tone = np.sin(2 * np.pi * frequency * time + start_phase)
    return tone

def cos_amp_modulator(signal, modulator_freq, fs, mod_index=1):
    '''Cosinus amplitude modulator

    Returns a cosinus amplitude modulator following the equation:
    ..math:: 1 + m * \cos{2 * \pi * f_m * t}

    where m is the modulation depth, f_m is the modualtion frequency
    and t is the

    Parameters:
    -----------
    signal : ndarray
        An input array that is used to determine the length of the
        modulator.

    modulator_freq : float
        The frequency of the cosine modulator.

    fs : float
        The sample frequency of the input signal.

    mod_index: float, optional
        The modulation index. (Default = 1)

    Returns:
    --------
    ndarray : The modulator

    '''

    if isinstance(signal, np.ndarray):
        time = get_time(signal, fs)
    elif isinstance(signal, int):
        time = get_time(np.zeros(signal), fs)
    else:
        raise TypeError("Signal must be numpy ndarray or int")

    modulator = 1 + mod_index * np.cos(2 * np.pi * modulator_freq * time)

    return modulator


def cosine_fade_window(signal, rise_time, fs, n_zeros=0):
    '''Cosine fade-in and fade-out window.

    This function generates a window function with a cosine fade in
    and fade out.

    Parameters:
    -----------
    signal: ndarray
        The length of the array will be used to determin the window length.
    rise_time : scalar
        Duration of the cosine fade in and fade out in seconds. The number of samples
        is determined via rounding to the nearest integer value.
    fs : scalar
        The sampling rate in Hz
    n_zeros : int, optional
        Number of zeros to add at the end and at the beginning of the window. (Default = 0)

    Returns:
    --------
    ndarray : The fading window

    '''

    assert isinstance(n_zeros, int)

    r = int(np.round(rise_time * fs))
    window = np.ones(len(signal) - 2 * n_zeros)
    flank = 0.5 * (1 + np.cos(np.pi / r * (np.arange(r) - r)))
    window[:r] = flank
    window[-r:] = flank[::-1]

    window = zero_buffer(window, n_zeros)
    return window


def delay_signal(signal, delay, fs):
    '''Delay by phase shifting in the frequncy domain.

    This function delays a given signal in the frequncy domain
    allowing for subsample time shifts.

    Parameters
    ----------
    signal : ndarray
        The signal to shift
    delay : scalar
        The delay in seconds
    fs :  scalar
        The signals sampling rate in Hz

    Returns
    -------
     ndarray : A array of shape [N, 2] where N is the length of the
         input signal. [:, 0] is the 0 padded original signal, [:, 1]
         the delayed signal

    '''

    #Only Positive Delays allowed
    assert delay >= 0

    # save the original length of the signal
    len_sig = len(signal)

    #due to the cyclic nature of the shift, pad the signal with
    #enough zeros
    n_pad = np.int(np.ceil(np.abs(delay * fs)))
    pad = np.zeros(n_pad)
    signal = np.concatenate([pad, signal, pad])

    #Apply FFT
    signal = pad_for_fft(signal)
    ft_signal = np.fft.fft(signal)

    #Calculate the phases need for shifting and apply them to the
    #spectrum
    freqs = np.fft.fftfreq(len(ft_signal), 1. / fs)
    ft_signal *= np.exp(-1j * 2 * np.pi * delay * freqs)

    #Inverse transform the spectrum and leave away the imag. part if
    #it is really small
    shifted_signal = np.fft.ifft(ft_signal)
    shifted_signal = np.real_if_close(shifted_signal, 1000)

    both = np.column_stack([signal, shifted_signal])

    # cut away the buffering
    both = both[n_pad:len_sig + 2 * n_pad, :]
    return both


def pad_for_fft(signal):
    '''Zero buffer a signal with zeros so that it reaches the next closest
       :math`$2^n$` length.

       This Function attaches zeros to a signal to adjust the length of the signal
       to a multiple of 2 for efficent FFT calculation.

       Parameters:
       -----------
       signal : ndarray
           The input signal

       Returns:
       --------
       ndarray : The zero bufferd output signal.

    '''
    exponent = np.ceil(np.log2(len(signal)))
    n_out = 2**exponent
    out_signal = np.zeros(int(n_out))
    out_signal[:len(signal)] = signal
    return out_signal

def get_time(signal, fs):
    '''Time axis of a given signal.

    This function generates a time axis for a given signal at a given
    sample rate.

    Parameters:
    -----------
    signal : ndarray
        The input signal for which to generate the time axis
    fs : scalar
        The sampling rate in Hz

    Returns:
    --------
    ndarray : The time axis in seconds

    '''
    dt = 1. / fs
    max_time = len(signal) * dt
    time = np.arange(0, max_time , dt)

    # Sometimes, due to numerics arange creates an extra sample which
    # needs to be removed
    if len(time) == len(signal) + 1:
        time = time[:-1]
    return time

def zero_buffer(signal, number):
    '''Add a number of zeros to both sides of a signal

    Parameters:
    -----------
    signal: ndarray
        The input Signal
    number : int
        The number of zeros that should be added

    Returns:
    --------
    ndarray : The bufferd signal

    '''
    assert isinstance(number, int)

    buf = np.zeros(number)
    signal = np.concatenate([buf, signal, buf])

    return signal
