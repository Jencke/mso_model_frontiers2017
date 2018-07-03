'''Plot the results of the inhibitory variation script

You need to previously run the inhibitory_variation.py
script so that the inhibitory_variation.h5 result file
is generated in the work subfolder

'''
import numpy as np
import matplotlib.pyplot as plt
import thorns
import scipy.optimize as opt


res = thorns.util.loaddb('inhibitory_variation')
res = res.reset_index()

calc_fr = lambda x: thorns.firing_rate(thorns.trim(x, start=0.045, stop=0.145))


def gaus(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0)**2 / (sigma**2)) + b#

res['rate_left'] = res.spikes_left.apply(calc_fr)
res['rate_right'] = res.spikes_right.apply(calc_fr)
sel_curve = res[res.str_i == res.str_i.unique()[-2]]

fig, ax = plt.subplots(1, 1)
for strength, sg in res.groupby('str_i'):
    ax.plot(sg.itd * 1e3, sg.rate_left)
ax.set_ylim(0, 90)
ax.set_ylabel('Firing rate / sps')
ax.set_xlabel('ITD / ms')
ax.set_yticks([0, 30, 60, 90])
ax.set_xlim(-1, 1)
