import brian
from brian import (
    second,
    defaultclock
    )

def set_fs(fs):
    brian.defaultclock.dt = (1/fs) * second

def reset_defaultclock():
    brian.defaultclock.t = 0 * second

reset = reset_defaultclock

def run(duration, objects, **kwargs):
    """Run a brian simulation

    This is a convenience function to easily
    run a brian network while also offering a flexible
    way of introducing Monitors.

    Parameters
    ----------
    duration : float
        Duration of the simulation in seconds.

    objects : list
        A collection of Brian objects to be simulated.

    export_dict : dict
        A dictionary of state trace to export. The Key gives
        the neuron group to read the state from and the value
        is a list of strings naming the state variables.
        {neuron1:['v', 'n']} for example would create state
        monitors for the variables v and n of neuron1

    **kwargs :
       Further kwargs ar passed to the brian network run function

    Returns
    -------
    If export_dict is given as a kwarg, The function this dictionary
    where the keys has been replaced by a pandas.DataFrame that contains
    the time and value treaces of the requested variables.

    """

    import pandas

    # Reset brian defaultclock
    brian.defaultclock.t = 0 * second


    # If export_dict is given, create a number
    # of State Monitors for the requested variables
    monitor_objects = []
    monitor_dict = {}
    if "export_dict" in kwargs:
        export_dict = kwargs.pop("export_dict")

        for o,l in export_dict.iteritems():
            monitor_dict[o] = []
            for i in l:
                monitor = brian.StateMonitor(o, i, record=True)
                monitor_dict[o].append(monitor)
                monitor_objects.append(monitor)

    net = brian.Network(objects + monitor_objects)

    kwargs.setdefault('report', 'text')
    kwargs.setdefault('report_period', 1)

    net.run(
        duration*second,
        **kwargs
    )

    for o,l in monitor_dict.iteritems():
        pds_dict = {}
        for i,m in enumerate(l):
            var_name = export_dict[o][i]
            data = m.values
            pds_dict[var_name] = list(data)
        pds_dict['time'] =  len(data) * [m.times]
        pds_frame = pandas.DataFrame(pds_dict)
        monitor_dict[o] = pds_frame


    return monitor_dict
