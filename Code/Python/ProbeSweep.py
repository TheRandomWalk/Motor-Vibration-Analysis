import numpy

import DAQ


# Settings

filename = 'Data/16V_Sweep.npy'

start = 1.000E-3
stop  = 2.000E-3
steps = 35870


# Code

daq = DAQ.DAQ()

probe = daq.probeCurve(numpy.linspace(start, stop, steps), 0)

daq.setThrottle(0.950E-3)

del daq

print('Record: {:.3f} s'.format(probe.shape[0] / 200_000))

numpy.save(filename, probe)
