import numpy

import DAQ


# Settings

filename = 'Data/16V_Constant.npy'

duration = 12.
delay = 5.

start = 1.000E-3
stop  = 2.000E-3
step  = 0.100E-3


# Code

daq = DAQ.DAQ()

data = []

for v in numpy.linspace(start, stop, int(1 + numpy.round((stop - start) / step))):
    print('Throttle: {:.3f} ms...'.format(v * 1000.))
    
    probe = daq.probe(v, duration, delay)
    data.append(probe)

daq.setThrottle(0.950E-3)

del daq

numpy.save(filename, numpy.array(data))