import numpy

import DAQ


# Settings

filename = 'Data/16V_Step.npy'

curve = []

curve.append(numpy.zeros(50) + 0.950E-3)
curve.append(numpy.ones(100) + 2.000E-3)
curve.append(numpy.zeros(50) + 0.950E-3)

curve = numpy.concatenate(curve)


# Code

daq = DAQ.DAQ()

probe = daq.probeCurve(curve, 5)

del daq

print('Record: {:.1f} s'.format(probe.shape[0] / 200_000))

numpy.save(filename, probe)
