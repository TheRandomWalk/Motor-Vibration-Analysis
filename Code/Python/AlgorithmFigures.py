import copy
import numpy
import numpy.fft as fft
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import matplotlib.pyplot as pyplot

import Lib


# Settings

inputSlow           = 'Data/16V_Sweep.npy'
inputFast           = 'Data/16V_Step.npy'
outputBase          = 'Data/Rotation_Base.png'
outputOverlay       = 'Data/Rotation_Overlay.png'
outputFast          = 'Data/Rotation_Acceleration.png'
outputTrackingTime  = 'Data/Tracking_Time.png'
outputTrackingOrder = 'Data/Tracking_Order.png'

samplingRate = 200_000
ratio        = 7
duration     = 25

divisions       = 2048
expansionFactor = 8


# Functions

def findRotation(f, threshold = 0.25, dilation = 0.01 / 7., samplingRate = 200_000.):
    div = samplingRate / 2.

    b, a = signal.bessel(1, 100. / div, 'highpass')
    f = copy.deepcopy(signal.filtfilt(b, a, f))

    a = [None] * 5
    b = [None] * 5

    div = samplingRate / 2.
    
    b[0], a[0] = signal.bessel(3,   375. / div, 'lowpass')
    b[1], a[1] = signal.bessel(3,   750. / div, 'lowpass')
    b[2], a[2] = signal.bessel(3, 1_500. / div, 'lowpass')
    b[3], a[3] = signal.bessel(3, 3_000. / div, 'lowpass')
    b[4], a[4] = signal.bessel(3, 6_000. / div, 'lowpass')

    signal0 = signal.filtfilt(b[0], a[0], f)
    signal1 = signal.filtfilt(b[1], a[1], f)
    signal2 = signal.filtfilt(b[2], a[2], f)
    signal3 = signal.filtfilt(b[3], a[3], f)
    signal4 = signal.filtfilt(b[4], a[4], f)

    band = []
    band.append(signal0)
    band.append(signal1 - signal0)
    band.append(signal2 - signal1)
    band.append(signal3 - signal2)
    band.append(signal4 - signal3)

    frequency, time, spectrum = signal.spectrogram(f, samplingRate, nperseg = 1024, detrend = 'constant')

    start = numpy.argmax(frequency > 100)
    stop  = numpy.argmin(frequency < 1500) + 1

    frequency = frequency[start : stop]
    spectrum = spectrum[start : stop, :]
    
    t = numpy.arange(f.size) / samplingRate

    mainFrequency = frequency[numpy.argmax(spectrum, axis = 0)]
    mainFrequency = interpolate.interp1d(time, mainFrequency, fill_value = 'extrapolate')(t)

    harmonics = 3.

    f1 = numpy.clip((mainFrequency -   375. / harmonics) /   (375. / harmonics), 0, 1)
    f2 = numpy.clip((mainFrequency -   750. / harmonics) /   (750. / harmonics), 0, 1)
    f3 = numpy.clip((mainFrequency - 1_500. / harmonics) / (1_500. / harmonics), 0, 1)
    f4 = numpy.clip((mainFrequency - 3_000. / harmonics) / (3_000. / harmonics), 0, 1)

    filtered = band[0] + f1 * band[1] + f2 * band[2] + f3 * band[3] + f4 * band[4]

    rotation = numpy.where(numpy.diff((filtered > 0).astype(int)) == -1)[0]

    keep = numpy.abs(filtered) > threshold
    keep = ndimage.binary_dilation(keep, iterations = int(dilation * samplingRate))
    rotation = rotation[keep[rotation] == 1]

    return rotation / samplingRate, filtered


# Preprocessing

print('Preprocessing...')

data = numpy.load(inputSlow)

v = data[:, 1]
t = numpy.arange(v.size) / samplingRate

zeros, filtered = findRotation(v)

frequency = 1 / ((zeros[1:] - zeros[:-1]))


# Base plot

print('Generating base plot...')

ticks = []

for i in range(7):
    f = 200 * (i + 1)
    idx = numpy.argmin(numpy.abs(frequency - f)) + 1

    pyplot.plot((t - zeros[idx]) * 1000. + duration / 2., 0.9 * v / v.max() + i + 0.05, 'C0', linewidth = .75)
    ticks.append('{:d}'.format(f))

ticks = numpy.array(ticks)

pyplot.xlim(0, 25)
pyplot.ylim(0, 7)

pyplot.xlabel('Time (ms)')
pyplot.ylabel('Frequency (Hz)')

pyplot.yticks(numpy.arange(ticks.size) + .5, ticks)

pyplot.gcf().set_size_inches(16, 4)
pyplot.savefig(outputBase, dpi = 300)
pyplot.clf()


# Overlayed plot

print('Generating overlay plot...')

for i in range(7):
    f = 200 * (i + 1)
    idx = numpy.argmin(numpy.abs(frequency - f)) + 1

    pyplot.plot((t - zeros[idx]) * 1000. + duration / 2., 0.9 * v / v.max() + i + 0.05, 'C0', linewidth = .75)
    pyplot.plot((t - zeros[idx]) * 1000. + duration / 2., 0.9 * filtered / v.max() + i + 0.55, 'C1', linewidth = 2)
    pyplot.plot((zeros - zeros[idx]) * 1000. + duration / 2., numpy.zeros(zeros.size) + i + 0.55, 'ok', markersize = 5)

pyplot.xlim(0, 25)
pyplot.ylim(0, 7)

pyplot.xlabel('Time (ms)')
pyplot.ylabel('Frequency (Hz)')

pyplot.yticks(numpy.arange(ticks.size) + .5, ticks)

pyplot.gcf().set_size_inches(16, 4)
pyplot.savefig(outputOverlay, dpi = 300)
pyplot.clf()


# Preprocessing

print('Preprocessing...')

data = numpy.load(inputFast)

window = [0.45, 0.55]

v = data[:, 1]
t = numpy.arange(v.size) / samplingRate

zeros, filtered = findRotation(v)


# Acceleration comparison plot

print('Generating acceleration comparison plot...')

slice = numpy.logical_and(t >= window[0], t <= window[1])

t = t[slice] - window[0]
v = v[slice]
zeros = zeros - window[0]
filtered = filtered[slice]

v -= v.min()
v /= v.max()

filtered = filtered / (numpy.abs(filtered).max() * 2)

pyplot.plot(t * 1000., v + 1, linewidth = .75)
pyplot.plot(t * 1000., filtered + .5, linewidth = 1)
pyplot.plot(zeros * 1000., numpy.zeros(zeros.size) + .5, 'ok', markersize = 5)

pyplot.xlim(0, (window[1] - window[0]) * 1000.)

pyplot.xlabel('Time (ms)')

pyplot.yticks([0.5, 1.5], ['Filtered', 'Unfiltered'], rotation = 'vertical')

pyplot.gcf().set_size_inches(16, 4)
pyplot.savefig(outputFast, dpi = 300)
pyplot.clf()


# Preprocessing (25 to 125 Hz acceleration,  100 ms, 7.5 rotations)

print('Preprocessing...')

t = numpy.arange(0., 0.15, 1. / 200_000.)
f = -numpy.sin((25. + 500. * t) * (2 * numpy.pi) * t)

rotation = (-25 + numpy.sqrt(625. + 2_000. * numpy.arange(0, 9))) / 1_000.
tracked = Lib.orderTracking(rotation, f, divisions = divisions, expansionFactor = expansionFactor)


# Order tracking time plot

print('Generating order tracking time plot...')

pyplot.plot(t * 1000., f)
pyplot.plot(rotation * 1000., numpy.zeros(rotation.size), 'ok')
pyplot.xlim(0, 100)
pyplot.xlabel('Time (ms)')
pyplot.ylabel('Acceleration ($m/s^2$)')

pyplot.gcf().set_size_inches(16, 2)
pyplot.gcf().subplots_adjust(bottom = 0.25)
pyplot.savefig(outputTrackingTime, dpi = 300)
pyplot.clf()


# Order tracking rotation plot

print('Generating order tracking rotation plot...')

pyplot.plot(numpy.arange(tracked.size) / 2048, tracked)
pyplot.plot(numpy.arange(8), numpy.zeros(8), 'ok')
pyplot.xlim(0, 7.5)
pyplot.xlabel('Rotation')
pyplot.ylabel('Acceleration ($m/s^2$)')

pyplot.gcf().set_size_inches(16, 2)
pyplot.gcf().subplots_adjust(bottom = 0.25)
pyplot.savefig(outputTrackingOrder, dpi = 300)
pyplot.clf()



