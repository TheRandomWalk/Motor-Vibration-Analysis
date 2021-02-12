import numpy
import scipy.signal as signal
import matplotlib.pyplot as pyplot
import tqdm

import Lib


# Settings

input                      = 'Data/16V_Sweep.npy'
outputSpeed                = 'Data/Sweep_Speed.png'
outputSpectrogramFrequency = 'Data/Sweep_Spectrogram_Frequency.png'
outputSpectrogramOrder     = 'Data/Sweep_Spectrogram_Order.png'
outputSpectrumFrequency    = 'Data/Sweep_Spectrum_Frequency.png'
outputSpectrumOrder        = 'Data/Sweep_Spectrum_Order.png'

sampleRate  = 200_000.
sensitivity = 0.02          # V / (m / s^2)
ratio       = 7

divisions       = 2048
expansionFactor = 8


# Preprocessing

print('Preprocessing...')

data = numpy.load(input)

phase = data[:, 1]

tVibration = data[:, 0]
t          = numpy.arange(tVibration.size) / sampleRate

rotation   = Lib.findRotations(phase)

rVibration = Lib.orderTracking(rotation, data[:, 0], divisions = divisions / ratio, expansionFactor = expansionFactor)
r          = numpy.arange(rVibration.size) / divisions

print('Total rotations: {:,.0f}'.format(rotation.size / ratio))


# Rotational speed plot

print('Generating speed plot...')

time  = (rotation[1:] + rotation[:-1]) * .5 
speed = 1. / ((rotation[1:] - rotation[:-1]) * ratio)

pyplot.plot(time, speed)

pyplot.xlim(0, time.max())
pyplot.ylim(0, 200)

pyplot.xlabel('Time (s)')
pyplot.ylabel('Rotational speed (Hz)')

pyplot.gcf().set_size_inches(16, 4)
pyplot.savefig(outputSpeed, dpi = 300)
pyplot.clf()


# Frequency spectrum

print('Generating frequency spectrum...')

frequency, psd = signal.welch(tVibration, fs = sampleRate, nperseg = 1024 * 256, noverlap = 1024 * 2, detrend = 'constant')

slice = frequency <= 1000 + frequency[1]

frequency = frequency[slice]
psd = psd[slice]

pyplot.semilogy(frequency, psd)
pyplot.xlim(0, 1_000)

pyplot.xlabel('Frequency (Hz)')
pyplot.ylabel('Vibration ($m/s^2$ PSD)')

pyplot.gcf().set_size_inches(16, 4)
pyplot.savefig(outputSpectrumFrequency, dpi = 300)
pyplot.clf()


# Order spectrum

print('Generating order spectrum...')

order, psd = signal.welch(rVibration, fs = divisions, nperseg = 1024 * 512, noverlap = 1024 * 4, detrend = 'constant')

slice = order <= 5 + order[1]

order = order[slice]
psd = psd[slice]

pyplot.semilogy(order, psd)
pyplot.xlim(0, 5)

pyplot.ylabel('Vibration ($m/s^2$ PSD)')
pyplot.xlabel('Order')

pyplot.savefig(outputSpectrumOrder, dpi = 300)
pyplot.gcf().set_size_inches(16, 4)
pyplot.clf()


# Frequency spectrogram

print('Generating frequency spectrogram...', end = '')

frequency, time, spectrum = signal.spectrogram(tVibration, sampleRate, nperseg = 1024 * 64, detrend = 'constant')

slice = frequency <= 1000 + frequency[1]
frequency = frequency[slice]
spectrum = spectrum[slice, :]

spectrum = numpy.log10(spectrum)

spectrum -= spectrum.min()
spectrum /= spectrum.max()

print(' {:,d} x {:,d}'.format(spectrum.shape[1], spectrum.shape[0]))

pyplot.pcolormesh(time, frequency, spectrum, shading = 'gouraud', cmap = 'inferno')
pyplot.ylim(0, 1_000)

pyplot.xlabel('Time (s)')
pyplot.ylabel('Frequency (Hz)')

pyplot.gcf().set_size_inches(16, 9)
pyplot.savefig(outputSpectrogramFrequency, dpi = 300)
pyplot.clf()


# Order spectrogram

print('Generating order spectrogram...', end = '')

order, time, spectrum = signal.spectrogram(rVibration, divisions, nperseg = 1024 * 128, detrend = 'constant')

slice = order <= 5 + order[1]
order = order[slice]
spectrum = spectrum[slice, :]

spectrum = numpy.log10(spectrum)

spectrum -= spectrum.min()
spectrum /= spectrum.max()

print(' {:,d} x {:,d}'.format(spectrum.shape[1], spectrum.shape[0]))

pyplot.pcolormesh(time, order, spectrum, shading = 'gouraud', cmap = 'inferno')
pyplot.ylim(0, 5)

pyplot.xlabel('Rotation')
pyplot.ylabel('Order')

pyplot.gcf().set_size_inches(16, 9)
pyplot.savefig(outputSpectrogramOrder, dpi = 300)
pyplot.clf()

