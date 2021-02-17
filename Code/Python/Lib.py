import copy
import numpy
import numpy.fft as fft
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate


def findRotations(f, threshold = 0.25, dilation = 0.01 / 7., samplingRate = 200_000.):
    div = samplingRate / 2.

    b, a = signal.bessel(1, 100. / div, 'highpass')
    f = copy.deepcopy(signal.filtfilt(b, a, f))

    a = [None] * 5
    b = [None] * 5

    div = samplingRate / 2.
    
    b[0], a[0] = signal.bessel(3,   312.5 / div, 'lowpass')
    b[1], a[1] = signal.bessel(3,   625.  / div, 'lowpass')
    b[2], a[2] = signal.bessel(3, 1_250.  / div, 'lowpass')
    b[3], a[3] = signal.bessel(3, 2_500.  / div, 'lowpass')
    b[4], a[4] = signal.bessel(3, 5_000.  / div, 'lowpass')

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
    stop  = numpy.argmin(frequency < 1_500) + 1

    frequency = frequency[start : stop]
    spectrum = spectrum[start : stop, :]
    
    t = numpy.arange(f.size) / samplingRate

    mainFrequency = frequency[numpy.argmax(spectrum, axis = 0)]
    mainFrequency = interpolate.interp1d(time, mainFrequency, fill_value = 'extrapolate')(t)

    harmonics = 3.

    f1 = numpy.clip((mainFrequency -   312.5 / harmonics) /   (312.5 / harmonics), 0, 1)
    f2 = numpy.clip((mainFrequency -   625.  / harmonics) /   (625.  / harmonics), 0, 1)
    f3 = numpy.clip((mainFrequency - 1_250.  / harmonics) / (1_250.  / harmonics), 0, 1)
    f4 = numpy.clip((mainFrequency - 2_500.  / harmonics) / (2_500.  / harmonics), 0, 1)

    filtered = band[0] + f1 * band[1] + f2 * band[2] + f3 * band[3] + f4 * band[4]

    rotation = numpy.where(numpy.diff((filtered > 0).astype(int)) == -1)[0]

    keep = numpy.abs(filtered) > threshold
    keep = ndimage.binary_dilation(keep, iterations = int(dilation * samplingRate))
    rotation = rotation[keep[rotation] == 1]

    # Interpolate to find zero crossing to improve precision

    return rotation / samplingRate


def orderTracking(rotation, acceleration, samplingRate = 200_000, divisions = 2048, expansionFactor = 1.):
    if rotation.size <= 2:
        return numpy.array([])
    else:
        if expansionFactor != 1:
            expanded = signal.resample(acceleration, acceleration.size * expansionFactor)
        else:
            expanded = acceleration

        expandedSamplingRate = samplingRate * expansionFactor
        
        fractionalRotation = numpy.arange((rotation.size - 1) * divisions) / divisions

        index = numpy.arange(0, rotation.size)
        x = index
        y = rotation[index]

        t = interpolate.interp1d(x, y, kind = 'quadratic')(fractionalRotation)

        x = numpy.arange(0, expanded.size) / expandedSamplingRate
        y = expanded
       
        t = numpy.clip(t, x[0], x[-1])
        
        orderAcceleration = interpolate.interp1d(x, y)(t)

        return orderAcceleration

