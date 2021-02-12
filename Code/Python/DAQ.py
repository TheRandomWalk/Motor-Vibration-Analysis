import time
import numpy
import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.stream_readers as stream_readers
import nidaqmx.stream_writers as stream_writers


class DAQ:
    def __init__(self, bufferSize = 20_000):
        self.taskIn_  = nidaqmx.Task()
        self.taskIn_.ai_channels.add_ai_voltage_chan("MyDAQ1/ai0:1")
        self.taskIn_.timing.cfg_samp_clk_timing(rate = 200_000, sample_mode = constants.AcquisitionType.CONTINUOUS, samps_per_chan = bufferSize)
        
        self.analogMultiChannelReader_ = stream_readers.AnalogMultiChannelReader(self.taskIn_.in_stream)
        self.taskIn_.register_every_n_samples_acquired_into_buffer_event(bufferSize, self.callback)

        self.taskOut_ = nidaqmx.Task()
        self.taskOut_.co_channels.add_co_pulse_chan_time('myDAQ1/ctr0')
        self.taskOut_.timing.cfg_implicit_timing(sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)

        self.counterWriter_ = stream_writers.CounterWriter(self.taskOut_.out_stream, True)

        self.write_ = False
        
        self.counterWriter_.write_one_sample_pulse_time(.00095, .0015, timeout = nidaqmx.constants.WAIT_INFINITELY)
        time.sleep(3.0)


    def __del__(self): 
        self.taskIn_.close()
        self.taskOut_.close()

    def setThrottle(self, throttle):
        self.counterWriter_.write_one_sample_pulse_time(throttle, .0015, timeout = nidaqmx.constants.WAIT_INFINITELY)


    def probe(self, throttle, duration, delay = 3.):
        self.data_ = [[], []]
        
        throttle = numpy.clip(0.00095, 0.002, throttle)

        self.counterWriter_.write_one_sample_pulse_time(throttle, .0025 - throttle, timeout = nidaqmx.constants.WAIT_INFINITELY)
            
        self.taskIn_.start()
        time.sleep(delay)
        self.write_ = True

        while (len(self.data_[0]) < 200_000. * duration):
            time.sleep(.05)

        self.write_ = False
        self.taskIn_.stop()

        return numpy.array(self.data_)[:, -int(200_000 * duration) :].T


    def probeCurve(self, curve, delay = 3.):
        self.data_ = [[], []]
        
        curve = numpy.clip(0.001, 0.002, curve)
       
        self.taskIn_.start()
        time.sleep(delay)
        self.write_ = True

        while (len(self.data_[0]) == 0):
            time.sleep(.05)

        self.counterWriter_.write_many_sample_pulse_time(numpy.array(curve), .0025 - numpy.array(curve), nidaqmx.constants.WAIT_INFINITELY)

        self.write_ = False
        self.taskIn_.stop()

        return numpy.array(self.data_).T


    def callback(self, taskHandle, eventType, samples, callbackData):
        buffer = numpy.zeros((2, samples))
        self.analogMultiChannelReader_.read_many_sample(buffer, samples, timeout = constants.WAIT_INFINITELY)

        if self.write_:
            buffer = buffer.tolist()

            self.data_[0].extend(buffer[0])
            self.data_[1].extend(buffer[1])
        
        return 0



