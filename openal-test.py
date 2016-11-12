#from openal import al, alc
import struct

import itertools
from pyglet.media.drivers.openal import lib_openal as al
from pyglet.media.drivers.openal import lib_alc as alc
import wave
import numpy as np
import ctypes


class WaveStream(object):
    def __init__(self, filename):
        wav = wave.open(filename)
        self.__wav = wav
        self.__channels = wav.getnchannels()
        self.__bit_rate = wav.getsampwidth() * 8
        self.__sample_rate = wav.getframerate()
        self.__num_frames = wav.getnframes()
        self.__current_frame = 0

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.__wav is not None:
            self.__wav.close()
            self.__wav = None

    @property
    def channels(self):
        return self.__channels

    @property
    def bit_rate(self):
        return self.__bit_rate

    @property
    def sample_rate(self):
        return self.__sample_rate

    @property
    def num_frames(self):
        return self.__num_frames

    @property
    def current_frame(self):
        return self.__current_frame

    @property
    def duration(self):
        return self.num_frames / float(self.sample_rate)

    @property
    def format(self):
        fm = {
            (1, 8): al.AL_FORMAT_MONO8,
            (2, 8): al.AL_FORMAT_STEREO8,
            (1, 16): al.AL_FORMAT_MONO16,
            (2, 16): al.AL_FORMAT_STEREO16,
        }
        return fm[(self.channels, self.bit_rate)]

    def read_frames(self, n=1):
        assert n > 0
        assert self.current_frame < self.num_frames
        return self.__wav.readframes(n)


def load_wav():
    wave_file = 'C:\Users\Markus\Downloads\CHVRCHES - Leave A Trace (Goldroom Remix).wav'
    wav = WaveStream(wave_file)
    data = wav.read_frames(wav.num_frames)
    return wav.sample_rate, wav.format, data


def main():
    rate, format, data = load_wav()
    sample_len = len(data)
    both_channels = data

    #samples = struct.unpack('%ih' % (sample_len/2), data)
    #left_channel = struct.pack('%ih' % (sample_len/4), *samples[0::2])
    #right_channel = struct.pack('%ih' % (sample_len/4), *samples[1::2])

    # let OpenAL select the audio device
    device = alc.alcOpenDevice(None)
    if not device:
        error = alc.alcGetError()
        # do something with the error, which is a ctypes value
        return -1

    # the context is (usually) unique per application
    context = alc.alcCreateContext(device, None)
    alc.alcMakeContextCurrent(context)

    # OpenAL source defines a sound source
    source = al.ALuint()
    al.alGenSources(1, source)
    #al.alSourcef(source, al.AL_PITCH, 1)
    #al.alSourcef(source, al.AL_GAIN, 1)
    al.alSource3f(source, al.AL_POSITION, 0, 0, 0)
    al.alSource3f(source, al.AL_VELOCITY, 0, 0, 0)
    al.alSourcei(source, al.AL_LOOPING, 0)

    error = alc.alcGetError(device)
    assert not error

    # audio buffer
    buffer = al.ALuint()
    al.alGenBuffers(1, buffer)
    #al.alBufferData(buffer, al.AL_FORMAT_MONO16, left_channel, sample_len/2, rate)
    al.alBufferData(buffer, format, both_channels, sample_len, rate)

    # binding the source to the buffer
    al.alSourceQueueBuffers(source, 1, buffer)

    # playing the source
    al.alSourcePlay(source)

    # wait for the source to finish
    print('Playing ...')
    state = al.ALint(0)
    while True:
        al.alGetSourcei(source, al.AL_SOURCE_STATE, state)
        if state.value != al.AL_PLAYING:
            break
    print('Done playing.')
    al.alSourcei(source, al.AL_BUFFER, 0)

    # clean up
    al.alDeleteBuffers(1, buffer)
    al.alDeleteSources(1, source)
    alc.alcDestroyContext(context)
    alc.alcCloseDevice(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
