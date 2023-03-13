import numpy as np
import torch
from torch import Tensor

# code credit to
# https://github.com/katieshiqihe/music_in_python

def get_piano_notes():
    '''
    Get the frequency in hertz for all keys on a standard piano.
    Returns
    -------
    note_freqs : dict
        Mapping between note name and corresponding frequency.
    '''
    
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    base_freq = 440 #Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]
    
    note_freqs = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0 # stop
    return note_freqs

def get_sine_wave(
    frequency, 
    duration, 
    sample_rate=44100, 
    amplitude=4096
):
    '''
    Get pure sine wave. 
    Parameters
    ----------
    frequency : float
        Frequency in hertz.
    duration : float
        Time in seconds.
    sample_rate : int, optional
        Wav file sample rate. The default is 44100.
    amplitude : int, optional
        Peak Amplitude. The default is 4096.
    Returns
    -------
    wave : TYPE
        DESCRIPTION.
    '''
    t = np.linspace(0, duration, int(sample_rate*duration))
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave


# outcome function simulating signals from pressing two keys in an octave

class Harmony(SyntheticTestFunction):
    r"""
    Class for simulating playing two notes together
    """

    dim = 2
    # _bounds = # has to do with amplitude

    def __init__(self):
        super().__init__()
        pass
    
    def evaluate_true(self, X):
        pass

        # first map X to keys
        # get the signals for individual keys
        # then add them up

    

# utility function measuring consonance

class Consonance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, Y: Tensor):
        pass