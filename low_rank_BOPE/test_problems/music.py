from typing import Optional

import numpy as np
import torch
from botorch.test_functions import SyntheticTestFunction
from torch import Tensor

# code credit to
# https://github.com/katieshiqihe/music_in_python


def get_combined_sine_wave(
    frequencies: Tensor, 
    duration: float, 
    sample_rate: Optional[int] = 44100, 
    amplitude: Optional[int] = 128
):
    """
    Get (combined) sine wave of specified frequencies.

    Args:
        frequencies: `n_frequencies x 1` tensor of frequencies in hertz
        duration: time of measurement in seconds
        sample_rate: number of measurements per second
        amplitude: peak amplitude
    Returns:
        `duration*sample_rate x 1` tensor of wave signals
    """

    waves = []
    t = np.linspace(0, duration, int(sample_rate*duration))
    for frequency in frequencies:
        waves.append(amplitude*np.sin(2*np.pi*frequency*t))

    return torch.tensor(np.sum(waves, axis=0), dtype=torch.double)


# outcome function simulating signals from pressing two keys in an octave
# NOTE: heat map of utility over [0,1]^2 would be cool


class HarmonyOneKey(SyntheticTestFunction):
    r"""
    Class for simulating playing two notes together in an octave,
    where the first note must be C. 
    """

    dim = 1
    _bounds = torch.tensor([[0., 1.]]) # press two keys together

    def __init__(self, base_key = 51, duration = 0.01): # key 52 = C5
        super().__init__()
        # self.outcome_bounds has to do with amplitude
        self.note_freqs = [2**((n+1-49)/12)*440 for n in range(88)]
        self.base_key = base_key
        self.base_freq = self.note_freqs[base_key]
        self.key_size = 1/12
        self.duration = duration
        
        
    def evaluate_true(self, X):
        
        keys = torch.div(X, self.key_size, rounding_mode="floor") + 1
        Y = []

        for sample_idx in range(X.shape[-2]):
            key = int(self.base_key + keys[sample_idx].item())
            freq = self.note_freqs[key]
            Y.append(get_combined_sine_wave([self.base_freq, freq], duration = self.duration))
        
        return torch.vstack(Y)


class HarmonyTwoKeys(SyntheticTestFunction):
    r"""
    Class for simulating playing two notes together in an octave.
    """

    dim = 2
    _bounds = torch.tensor([[0., 1.], [0., 1.]]) # press two keys together

    def __init__(self):
        super().__init__()
        # self.outcome_bounds has to do with amplitude
        pass
    
    def evaluate_true(self, X):
        pass

        # first map X to keys, discretize [0,1] to 88 intervals
        # get the signals for individual keys
        # then add them up

    

# utility function measuring consonance

class Consonance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, Y: Tensor):
        pass

        # NOTE: this is mapping from signal to utility, not from frequency to utility! 
        # how about doing a fourier decompsition here and design some function in the frequency domain

        """
        new idea from peter: compute similarity with known pleasant spectra

        concretely: get the signal vectors of all combinations, do FFT (get base spectra), 
        compute pleasantness based on Plompt and Levelt (to doublecheck)
        store it in a dictionary (?) mapping spectrum to pleasantness value
        
        then, given a new sound wave, the utility is computed as follows
        - first do FFT on the sound wave;
        - then compute the distance of the spectrum with the 12 (?) other spectra 
          (Euclidean? cosine similarity? probably need a normalizatio step?)
        - option 1: get the closest base spectrum, assign its pleasantness as the util value
        - option 2: compute the distance to all base spectra, 
          take a weighted average of their pleasantness values (weights = 1/distance or something)
        
        """





        