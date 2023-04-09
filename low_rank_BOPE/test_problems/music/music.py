from typing import Optional

import numpy as np
import torch
from botorch.test_functions import SyntheticTestFunction
from scipy.fft import fft, fftfreq
from torch import Tensor

# code credit to
# https://github.com/katieshiqihe/music_in_python

# dissonance measure from approximated Plomp-Levelt curve
# adapted from https://alpof.wordpress.com/2015/04/05/consonance-calculations-1/ 
DISSONANCE = {
    1: 0.6119546974596196,
    2: 0.4555781663778304,
    3: 0.31555474895490304, # minor third
    4: 0.33938492778443635, # major third
    5: 0.2336166616600329, # perfect fourth
    6: 0.40697871585987755,
    7: 0.08247001463897903, # major fifth
    8: 0.38310515863591793,
    9: 0.13115879976347125, # major sixth
    10: 0.2592949302193497,
    11: 0.34862700080473974,
    12: 0.005167515119296202 # octave
}
# TODO: think later if we want more (spectrum, dissonance) pairs


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


class HarmonyOneKey(SyntheticTestFunction):
    r"""
    Class for simulating playing two notes together in an octave,
    where the first note must be C. 
    """

    dim = 1
    _bounds = torch.tensor([[0., 1.]]) # press two keys together

    def __init__(
        self, 
        base_key = 48, # key 49 = A4
        sample_rate = 44100,
        duration = 0.01, 
        amplitude = 1
    ): 
        super().__init__()
        # self.outcome_bounds has to do with amplitude
        self.note_freqs = [2**((n+1-49)/12)*440 for n in range(88)]
        self.base_key = base_key
        self.base_freq = self.note_freqs[base_key]
        self.key_size = 1/12
        self.sample_rate = sample_rate
        self.duration = duration
        self.amplitude = amplitude
        
        self.outcome_dim = int(self.sample_rate * self.duration)
        
        
    def evaluate_true(self, X):
        
        keys = torch.div(X, self.key_size, rounding_mode="floor") + 1
        Y = []

        for sample_idx in range(X.shape[-2]):
            key = int(self.base_key + keys[sample_idx].item())
            freq = self.note_freqs[key]
            Y.append(
                get_combined_sine_wave(
                    frequencies=[self.base_freq, freq], 
                    duration=self.duration,
                    sample_rate=self.sample_rate,
                    amplitude=self.amplitude
                )
            )
        
        return torch.vstack(Y)

  
class Consonance(torch.nn.Module):
    def __init__(self, model_spectra: list, dissonance_vals: list, similarity_eps: float = 0.0001):
        super().__init__()
        self.model_spectra = model_spectra
        self.dissonance_vals = dissonance_vals
        self.similarity_eps = similarity_eps
    
    def forward(self, Y: Tensor):
        r"""
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
          take a weighted average of their pleasantness values (weights = 1/(distance+epsilon) or something)
        
        """

        res_all = []
        for y in Y:
            res_all.append(self.get_consonance(y))
        
        return torch.tensor(res_all).unsqueeze(1)


    def get_consonance(self, y: Tensor):

        # transform y to an array
        y_np = y.detach().numpy()
        yf = fft(y_np)
        yf = 2.0/len(y_np) * np.abs(yf[:len(self.model_spectra[0])])

        # compute a distance measure between yf and all the model spectra
        # cosine? L2?
        # TODO: think through, enable more options

        spectrum_similarity = []
        for spectrum in self.model_spectra:
            spectrum_distance = np.linalg.norm(yf-spectrum) # TODO: use vector operation later
            spectrum_similarity.append(1/(spectrum_distance+self.similarity_eps)) # TODO: come back to this
        spec_sim_sum = sum(spectrum_similarity)

        return np.dot(
            np.array(spectrum_similarity) / spec_sim_sum,
            1-np.array(list(self.dissonance_vals))
        )


def get_model_spectra(trunc_length: int):

    problem = HarmonyOneKey()
    test_X = torch.arange(0,1,1/12).unsqueeze(1)
    test_Y = problem(test_X)

    model_spectra = []

    for i in range(12):
        test_y = test_Y[i].detach().numpy()
        yf = fft(test_y)
        model_spectra.append(2.0/441*np.abs(yf[:trunc_length])) 

    torch.save(
        model_spectra, 
        f"/home/yz685/low_rank_BOPE/low_rank_BOPE/test_problems/music/model_spectra_{trunc_length}.pt"
    )

    return model_spectra


# outcome function simulating signals from pressing two keys in an octave
# NOTE: heat map of utility over [0,1]^2 would be cool
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


  