from typing import Optional, List

import numpy as np
import torch
from botorch.test_functions import SyntheticTestFunction
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
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
    frequencies: List, 
    amplitudes: List[float],
    duration: float, 
    sample_rate: Optional[int] = 44100, 
):
    """
    Get (combined) sine wave of specified frequencies.

    Args:
        frequencies: `n_frequencies x 1` tensor of frequencies in hertz
        amplitudes: peak amplitudes of each of the `n_frequencies` waves
        duration: time of measurement in seconds
        sample_rate: number of measurements per second
    Returns:
        `duration*sample_rate x 1` tensor of wave signals
    """

    waves = []
    t = np.linspace(0, duration, int(sample_rate*duration))
    for frequency, amplitude in zip(frequencies, amplitudes):
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
        # self.key_size = 1/12
        self.sample_rate = sample_rate
        self.duration = duration
        self.amplitude = amplitude
        
        self.outcome_dim = int(self.sample_rate * self.duration)
        
        
    def evaluate_true(self, X):
        
        # keys = torch.div(X, self.key_size, rounding_mode="floor") + 1 # rounding issues
        keys = torch.round(X * 12) + 1
        # print("keys: ", keys)
        Y = []

        for sample_idx in range(X.shape[-2]):
            key = int(self.base_key + keys[sample_idx].item())
            freq = self.note_freqs[key]
            # print("self.base_freq:", self.base_freq, "key: ", key, "frequency: ", freq)
            Y.append(
                get_combined_sine_wave(
                    frequencies=[self.base_freq, freq], 
                    duration=self.duration,
                    sample_rate=self.sample_rate,
                    amplitude=self.amplitude
                )
            )
        
        return torch.vstack(Y)


class NewHarmonyOneKey(SyntheticTestFunction):
    r"""
    Class for simulating playing multiple keys in the octave above A4 together,
    with potentially different intensities.
    """

    dim = 12
    _bounds = torch.tensor([[0., 1.] for _ in range(12)])

    def __init__(
        self, 
        base_key = 48, # key 49 = A4
        sample_rate = 44100,
        duration = 0.01, 
        base_amplitude = 1
    ): 
        super().__init__()
        # self.outcome_bounds has to do with amplitude
        self.all_note_freqs = [2**((n+1-49)/12)*440 for n in range(88)]
        self.note_freqs = self.all_note_freqs[49:61]
        self.base_key = base_key
        self.base_freq = self.all_note_freqs[base_key]
        self.base_amplitude = base_amplitude

        self.sample_rate = sample_rate
        self.duration = duration
        
        self.outcome_dim = int(self.sample_rate * self.duration)
        
        
    def evaluate_true(self, X):
             
        Y = []

        for sample_idx in range(X.shape[-2]):
            Y.append(
                get_combined_sine_wave(
                    frequencies=[self.base_freq] + self.note_freqs,
                    # TODO: later try 2*X[sample_idx] for amplitude
                    amplitudes = [self.base_amplitude] + X[sample_idx].tolist(),
                    duration=self.duration,
                    sample_rate=self.sample_rate,
                )
            )
        
        return torch.vstack(Y)
    
########################################################################################
######################################################################################## 

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
    take a weighted average of their pleasantness values (weights = 1/(distance+epsilon) or something)

"""
class Consonance(torch.nn.Module):
    def __init__(
        self, 
        model_spectra: list, 
        dissonance_vals: list, 
        util_type: str = "inverse_l2_nn",
        similarity_eps: float = 0.5, # TODO: make this a kwarg
    ):
        r"""
        Quantifies the level of pleasantness of a sound signal vector.

        Args:
            model_spectra:
            dissonance_vals:
            util_type: 
                "inverse_l2_normalized":
                "inverse_l2_unnormalized":
                "dot_product": 
            similarity_eps: 

        """

        super().__init__()
        self.model_spectra = model_spectra
        self.dissonance_vals = dissonance_vals
        self.util_type = util_type
        self.similarity_eps = similarity_eps # TODO: make kwargs
    
    def forward(self, Y: Tensor):

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
        # TODO: think through, enable more options

        if self.util_type.startswith("inverse_l2"):
            spectrum_similarity = []
            for spectrum in self.model_spectra:
                spectrum_distance = np.linalg.norm(yf-spectrum) # TODO: use vector operation later
                spectrum_similarity.append(1/(spectrum_distance+self.similarity_eps)) # TODO: come back to this
            if self.util_type == "inverse_l2_normalized_wmean":
                spec_sim_sum = sum(spectrum_similarity)
                res = np.dot(
                    np.array(spectrum_similarity) / spec_sim_sum,
                    # 1-np.array(self.dissonance_vals)
                    -np.log(np.array(self.dissonance_vals))
                )
            elif self.util_type == "inverse_l2_unnormalized_wmean":
                res = np.dot(
                    np.array(spectrum_similarity),
                    # 1-np.array(self.dissonance_vals)
                    -np.log(np.array(self.dissonance_vals))
                )
            elif self.util_type == "inverse_l2_nn":
                ind = np.argmax(spectrum_similarity)
                res = -np.log(self.dissonance_vals[ind])


        return res


class NewConsonance(torch.nn.Module):
    def __init__(
        self, 
        dissonance_vals: list, 
        sigma: float = 100
    ):
        r"""
        Quantifies the level of pleasantness of a sound signal vector.

        Args:
            dissonance_vals: 
            sigma: 

        """

        super().__init__()
        self.dissonance_vals = dissonance_vals
        self.sigma = sigma
    
    def forward(self, Y: Tensor):

        res_all = []
        for y in Y:
            res_all.append(self.get_pleasantness(y))
        
        return torch.tensor(res_all).unsqueeze(1)


    def get_pleasantness(self, y: Tensor):

        # transform y to an array
        y_np = y.detach().numpy()
        peaks, amps = self.get_spectra_peaks(y_np)

        # pleasantness due to frequency difference 
        num_keys_apart = int(np.abs(np.log2(peaks[1]/peaks[0]))*12)
        # print("num keys apart: ", num_keys_apart)
        cons_1 = -np.log(self.dissonance_vals[num_keys_apart]) # TODO: double check

        # pleasantness due to amplitude difference
        cons_2 = np.exp(-(amps[0]-amps[1])**2 / self.sigma**2) \
            * np.exp(-(220-max(amps))**2 / self.sigma**2) # TODO: double check

        return cons_1 * cons_2


    def get_spectra_peaks(self, y: np.array):
        r"""
        Input a sound signal, get its two largest positive peaks, 
        return the frequencies and amplitudes of the peaks.
        """
        spec = []
        for freq in np.arange(420, 900, 1):
            spec.append(np.dot(y, get_combined_sine_wave([freq], [1], 0.01)))
        
        # then find the top 2 peaks of res, return in increasing order
        peaks, properties = find_peaks(spec, height=0.1) # TODO: can set height arg
        top2_peaks_idx = np.argpartition(properties["peak_heights"], -2)[-2:]

        top2_peaks = [440+peaks[top2_peaks_idx][i] for i in [0,1]]
        top2_amps = properties["peak_heights"][top2_peaks_idx] 

        return top2_peaks, top2_amps

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


  