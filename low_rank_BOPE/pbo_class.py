
from low_rank_BOPE.src.transforms import (compute_weights,
                                          get_latent_ineq_constraints)


class PboExperiment:

    # add weighted PCA, add explicit computation in latent space

    attr_list = {}

    def __init__(self):
        pass

    def generate_random_pref_data(self):
        pass
    
    def fit_pref_model(self):
        pass

    def run_pref_learning(self):
        pass

    def find_max_posterior_mean(self):
        pass

    def run_PBO_loop(self):
        self.generate_random_pref_data()
        for method in self.methods:
            # run
            # save