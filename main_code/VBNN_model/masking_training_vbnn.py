import numpy as np
from typing import List
from .training_vbnn import VBNN_improving, VBNN_SVI_improving

class MixinBase(VBNN_improving, VBNN_SVI_improving): 
    pass

class IterativePruningMixin(MixinBase):
    """
    Mixin that adds iterative pruning capabilities to training algorithms.
    Overrides 'algorithm' (CAVI) and 'svi_alg' (SVI).
    """
    # Explicit type hints for attributes used in the mixin
    elbo_total: List[float]
    epoch_no: int
    cache_valid: bool
    L: int

    def algorithm(self, epochs=100, rate=0.000001, EM_step=False, 
                  pruning_interval=20, pruning_alpha=0.01):
        """
        Modified CAVI algorithm with iterative masking.
        """
        self.elbo_total = []
        self.epoch_no = 0
        
        print(f"Starting Iterative CAVI: {epochs} epochs, Pruning every {pruning_interval} (alpha={pruning_alpha})")

        for _ in range(epochs):
            self.cache_valid = False
            self._compute_forward_pass()
            # Standard CAVI Updates (Methods from VBNN_improving)
            for k in range(self.L): 
                self.update_hid_part1(k)
            self.update_out_part1()
            self.update_a()
            for k in range(self.L): 
                self.update_hid_part2(k)
            self.update_out_part2()
            
            if EM_step:
                self._new_delta()

            # Iterative Sparsification
            if self.epoch_no > 0 and self.epoch_no % pruning_interval == 0:
                print(f"  [Epoch {self.epoch_no}] Performing FDR Masking.")
                # Method from VBNNSparsityMixin
                self.mask_parameters(alpha=pruning_alpha)

            # Convergence Check
            if self.epoch_no % 20 == 0:
                self.elbo()
            
            if len(self.elbo_total) > 3:
                if np.abs(1 - self.elbo_total[-2]/self.elbo_total[-1]) < rate:
                    print(f"Converged at epoch {self.epoch_no}")
                    break
            
            self.epoch_no += 1
            
        self.mask_parameters(alpha=pruning_alpha)

    def svi_alg(self, epochs=1, forgrate=0.9, EM_step=False, rate_local=1e-4,
                pruning_interval=20, pruning_alpha=0.01):
        """
        Modified SVI algorithm with iterative masking.
        """
        self.elbo_total = []
        self.epoch_no = 0
        
        print(f"Starting Iterative SVI: {epochs} epochs, Pruning every {pruning_interval} (alpha={pruning_alpha})")

        for _ in range(epochs):
            self.cache_valid = False
            self._compute_forward_pass()
            # Standard SVI Updates (Methods from VBNN_SVI_improving)
            self.update_cavi_params()
            self.find_optimal_local_params(rate_local=rate_local)
            self.update_global_params(forgrate=forgrate)
            if EM_step:
                self._new_delta()
            # Iterative Sparsification
            if self.epoch_no > 0 and self.epoch_no % pruning_interval == 0:
                print(f"  [Epoch {self.epoch_no}] Performing FDR Masking.")
                self.mask_parameters(alpha=pruning_alpha)
            if self.epoch_no % 20 == 0:
                self.elbo()
            self.epoch_no += 1
            self._init_sample_params()
        self.mask_parameters(alpha=pruning_alpha)


class VBNN_Iterative_CAVI(IterativePruningMixin, VBNN_improving):
    """Concrete class for Iterative CAVI training."""
    pass

class VBNN_Iterative_SVI(IterativePruningMixin, VBNN_SVI_improving):
    """Concrete class for Iterative SVI training."""
    pass