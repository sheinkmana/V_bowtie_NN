from .sparsify_vbnn import VBNNSparsityMixin

class VBNNBase(VBNNSparsityMixin):
    """
    Base class for Variational Bayesian Neural Networks.
    
    Inheritance Hierarchy:
    VBNNBase -> VBNNSparsityMixin -> VBNNPredictionMixin -> VBNNCore
    """
    pass