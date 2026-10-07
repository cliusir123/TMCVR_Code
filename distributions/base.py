import abc

class TargetDistribution(abc.ABC):

    def __init__(self, dim):
        self.dim = dim

    @abc.abstractmethod
    def log_prob(self, x):
        pass

    def sample(self, n, generator=None):
        raise NotImplementedError(f'{type(self).__name__} has no exact sampler')

    @property
    def test_functions(self):
        return []
