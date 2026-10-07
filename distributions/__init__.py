from functools import partial
from .base import TargetDistribution
from .banana import ProductBanana
from .haario import HaarioBanana
from .funnel import NealFunnel
from .posterior import Posterior
from .logistic import LogisticRegression, SparseLogistic
from .hierarchical import SV_ADT, EightSchools, StochasticVolatility
from .ode import LotkaVolterra
from .multimodal import Phi4, SensorNetwork
from .posteriordb import PDB_POSTERIORS
from .mixture_gauss import GaussMixture, TMixture
from .synthetic import BananaBench, FunnelBench
TARGETS = {'banana': BananaBench, 'banana_product': ProductBanana, 'funnel': FunnelBench}
POSTERIORS = {'logistic_german': partial(LogisticRegression, 'german'), 'logistic_heart': partial(LogisticRegression, 'heart'), 'sparse_logistic_german': partial(SparseLogistic, 'german'), 'sparse_logistic_heart': partial(SparseLogistic, 'heart'), 'stochvol_T50': partial(StochasticVolatility, 50), 'stochvol_T100': partial(StochasticVolatility, 100), 'eight_schools': EightSchools, 'lotka_volterra': LotkaVolterra, 'sensor': SensorNetwork, **{f'phi4_N{n}': partial(Phi4, n) for n in (16, 32, 64, 128)}, **PDB_POSTERIORS, 'gmix_d50_h6': partial(GaussMixture, 50, 6.0), 'gmix_d20_h4': partial(GaussMixture, 20, 4.0), 'gmix_d10_h6': partial(GaussMixture, 10, 6.0), 'tmix_d10_h6': partial(TMixture, 10, 6.0, 5.0), **{f'stochvol_adt_d{d}': partial(StochasticVolatility, d - 3, name=f'stochvol_adt_d{d}', params=SV_ADT) for d in (50, 100)}}
TARGETS.update(POSTERIORS)

def make_target(name, **kwargs):
    try:
        return TARGETS[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown target '{name}' (choose from {sorted(TARGETS)})") from None
__all__ = ['TargetDistribution', 'ProductBanana', 'HaarioBanana', 'NealFunnel', 'Posterior', 'LogisticRegression', 'SparseLogistic', 'EightSchools', 'StochasticVolatility', 'LotkaVolterra', 'SensorNetwork', 'Phi4', 'TARGETS', 'POSTERIORS', 'make_target']
