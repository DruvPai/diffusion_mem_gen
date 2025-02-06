from typing import Dict
from jax import Array
import equinox as eqx

from diffusion_mem_gen.models.gmm import IsoHomGMMSharedParametersEstimator, IsoHomGMMSplitParametersEstimator
from diffusion_mem_gen.models.colored_template import ColoredTemplatesSharedParametersEstimator

class LearnedVariances(eqx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, key: Array, net: eqx.Module) -> Dict[str, Array]:
        metric_dict = dict()
        if isinstance(net, IsoHomGMMSplitParametersEstimator):
            for i in range(len(net.ts)):
                metric_dict[f"t={net.ts[i]:.3f}"] = net.std[i]**2
        elif isinstance(net, IsoHomGMMSharedParametersEstimator):
            metric_dict[""] = net.std**2
        elif isinstance(net, ColoredTemplatesSharedParametersEstimator):
            metric_dict[""] = net.color_std**2
        return metric_dict
