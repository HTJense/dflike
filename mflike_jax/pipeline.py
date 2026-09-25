from jax import numpy as jnp
from .theory import Theory
from .likelihood import Likelihood


class Pipeline:
    def __init__(self, components: list):
        self.components = components
        self.theories = [comp for comp in self.components
                         if isinstance(comp, Theory)]
        self.likelihoods = [comp for comp in self.components
                            if isinstance(comp, Likelihood)]

        self.parameters = set([])
        for comp in self.components:
            self.parameters |= set(comp.parameters)
        self.parameters = list(self.parameters)

        self.th_indices = [
            jnp.array([
                self.parameters.index(i) for i in th.parameters
            ]) for th in self.theories
        ]

        self.like_indices = [
            jnp.array([
                self.parameters.index(i) for i in like.parameters
            ]) for like in self.likelihoods
        ]

    def logposterior(self, theta: jnp.ndarray) -> float:
        products = {}
        left_to_compute = list(enumerate(self.theories))
        while len(left_to_compute) > 0:
            computed_something = False
            for i, th in left_to_compute:
                if all([x in products for x in th.inputs]):
                    th_theta = theta[self.th_indices[i]]
                    products |= th.compute(th_theta, **products)
                    left_to_compute.remove((i, th))
                    computed_something = True
                    break

            if not computed_something and len(left_to_compute) > 0:
                raise RuntimeError("Failed to compute everything - "
                                   "Did you check your theory dependencies?")

        logp = jnp.sum(jnp.array([
            like.loglike(theta[idx], **products)
            for idx, like in zip(self.like_indices, self.likelihoods)
        ]))

        return logp
