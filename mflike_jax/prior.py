import jax
import jax.numpy as jnp
from typing import Callable
from .likelihood import Likelihood


class Prior(Likelihood):
    def __init__(self, priors: dict = {}):
        self.priors = []
        self.H_priors = []
        self.parameters = []

        for par in priors:
            self.add_prior(par, priors[par])

    def add_prior(self, parameter: str, logprior: Callable):
        self.priors.append(logprior)
        self.H_priors.append(jax.hessian(logprior))
        self.parameters.append(parameter)

    def loglike(self, theta: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.sum(jnp.array([
            pi(th) for pi, th in zip(self.priors, theta)
        ]))

    def fisher(self, theta: jnp.ndarray, **kwargs) -> jnp.ndarray:
        res = jnp.zeros((len(self.parameters)))

        for i, pi2 in enumerate(self.H_priors):
            res = res.at[i].set(-pi2(theta[i]))

        return jnp.diag(res)
