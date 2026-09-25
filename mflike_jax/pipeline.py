import jax
import jax.numpy as jnp
import jax.scipy as jsc
from .theory import Theory
from .likelihood import Likelihood, GaussianLikelihood


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

    def compute(self, theta: jnp.ndarray) -> dict:
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

        return products

    def logposterior(self, theta: jnp.ndarray) -> float:
        products = self.compute(theta)

        logp = jnp.sum(jnp.array([
            like.loglike(theta[idx], **products)
            for idx, like in zip(self.like_indices, self.likelihoods)
        ]))

        return logp

    def fisher(self, theta: jnp.ndarray) -> jnp.ndarray:
        F = jnp.zeros((theta.size, theta.size))
        products = self.compute(theta)

        for idx, like in zip(self.like_indices, self.likelihoods):
            if isinstance(like, GaussianLikelihood):
                def model(x):
                    products = self.compute(x)
                    return like.get_model(x[idx], **products)

                J = jax.jacfwd(model)(theta)
                L = like.get_covariance_cholesky(theta[idx], **products)
                W = jsc.linalg.solve_triangular(L, J, lower=True)

                F = F + W.T @ W
            else:
                F_like = like.fisher(theta[idx], **products)
                F.at[jnp.ix_(idx,idx)].add(F_like)

        return F
