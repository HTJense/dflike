from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import jax.scipy as jsc


class Likelihood(ABC):
    @abstractmethod
    def loglike(self, theta, **kwargs):
        ...

    @abstractmethod
    def fisher(self, theta, **kwargs):
        ...


class GaussianLikelihood(Likelihood):
    data_vec: jnp.ndarray
    covariance: jnp.ndarray
    logp_const: float

    @abstractmethod
    def get_model(self, theta, **kwargs):
        ...

    def get_covariance(self, theta, **kwargs):
        return self.covariance

    def get_covariance_cholesky(self, theta, **kwargs):
        covariance = self.get_covariance(theta, **kwargs)
        return jnp.linalg.cholesky(covariance)

    def get_whitened_residual(self, theta, **kwargs):
        mu = self.get_model(theta, **kwargs)
        r = self.data_vec - mu
        L = self.get_covariance_cholesky(theta, **kwargs)
        return jsc.linalg.solve_triangular(L, r, lower=True)

    def loglike(self, theta, **kwargs):
        return -0.5 * self.chisquare(theta, **kwargs) + self.logp_const

    def chisquare(self, theta, **kwargs):
        w = self.get_whitened_residual(theta, **kwargs)
        return jnp.sum(w ** 2)

    def fisher(self, theta, **kwargs):
        J = jax.jacfwd(lambda x: self.get_model(x, **kwargs))(theta)
        L = self.get_covariance_cholesky(theta, **kwargs)
        W = jsc.linalg.solve_triangular(L, J, lower=True)

        return W.T @ W
