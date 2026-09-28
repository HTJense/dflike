import jax
import jax.numpy as jnp
import jax.scipy as jsc
from .likelihood import Likelihood, GaussianLikelihood
from .theory import Theory


class Pipeline:
    def __init__(self, components: list):
        self.components = components
        self.theories = [comp for comp in self.components
                         if isinstance(comp, Theory)]
        self.likelihoods = [comp for comp in self.components
                            if isinstance(comp, Likelihood)]

        self.parameters_full = list(dict.fromkeys(
            p for comp in self.components for p in comp.parameters
        ))
        self.parameters = self.parameters_full.copy()
        self.theta_default = jnp.array([jnp.nan for _ in self.parameters_full])
        self.idx_expand = jnp.array(range(len(self.parameters)))

        self.th_indices = [
            jnp.array([
                self.parameters_full.index(i) for i in th.parameters
            ]) for th in self.theories
        ]

        self.like_indices = [
            jnp.array([
                self.parameters_full.index(p) for p in like.parameters
            ]) for like in self.likelihoods
        ]
        self.like_free_indices = [
            idx.copy() for idx in self.like_indices
        ]

    def restrict(self, theta_fixed: dict):
        self.parameters = self.parameters_full.copy()
        for p in theta_fixed:
            self.parameters.remove(p)

        self.theta_default = jnp.array([
            theta_fixed.get(p, jnp.nan) for p in self.parameters_full
        ])
        self.idx_expand = jnp.array([
            self.parameters_full.index(p) for p in self.parameters
        ])
        self.like_free_indices = [
            jnp.array([
                like.parameters.index(p)
                for p in like.parameters
                if p in self.parameters
            ])
            for like in self.likelihoods
        ]

    def expand(self, theta_free: jnp.ndarray) -> jnp.ndarray:
        if theta_free.shape == self.theta_default.shape:
            # This vector has already been expanded, so don't do it again!
            return theta_free

        return self.theta_default.at[self.idx_expand].set(theta_free)

    def compute(self, theta: jnp.ndarray) -> dict:
        theta_full = self.expand(theta)
        products = {}
        left_to_compute = list(enumerate(self.theories))
        while len(left_to_compute) > 0:
            computed_something = False
            for i, th in left_to_compute:
                if all([x in products for x in th.inputs]):
                    th_theta = theta_full[self.th_indices[i]]
                    products |= th.compute(th_theta, **products)
                    left_to_compute.remove((i, th))
                    computed_something = True
                    break

            if not computed_something and len(left_to_compute) > 0:
                raise RuntimeError("Failed to compute everything - "
                                   "Did you check your theory dependencies?")

        return products

    def logposterior(self, theta: jnp.ndarray) -> float:
        """
            Compute all theories and yield the log-posterior.
        """
        theta_full = self.expand(theta)
        products = self.compute(theta_full)

        logp = jnp.sum(jnp.array([
            like.loglike(theta_full[idx], **products)
            for idx, like in zip(self.like_indices, self.likelihoods)
        ]))

        return logp

    def fisher(self, theta: jnp.ndarray) -> jnp.ndarray:
        F = jnp.zeros((theta.size, theta.size))

        theta_full = self.expand(theta)
        products = self.compute(theta_full)

        for idx, f_idx, like in zip(self.like_indices, self.like_free_indices,
                                    self.likelihoods):
            if isinstance(like, GaussianLikelihood):
                def model(x):
                    x_f = self.expand(x)
                    products = self.compute(x_f)
                    return like.get_model(x_f[idx], **products)

                J = jax.jacfwd(model)(theta)
                L = like.get_covariance_cholesky(theta, **products)
                W = jsc.linalg.solve_triangular(L, J, lower=True)

                F = F + W.T @ W
            else:
                F_like = like.fisher(theta_full[idx], **products)
                F = F.at[jnp.ix_(f_idx, f_idx)].add(
                    F_like[jnp.ix_(f_idx, f_idx)]
                )

        return F
