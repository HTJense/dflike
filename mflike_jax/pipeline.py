import jax
import jax.numpy as jnp
import jax.scipy as jsc
from typing import Callable, Optional
import optax
from .likelihood import Likelihood, GaussianLikelihood
from .prior import Prior
from .theory import Theory


class Pipeline:
    def __init__(self, components: list, prior: Optional[Prior] = None):
        used_components = []
        for comp in components:
            if isinstance(comp, Prior):
                if prior is not None:
                    raise ValueError("Prior present in component list and "
                                     "passed on separately! Should be one or "
                                     "the other.")
                prior = comp
            elif isinstance(comp, Likelihood) or isinstance(comp, Theory):
                used_components.append(comp)
            else:
                raise ValueError("What am I to do with {comp}?")

        components = used_components
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
        self.free_to_full = jnp.array(range(len(self.parameters)))

        self.prior = prior
        self.prior_indices = jnp.array([])
        self.prior_free_indices = jnp.array([])
        if self.prior is not None:
            self.prior_indices = jnp.array([
                self.parameters_full.index(i) for i in self.prior.parameters
            ])
            self.prior_free_indices = self.prior_indices.copy()

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

        self.jacobians = None
        self.models = None

    def restrict(self, theta_fixed: dict):
        self.parameters = self.parameters_full.copy()
        for p in theta_fixed:
            self.parameters.remove(p)

        self.theta_default = jnp.array([
            theta_fixed.get(p, jnp.nan) for p in self.parameters_full
        ])
        self.free_to_full = jnp.array([
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
        if self.prior is not None:
            self.prior_free_indices = jnp.array([
                self.prior.parameters.index(p)
                for p in self.prior.parameters
                if p in self.parameters
            ])

        # Clear the Jacobians cache (they should be re-built).
        self.jacobians = None
        self.models = None

    def expand(self, theta_free: jnp.ndarray) -> jnp.ndarray:
        if theta_free.shape == self.theta_default.shape:
            # This vector has already been expanded, so don't do it again!
            return theta_free

        return self.theta_default.at[self.free_to_full].set(theta_free)

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

    def logprior(self, theta: jnp.ndarray) -> float:
        logpi = 0.0
        if self.prior is not None:
            theta_full = self.expand(theta)
            logpi = self.prior.logprior(theta_full[self.prior_indices])

        return logpi

    def loglike(self, theta: jnp.ndarray) -> float:
        """
            Compute all theories and yield the log-likelihood.
        """
        theta_full = self.expand(theta)
        products = self.compute(theta)

        loglike = jnp.sum(jnp.array([
            like.loglike(theta_full[idx], **products)
            for idx, like in zip(self.like_indices, self.likelihoods)
        ]))

        return loglike

    def logposterior(self, theta: jnp.ndarray) -> float:
        """
            Alias for `logprior(theta) + loglike(theta)`.
        """
        return self.logprior(theta) + self.loglike(theta)

    def get_models(self) -> list[Callable]:
        if self.models is None:
            self.models = []

            for idx, like in zip(self.like_indices, self.likelihoods):
                if isinstance(like, GaussianLikelihood):
                    def model(x):
                        x_f = self.expand(x)
                        products = self.compute(x_f)
                        return like.get_model(x_f[idx], **products)

                    self.models.append(jax.jit(model))
                else:
                    # Doesn't necessarily exist.
                    self.models.append(None)

        return self.models

    def get_jacobians(self) -> list[Callable]:
        if self.jacobians is None:
            models = self.get_models()
            self.jacobians = []

            for idx, mod, like in zip(self.like_indices, models,
                                      self.likelihoods):
                if isinstance(like, GaussianLikelihood):
                    self.jacobians.append(jax.jit(jax.jacfwd(mod)))
                else:
                    # Doesn't necessarily exist.
                    self.jacobians.append(None)

        return self.jacobians

    def fisher(self, theta: jnp.ndarray) -> jnp.ndarray:
        F = jnp.zeros((theta.size, theta.size))

        theta_full = self.expand(theta)
        products = self.compute(theta_full)
        jac = self.get_jacobians()

        if self.prior is not None:
            idx = self.prior_free_indices
            F_prior = self.prior.fisher(theta_full[self.prior_indices])
            F = F.at[jnp.ix_(idx, idx)].add(F_prior[jnp.ix_(idx, idx)])

        for idx, J, f_idx, like in zip(self.like_indices, jac,
                                       self.like_free_indices,
                                       self.likelihoods):
            if isinstance(like, GaussianLikelihood):
                L = like.get_covariance_cholesky(theta_full[idx], **products)
                W = jsc.linalg.solve_triangular(L, J(theta), lower=True)

                F = F + W.T @ W
            else:
                F_like = like.fisher(theta_full[idx], **products)
                F = F.at[jnp.ix_(f_idx, f_idx)].add(
                    F_like[jnp.ix_(f_idx, f_idx)]
                )

        return F

    def minimize(self, theta_start: jnp.ndarray, max_steps: int,
                 **kwargs) -> list[jnp.ndarray]:
        # Find the best-fitting log-posterior.
        optimizer = optax.adam(**kwargs)
        state = optimizer.init(theta_start)

        vgrad = jax.jit(jax.value_and_grad(
            lambda theta: -self.logposterior(theta)
        ))

        chain = [theta_start]
        theta = theta_start.copy()

        for _ in range(max_steps):
            v, g = vgrad(theta)
            updates, state = optimizer.update(g, state, theta)
            theta = optax.apply_updates(theta, updates)

            if jnp.any(jnp.isnan(theta)):
                print("Non-finite theta.")
                break
            if not jnp.isfinite(v):
                print("Non-finite log-posterior.")
                break
            if jnp.any(jnp.isnan(g)):
                print("Non-finite gradient.")
                break

            chain.append(theta)

        return chain
