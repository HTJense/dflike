import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from scipy import constants


T_cmb = 2.72548
hk_GHz = constants.h * 1e9 / constants.k


@jax.jit
def _rj2cmb(nu, T=T_cmb):
    x = nu * hk_GHz / T
    return (jnp.expm1(x) / x) ** 2. / jnp.exp(x)


class ConstantSED:
    """
        f(nu) = 1
    """
    def __init__(self, **kwargs):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, nu, theta):
        return jnp.ones_like(nu)

    @property
    def n(self):
        return 0

class PowerLawSED:
    """
        f(nu) = (nu / nu_0) ** beta
    """
    def __init__(self, nu_0=150., **kwargs):
        self.nu_0 = nu_0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, nu, theta):
        return (nu / self.nu_0) ** theta[0] * (_rj2cmb(nu) / _rj2cmb(self.nu_0))

    @property
    def n(self):
        return 1

class ModifiedBlackBodySED:
    """
        f(nu) = (nu / nu_0) ** (beta + 1) / (e^(h nu / kB T) - 1)
    """
    def __init__(self, nu_0=150., **kwargs):
        self.nu_0 = nu_0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, nu, theta):
        x = nu * hk_GHz / theta[1]
        x0 = self.nu_0 * hk_GHz / theta[1]
        return (nu / self.nu_0) ** (theta[0] + 1.) * (_rj2cmb(nu) / _rj2cmb(self.nu_0)) * (jnp.expm1(x0) / jnp.expm1(x))

    @property
    def n(self):
        return 2

class ThermalSZSED:
    """
        f(nu) = (h nu / kB T) coth(h nu / 2 kB T) - 4
    """
    def __init__(self, nu_0=150., **kwargs):
        self.nu_0 = nu_0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, nu, theta):
        x = nu * hk_GHz / T_cmb
        x0 = self.nu_0 * hk_GHz / T_cmb
        fx = x / jnp.tanh(x / 2.) - 4.
        fx0 = x0 / jnp.tanh(x0 / 2.) - 4.
        return fx / fx0

    @property
    def n(self):
        return 0
