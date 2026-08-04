import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from scipy import constants


T_CMB = 2.72548


@jax.jit
def _rj2cmb(nu, T=T_CMB):
    x = nu * constants.h * 1e9 / (constants.k * T)
    return (jnp.expm1(x) / x) ** 2. / jnp.exp(x)


class ConstantSED:
    """
        f(nu) = 1
    """
    def __init__(self, **kwargs):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, nu):
        return jnp.ones_like(nu)


class PowerLawSED:
    """
        f(nu) = (nu / nu0) ** beta
    """
    def __init__(self, nu0=150., **kwargs):
        self.nu0 = nu0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, nu, beta=1.0, T_cmb=T_CMB):
        fnu = jnp.where(nu == 0, 1, nu)
        sed = (fnu / self.nu0) ** beta * (_rj2cmb(fnu,T_cmb) / _rj2cmb(self.nu0, T_cmb))
        return jnp.where(nu == 0, 0, sed)


class ModifiedBlackBodySED:
    """
        f(nu) = (nu / nu0) ** (beta + 1) / (e^(h nu / kB T) - 1)
    """
    def __init__(self, nu0=150., **kwargs):
        self.nu0 = nu0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, nu, T=1., beta=1.0, T_cmb=T_CMB):
        fnu = jnp.where(nu == 0, 1, nu)
        x = fnu * constants.h * 1e9 / (constants.k * T)
        x0 = self.nu0 * constants.h * 1e9 / (constants.k * T)
        sed = (fnu / self.nu0) ** (beta + 1.) * (jnp.expm1(x0) * _rj2cmb(fnu, T_cmb)) / (jnp.expm1(x) * _rj2cmb(self.nu0, T_cmb))
        return jnp.where(nu == 0, 0, sed)


class ThermalSZSED:
    """
        f(nu) = (h nu / kB T) coth(h nu / 2 kB T) - 4
    """
    def __init__(self, nu0=150., **kwargs):
        self.nu0 = nu0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, nu, T_cmb=T_CMB):
        fnu = jnp.where(nu == 0, 1, nu)
        x = fnu * constants.h * 1e9 / (constants.k * T_cmb)
        x0 = self.nu0 * constants.h * 1e9 / (constants.k * T_cmb)
        fx = x / jnp.tanh(x / 2.) - 4.
        fx0 = x0 / jnp.tanh(x0 / 2.) - 4.
        sed = fx / fx0
        return jnp.where(nu == 0, 0, sed)
