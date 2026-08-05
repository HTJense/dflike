import os
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp


class PowerLawCl:
    """
        C_ell = (ell / ell_0) ** alpha
    """
    def __init__(self, ell_0=3000, **kwargs):
        assert ell_0 > 0, "ell_0 must be positive."
        self.ell_0 = ell_0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, theta):
        cl = theta[0] * (ell / self.ell_0) ** theta[1]
        return jnp.where(ell <= 0, 0, cl)

    @property
    def n(self):
        return 2

    def amp(self, theta):
        return theta[0]


class PoissonCl:
    """
        C_ell = (ell (ell + 1) / ell_0 (ell_0 + 1)) ** alpha
    """
    def __init__(self, ell_0=3000, **kwargs):
        assert ell_0 > 0, "ell_0 must be positive."
        self.ell_0 = ell_0
        self.elp0 = ell_0 * (ell_0 + 1.)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, theta):
        ls = jnp.where(ell <= 0, 1, ell)
        elp = ls * (ls + 1.)
        cl = theta[0] * (elp / self.elp0) ** theta[1]
        return jnp.where(ell <= 0, 0, cl)

    @property
    def n(self):
        return 2

    def amp(self, theta):
        return theta[0]

class TemplateCl:
    """
       C_ell = C_ell^template / C_ell_0^template
    """
    def __init__(self, filename, ell_0=3000, **kwargs):
        assert ell_0 > 0, "ell_0 must be positive."
        filepath = os.path.join(os.path.dirname(__file__), "data", filename)
        ls, cl = np.loadtxt(filepath, unpack=True)
        ls = ls.astype(int)
        template_cl = np.zeros(ls.max() + 1)
        template_cl[ls] = cl
        self.ell_0 = ell_0
        self.template_cl = jnp.array(template_cl / template_cl[ell_0])

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, theta):
        return theta[0] * self.template_cl[ell]

    @property
    def n(self):
        return 1

    def amp(self, theta):
        return theta[0]

class RescaledTemplateCl(TemplateCl):
    """
       C_ell = C_ell^template / C_ell_0^template * (ell / ell_0) ** (alpha - alpha_0)
    """
    def __init__(self, filename, ell_0=3000, alpha_0=0.0, **kwargs):
        super().__init__(filename, ell_0, **kwargs)
        self.alpha_0 = alpha_0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, theta):
        ls = jnp.where(ell <= 0, 1, ell)
        cl = theta[0] * self.template_cl[ls] * (ls / self.ell_0) ** (theta[1] - self.alpha_0)
        return jnp.where(ell <= 0, 0, cl)

    @property
    def n(self):
        return 2

    def amp(self, theta):
        return theta[0]
