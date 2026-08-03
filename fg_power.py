import os
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp


class PowerLawCl:
    """
        C_ell = (ell / ell0) ** alpha
    """
    def __init__(self, ell0=3000, **kwargs):
        assert ell0 > 0, "ell0 must be positive."
        self.ell0 = ell0

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, amp=1., alpha=0.0):
        ls = jnp.where(ell <= 0, 1, ell)
        cl = amp * (ls / self.ell0) ** alpha
        return jnp.where(ell <= 0, 0, cl)


class PoissonCl:
    """
        C_ell = (ell (ell + 1) / ell0 (ell0 + 1)) ** alpha
    """
    def __init__(self, ell0=3000, **kwargs):
        assert ell0 > 0, "ell0 must be positive."
        self.ell0 = ell0
        self.elp0 = ell0 * (ell0 + 1.)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, amp=1., alpha=0.0):
        ls = jnp.where(ell <= 0, 1, ell)
        elp = ls * (ls + 1.)
        cl = amp * (elp / self.elp0) ** alpha
        return jnp.where(ell <= 0, 0, cl)


class TemplateCl:
    """
       C_ell = C_ell^template / C_ell0^template
    """
    def __init__(self, filename, ell0=3000, **kwargs):
        assert ell0 > 0, "ell0 must be positive."
        filepath = os.path.join(os.path.dirname(__file__), filename)
        ls, cl = np.loadtxt(filepath, unpack=True)
        ls = ls.astype(int)
        template_cl = np.zeros(ls.max() + 1)
        template_cl[ls] = cl
        self.ell0 = ell0
        self.template_cl = jnp.array(template_cl / template_cl[ell0])

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, amp=1.):
        return amp * self.template_cl[ell]


class RescaledTemplateCl(TemplateCl):
    """
       C_ell = C_ell^template / C_ell0^template * (ell / ell0) ** alpha
    """
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, amp=1., alpha=0.0):
        ls = jnp.where(ell <= 0, 1, ell)
        cl = amp * self.template_cl[ls] * (ls / self.ell0) ** alpha
        return jnp.where(ell <= 0, 0, cl)
