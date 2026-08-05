import numpy as np
from functools import partial
import jax
import jax.numpy as jnp


def _bp_int(nu, f, bp):
    return jnp.trapezoid(f * bp, nu)


class CrossProductModel:
    """
       C_ell^(nu1xnu2) = A * C_ell * f(nu1) * f(nu2)
    """
    def __init__(self, power, sed):
        self.power = power
        self.sed = sed

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, nus, bps, theta):
        theta_cl = theta[:self.power.n]
        theta_sed = theta[self.power.n:]

        cl = self.power(ell, theta_cl)

        f_nus = jnp.zeros((len(nus,)))
        for i, (nu, bp) in enumerate(zip(nus, bps)):
            f_nu = self.sed(nu, theta_sed)
            f_nus = f_nus.at[i].set(_bp_int(nu, f_nu, bp))
        return jnp.einsum("...i,...j,...l->...ijl", cl, f_nus, f_nus)

    @property
    def n(self):
        return self.power.n + self.sed.n


class CorrelatedCrossProductModel:
    """
        For two components, A and B, which are correlated:
        C_ell^(nu1xnu2) =
            a_A * C_ell^A f^A(nu1) f^A(nu2)
            + a_B * C_ell^B f^B(nu1) f^B(nu2)
            + a_(AxB) C_ell^(AxB) (f^A(nu1) f^B(nu2) + f^A(nu2) f^B(nu1)
    """
    def __init__(self, power1, power2, powerx, sed1, sed2):
        self.power1 = power1
        self.power2 = power2
        self.powerx = powerx
        self.sed1 = sed1
        self.sed2 = sed2
        self.n_p1 = jnp.arange(self.power1.n)
        self.n_p2 = jnp.arange(self.power2.n) + self.power1.n
        self.n_px = jnp.arange(self.powerx.n) + self.power1.n + self.power2.n
        self.n_s1 = jnp.arange(self.sed1.n) + self.power1.n + self.power2.n + self.powerx.n
        self.n_s2 = jnp.arange(self.sed2.n) + self.power1.n + self.power2.n + self.powerx.n + self.sed1.n

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, ell, nus, bps, theta):
        theta_cl1 = theta[self.n_p1]
        theta_cl2 = theta[self.n_p2]
        theta_clx = theta[self.n_px]
        theta_s1 = theta[self.n_s1]
        theta_s2 = theta[self.n_s2]

        cl1 = self.power1(ell, theta_cl1)
        cl2 = self.power2(ell, theta_cl2)
        clx = self.powerx(ell, theta_clx)

        f_nus1 = jnp.zeros((len(nus),))
        f_nus2 = jnp.zeros((len(nus),))
        for i, (nu, bp) in enumerate(zip(nus, bps)):
            f_nu = self.sed1(nu, theta_s1)
            f_nus1 = f_nus1.at[i].set(_bp_int(nu, f_nu, bp))

            f_nu = self.sed2(nu, theta_s2)
            f_nus2 = f_nus2.at[i].set(_bp_int(nu, f_nu, bp))

        comp1 = jnp.einsum("...i,...j,...l->...ijl", cl1, f_nus1, f_nus1)
        comp2 = jnp.einsum("...i,...j,...l->...ijl", cl2, f_nus2, f_nus2)
        compx = clx[:,None,None] * jnp.sqrt(self.power1.amp(theta_cl1) * self.power2.amp(theta_cl2)) * (f_nus1[None,:,None] * f_nus2[None,None,:] + f_nus1[None,None,:] * f_nus2[None,:,None])

        return comp1 + comp2 - compx
        """
        cl1 = self.power1(**power1_params)
        cl2 = self.power2(**power2_params)
        clx = self.powerx(**powerx_params)

        nus1 = sed1_params["nu"]
        bps1 = sed1_params["bp"]
        sed1_args = {k: v for k, v in sed1_params.items() if k not in ("nu", "bp")}

        f_nus1 = jnp.zeros((len(nus1),))
        for i, (nu1, bp1) in enumerate(zip(nus1, bps1)):
            f_nu = self.sed1(nu=nu1, **sed1_args)
            f_nus1 = f_nus1.at[i].set(_bp_int(nu1, f_nu, bp1))

        nus2 = sed2_params["nu"]
        bps2 = sed2_params["bp"]
        sed2_args = {k: v for k, v in sed2_params.items() if k not in ("nu", "bp")}

        f_nus2 = jnp.zeros((len(nus2),))
        for i, (nu2, bp2) in enumerate(zip(nus2, bps2)):
            f_nu = self.sed2(nu=nu2, **sed2_args)
            f_nus2 = f_nus2.at[i].set(_bp_int(nu2, f_nu, bp2))

        comp1 = jnp.einsum("...i,...j,...l->...ijl", cl1, f_nus1, f_nus1)
        comp2 = jnp.einsum("...i,...j,...l->...ijl", cl2, f_nus2, f_nus2)

        clx = jnp.broadcast_to(clx[:,None,None], comp1.shape)
        f_nus11 = jnp.broadcast_to(f_nus1[None,:,None], comp1.shape)
        f_nus12 = jnp.broadcast_to(f_nus1[None,None,:], comp1.shape)
        f_nus21 = jnp.broadcast_to(f_nus2[None,:,None], comp2.shape)
        f_nus22 = jnp.broadcast_to(f_nus2[None,None,:], comp2.shape)

        compx = clx * (f_nus11 * f_nus22 + f_nus21 * f_nus12)
        return comp1 + comp2 + compx
        """

    @property
    def n(self):
        return self.power1.n + self.power2.n + self.powerx.n + self.sed1.n + self.sed2.n
