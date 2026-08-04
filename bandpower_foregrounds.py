import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from cobaya.yaml import yaml_load_file
from cobaya.tools import resolve_packages_path
from scipy import constants

import fg_model as fgm
import fg_power as fgp
import fg_sed as fgf

T_CMB = 2.72548

@jax.jit
def _cmb2bb(nu, T=T_CMB):
    x = nu * constants.h * 1e9 / (constants.k * T)
    return jnp.exp(x) * (nu * x / jnp.expm1(x)) ** 2.

class BandpowerForegrounds:
    def __init__(self, config, likelihood, lmax=9000):
        self.config = yaml_load_file(config)
        defaults = self.config["normalisation"]
        self.ksz = fgm.CrossProductModel(fgp.TemplateCl("cl_ksz_bat.dat", **defaults), fgf.ConstantSED(**defaults))

        self.tsz_and_cib = fgm.CorrelatedCrossProductModel(
            fgp.RescaledTemplateCl("cl_tsz_150_bat.dat", **defaults),
            fgp.RescaledTemplateCl("cl_cib_Choi2020.dat", **defaults),
            fgp.TemplateCl("cl_sz_x_cib.dat", **defaults),
            fgf.ThermalSZSED(**defaults),
            fgf.ModifiedBlackBodySED(**defaults)
        )

        self.cibp = fgm.CrossProductModel(fgp.PoissonCl(**defaults), fgf.ModifiedBlackBodySED(**defaults))
        self.radio = fgm.CrossProductModel(fgp.PoissonCl(**defaults), fgf.PowerLawSED(**defaults))
        self.dust = fgm.CrossProductModel(fgp.PowerLawCl(ell0=500), fgf.ModifiedBlackBodySED(**defaults))

        self.ells = likelihood.ells
        self.experiments = self.config["experiments"]
        self.nu = [ jnp.array(likelihood.tracers[exp + "_s0"]["nu"]) for exp in self.experiments ]
        self.bp = [ jnp.array(likelihood.tracers[exp + "_s0"]["bp"] / np.trapezoid(likelihood.tracers[exp + "_s0"]["bp"], likelihood.tracers[exp + "_s0"]["nu"])) for exp in self.experiments ]

        self.parameters = [
            "a_tSZ", "alpha_tSZ", "a_kSZ",
            "xi", "a_c", "beta_c", "a_p", "beta_p",
            "a_s", "beta_s",
            "a_gtt", "a_gte", "a_gee",
            "a_pste", "a_psee",
            "alpha_c", "alpha_s", "T_d", "alpha_p", "alpha_dT", "alpha_dE", "beta_d", "T_effd"
        ]
        self.fg_index = jnp.array(np.arange(len(self.parameters)))
        for exp in self.experiments:
            self.parameters.append(f"bandint_shift_{exp}")
        self.bp_index = jnp.array([ self.parameters.index(f"bandint_shift_{exp}") for exp in self.experiments ])

    @partial(jax.jit, static_argnums=(0,))
    def apply_bandpass_shifts(self, bandint_theta):
        nus = []
        bps = []

        for i, (exp, nu, bp) in enumerate(zip(self.experiments, self.nu, self.bp)):
            nub = nu + bandint_theta[i]
            nus.append(nub)
            bps.append(bp * _cmb2bb(nub) / jnp.trapezoid(bp * _cmb2bb(nub), nub))

        return nus, bps

    @partial(jax.jit, static_argnums=(0,))
    def get_foreground_model(self, theta):
        nu, bp = self.apply_bandpass_shifts(theta[self.bp_index])

        params = { k: theta[i] for i, k in enumerate(self.parameters) }

        ksz = params["a_kSZ"] * self.ksz(
            {"ell": self.ells},
            {"nu": nu, "bp": bp}
        )
        tsz_and_cib = self.tsz_and_cib(
            {"amp": params["a_tSZ"], "ell": self.ells, "alpha": params["alpha_tSZ"]},
            {"amp": params["a_c"], "ell": self.ells, "alpha": params["alpha_c"] - 0.8},
            {"amp": -params["xi"] * jnp.sqrt(params["a_tSZ"] * params["a_c"]), "ell": self.ells},

            {"nu": nu, "bp": bp},
            {"nu": nu, "bp": bp, "T": params["T_d"], "beta": params["beta_c"]}
        )
        cibp = params["a_p"] * self.cibp(
            {"ell": self.ells, "alpha": params["alpha_p"]},
            {"nu": nu, "bp": bp, "T": params["T_d"], "beta": params["beta_p"]}
        )
        radiott = params["a_s"] * self.radio(
            {"ell": self.ells, "alpha": params["alpha_s"]},
            {"nu": nu, "bp": bp, "beta": params["beta_s"]}
        )
        dusttt = params["a_gtt"] * self.dust(
            {"ell": self.ells, "alpha": params["alpha_dT"]},
            {"nu": nu, "bp": bp, "T": params["T_effd"], "beta": params["beta_d"]}
        )

        radiote = params["a_pste"] * self.radio(
            {"ell": self.ells, "alpha": params["alpha_s"]},
            {"nu": nu, "bp": bp, "beta": params["beta_s"]}
        )
        dustte = params["a_gte"] * self.dust(
            {"ell": self.ells, "alpha": params["alpha_dE"]},
            {"nu": nu, "bp": bp, "T": params["T_effd"], "beta": params["beta_d"]}
        )

        radioee = params["a_psee"] * self.radio(
            {"ell": self.ells, "alpha": params["alpha_s"]},
            {"nu": nu, "bp": bp, "beta": params["beta_s"]}
        )
        dustee = params["a_gee"] * self.dust(
            {"ell": self.ells, "alpha": params["alpha_dE"]},
            {"nu": nu, "bp": bp, "T": params["T_effd"], "beta": params["beta_d"]}
        )

        fg_tt = tsz_and_cib + ksz + cibp + radiott + dusttt
        fg_te = radiote + dustte
        fg_ee = radioee + dustee

        return jnp.stack([ fg_tt, fg_te, fg_ee ])
