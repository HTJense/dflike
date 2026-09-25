import numpy as np
import jax
import jax.numpy as jnp
from cobaya.yaml import yaml_load_file
from scipy import constants

from .theory import Theory

from . import fg_model as fgm
from . import fg_power as fgp
from . import fg_sed as fgf

T_CMB = 2.72548


@jax.jit
def _cmb2bb(nu, T=T_CMB):
    x = nu * constants.h * 1e9 / (constants.k * T)
    return jnp.exp(x) * (nu * x / jnp.expm1(x)) ** 2.


class BandpowerForegrounds(Theory):
    def __init__(self, config: str | dict, likelihood, lmax: int = 9000):
        if type(config) is str:
            self.config = yaml_load_file(config)
        elif type(config) is dict:
            self.config = config
        else:
            raise TypeError("Configuration should be string (filename) or "
                            f"dictionary, but was given {type(config)}.")

        self.ells = likelihood.ells
        self.experiments = self.config["experiments"]

        if self.config["top_hat_band"] is None:
            self.load_bandpass_from_file(likelihood)
        else:
            self.build_bandpass(likelihood)

        if self.config["beam_profile"] is None:
            self.init_beam_flat(likelihood)
        elif "beam_from_file" in self.config["beam_profile"]:
            self.init_beam_from_file(likelihood)

        self.parameters = []
        for exp in self.experiments:
            self.parameters.append(f"bandint_shift_{exp}")
        self.bp_index = jnp.array(
            [self.parameters.index(f"bandint_shift_{exp}")
             for exp in self.experiments])

        self.build_foreground_model(self.config["components"])

    def load_bandpass_from_file(self, likelihood):
        self.nu = [likelihood.tracers[exp + "_s0"]["nu"]
                   for exp in self.experiments]
        self.bp = [likelihood.tracers[exp + "_s0"]["bp"]
                   for exp in self.experiments]

    def build_bandpass(self, likelihood):
        self.nu = []
        self.bp = []

        nsteps = self.config["top_hat_band"]["nsteps"]
        bandwidth = self.config["top_hat_band"]["bandwidth"]
        if type(bandwidth) is not list:
            bandwidth = [bandwidth for _ in self.experiments]
        nu_mid = self.config["bandint_freqs"]

        for i, (num, bw) in enumerate(zip(nu_mid, bandwidth)):
            self.nu.append(np.linspace(num - bw / 2., num + bw / 2., nsteps))
            self.bp.append(np.ones((nsteps,)))

    def init_beam_flat(self, likelihood):
        for i, (exp, bp, nu) in enumerate(zip(self.experiments, self.bp,
                                              self.nu)):
            bp_beam = bp[:, None] * np.ones((1, len(self.ells)))

            self.nu[i] = jnp.array(nu)
            if len(nu) > 1:
                bp_beam = bp_beam / np.trapezoid(bp_beam, nu, axis=0)

            self.bp[i] = jnp.array(bp_beam)

    def init_beam_from_file(self, likelihood):
        for i, (exp, bp, nu) in enumerate(zip(self.experiments, self.bp,
                                              self.nu)):
            beam = likelihood.tracers[exp + "_s0"]["beam"]
            bp_beam = bp[:, None] * beam[:, self.ells]

            self.nu[i] = jnp.array(nu)
            if len(nu) > 1:
                bp_beam = bp_beam / np.trapezoid(bp_beam, nu, axis=0)

            self.bp[i] = jnp.array(bp_beam)

    def build_foreground_model(self, config):
        # TODO: cleanup this function >_<
        self.foreground_components = {cl: [] for cl in config}
        self.fg_indices = {cl: [] for cl in config}
        defaults = self.config["normalisation"]

        for cl in config:
            cl_indices = []

            for component in config[cl]:
                model_config = config[cl][component]
                mod_name = list(model_config.keys())[0]
                mod = getattr(fgm, mod_name)
                model_products = []
                for product in model_config[mod_name]:
                    tmpl_name = list(product.keys())[0]
                    settings = product[tmpl_name]
                    settings = defaults | (settings or {})
                    if hasattr(fgp, tmpl_name):
                        template = getattr(fgp, tmpl_name)
                    elif hasattr(fgf, tmpl_name):
                        template = getattr(fgf, tmpl_name)
                    else:
                        raise ImportError(f"""Failed to find {tmpl_name} among
                        Cl/SED templates. Check spelling?""")

                    model_products.append(template(**settings))
                model = mod(*model_products)

                n_req = model.n
                n_prov = len(model_config["params"])

                assert n_req == n_prov, f"""Configuration provided {n_prov}
                parameters for component {component}, but {n_req} are
                required."""

                self.foreground_components[cl].append(model)

                indices = []
                for param in model_config["params"]:
                    if param not in self.parameters:
                        self.parameters.append(param)
                    indices.append(self.parameters.index(param))
                cl_indices.append(jnp.array(indices))

            self.fg_indices[cl] = cl_indices

    def apply_bandpass_shifts(self, bandint_theta):
        nus = []
        bps = []

        for i, (exp, nu, bp) in enumerate(zip(self.experiments, self.nu,
                                              self.bp)):
            nub = nu + bandint_theta[i]
            nus.append(nub)
            if len(nub) > 1:
                nup = jnp.broadcast_to(_cmb2bb(nub)[:, None], bp.shape)
                bp_beam = bp * nup
                bp_beam = bp_beam / jnp.trapezoid(bp_beam, nub, axis=0)
            else:
                bp_beam = bp
            bps.append(bp_beam)

        return nus, bps

    def compute(self, theta, **kwargs):
        nu, bp = self.apply_bandpass_shifts(theta[self.bp_index])

        foregrounds = [jnp.zeros((*self.ells.shape, len(self.experiments),
                                  len(self.experiments)))
                       for _ in self.foreground_components]

        for i, cl in enumerate(self.foreground_components):
            for idx, fg in zip(self.fg_indices[cl],
                               self.foreground_components[cl]):
                theta_fg = theta[idx]
                foregrounds[i] = foregrounds[i] + fg(self.ells, nu, bp,
                                                     theta_fg)

        return {"foregrounds": jnp.stack(foregrounds)}

    @property
    def inputs(self):
        return ()

    @property
    def outputs(self):
        return ("foregrounds",)
