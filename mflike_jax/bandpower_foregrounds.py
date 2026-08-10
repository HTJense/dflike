import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from cobaya.yaml import yaml_load_file
from cobaya.tools import resolve_packages_path
from scipy import constants

from . import fg_model as fgm
from . import fg_power as fgp
from . import fg_sed as fgf

T_CMB = 2.72548

@jax.jit
def _cmb2bb(nu, T=T_CMB):
    x = nu * constants.h * 1e9 / (constants.k * T)
    return jnp.exp(x) * (nu * x / jnp.expm1(x)) ** 2.

class BandpowerForegrounds:
    def __init__(self, config, likelihood, lmax=9000):
        self.config = yaml_load_file(config)

        self.ells = likelihood.ells
        self.experiments = self.config["experiments"]
        self.nu = [ jnp.array(likelihood.tracers[exp + "_s0"]["nu"]) for exp in self.experiments ]
        self.bp = [ jnp.array(likelihood.tracers[exp + "_s0"]["bp"] / np.trapezoid(likelihood.tracers[exp + "_s0"]["bp"], likelihood.tracers[exp + "_s0"]["nu"])) for exp in self.experiments ]

        self.parameters = [
        ]
        for exp in self.experiments:
            self.parameters.append(f"bandint_shift_{exp}")
        self.bp_index = jnp.array([ self.parameters.index(f"bandint_shift_{exp}") for exp in self.experiments ])

        self.build_foreground_model(self.config["components"])

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
                param_names = model_config["params"]
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
                        raise ImportError(f"Failed to find {tmpl_name} amongs Cl/SED templates. Check spelling?")

                    model_products.append( template(**settings) )
                model = mod(*model_products)

                n_req = model.n
                n_prov = len(model_config["params"])

                assert n_req == n_prov, f"Configuration provided {n_prov} parameters for component {component}, but {n_req} are required."

                self.foreground_components[cl].append(model)

                indices = []
                for param in model_config["params"]:
                    if param not in self.parameters:
                        self.parameters.append(param)
                    indices.append(self.parameters.index(param))
                cl_indices.append(jnp.array(indices))

            self.fg_indices[cl] = cl_indices

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

        foregrounds = [ jnp.zeros((*self.ells.shape, len(self.experiments), len(self.experiments))) for _ in self.foreground_components ]

        for i, cl in enumerate(self.foreground_components):
            for idx, fg in zip(self.fg_indices[cl], self.foreground_components[cl]):
                theta_fg = theta[idx]
                foregrounds[i] = foregrounds[i] + fg(self.ells, nu, bp, theta_fg)

        return jnp.stack(foregrounds)
