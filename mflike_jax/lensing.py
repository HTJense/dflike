import numpy as np
import sacc
import os
from functools import partial
import jax
import jax.numpy as jnp
from cobaya.yaml import yaml_load_file
from cobaya.tools import resolve_packages_path


class Lensing_jax:
    def __init__(self, config: str | dict):
        if type(config) is str:
            self.config = yaml_load_file(config)
        elif type(config) is dict:
            self.config = config
        else:
            raise TypeError("Configuration should be a string (filename) or "
                            f"dictionary, but was given {type(config)}.")

        data_path = self.config["data_folder"]
        if not os.path.isdir(data_path):
            data_path = os.path.join(resolve_packages_path(), "data",
                                     self.config["data_folder"])
        data = sacc.Sacc.load_fits(os.path.join(data_path,
                                                self.config["data_file"]))

        self.parameters = []
        _, cl, ind = data.get_ell_cl("cl_00", "ck", "ck", return_cov=False,
                                     return_ind=True)
        self.data_vec = jnp.array(cl)
        bpw = data.get_bandpower_windows(ind)
        self.ells = jnp.array(bpw.values)
        self.lmax = int(self.ells.max())
        self.binning_matrix = jnp.array(bpw.weight.T)
        self.covmat = jnp.array(data.covariance.covmat[:, :])
        self.inv_cov = jnp.linalg.inv(self.covmat)

    @partial(jax.jit, static_argnums=(0,))
    def bin_spectra(self, dlkk):
        model_vec = self.binning_matrix @ dlkk
        return model_vec

    @partial(jax.jit, static_argnums=(0,))
    def get_unbinned_model(self, dlpp, corrections, theta):
        dlkk = 2. * np.pi * dlpp[self.ells] / 4.
        return dlkk + corrections

    @partial(jax.jit, static_argnums=(0,))
    def get_model(self, dlpp, corrections, theta):
        model = self.get_unbinned_model(dlpp, corrections, theta)
        return self.bin_spectra(model)

    @partial(jax.jit, static_argnums=(0,))
    def chisquare(self, dlpp, corrections, theta):
        model_vec = self.get_model(dlpp, corrections, theta)
        delta = model_vec - self.data_vec
        chi2 = delta @ self.inv_cov @ delta

        return chi2


def get_cobaya_class():
    """
        This function allow one to import the lensing likelihood into cobaya
        using as `mflike_jax.lensing`.
    """
    from .cobaya import Lensing_jax_cobaya

    return Lensing_jax_cobaya
